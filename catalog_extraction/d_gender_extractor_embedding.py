import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from typing import List
from torch import Tensor
from transformers import AutoTokenizer, AutoModel


GENDER_TERMS = {
    'men': ['men', 'mens', 'man', 'male', 'males', 'boys', 'boy'],
    'women': ['women', 'womens', 'woman', 'female', 'females', 'girls', 'girl'],
    'unisex': ['unisex', 'kids', 'children', 'child']
}


def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'


def load_embedding_model():
    print("Loading multilingual E5 instruct model...")
    tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large-instruct')
    model = AutoModel.from_pretrained('intfloat/multilingual-e5-large-instruct')
    return tokenizer, model


def precompute_gender_embeddings(tokenizer, model):
    print("Computing embeddings for gender terms...")
    
    gender_embeddings = {}
    for gender, terms in GENDER_TERMS.items():
        batch_dict = tokenizer(terms, max_length=512, padding=True, truncation=True, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model(**batch_dict)
            embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        gender_embeddings[gender] = embeddings.cpu().numpy()
    
    return gender_embeddings


def predict_gender_by_similarity(product_name: str, tokenizer, model, gender_embeddings) -> List[str]:
    if pd.isna(product_name):
        return []
    
    task = 'Classify the product as men, women, or unisex based on the product name'
    instructed_query = get_detailed_instruct(task, product_name)
    
    batch_dict = tokenizer([instructed_query], max_length=512, padding=True, truncation=True, return_tensors='pt')
    
    with torch.no_grad():
        outputs = model(**batch_dict)
        prod_emb = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        prod_emb = F.normalize(prod_emb, p=2, dim=1)
    
    prod_emb = prod_emb.cpu().numpy()[0]
    
    max_similarities = {}
    for gender, term_embeddings in gender_embeddings.items():
        similarities = np.dot(term_embeddings, prod_emb)
        max_similarities[gender] = np.max(similarities)
    
    best_gender = max(max_similarities, key=max_similarities.get)
    best_score = max_similarities[best_gender]
    
    if best_score < 0.3:
        return []
    
    return [best_gender]


def extract_gender(df_products: pd.DataFrame) -> pd.DataFrame:
    tokenizer, model = load_embedding_model()
    gender_embeddings = precompute_gender_embeddings(tokenizer, model)
    
    print("Extracting genders using embedding similarity with instructions...")
    df_products['all_genders'] = df_products['name'].apply(
        lambda x: predict_gender_by_similarity(x, tokenizer, model, gender_embeddings)
    )
    
    return df_products


def evaluate_gender_extraction(df_products: pd.DataFrame, df_validation: pd.DataFrame) -> dict:
    df_merged = df_products.merge(df_validation, on='product_id', how='inner')
    
    df_merged = df_merged[df_merged['gender'].notna()].copy()
    
    df_merged['gender_lower'] = df_merged['gender'].str.lower().str.strip()
    
    df_merged['is_correct'] = df_merged.apply(
        lambda row: row['gender_lower'] in row['all_genders'],
        axis=1
    )
    
    total = len(df_merged)
    correct = df_merged['is_correct'].sum()
    accuracy = correct / total if total > 0 else 0
    
    df_wrong = df_merged[~df_merged['is_correct']].copy()
    
    results = {
        'total': total,
        'correct': correct,
        'wrong': len(df_wrong),
        'accuracy': accuracy,
        'df_wrong': df_wrong
    }
    
    return results


if __name__ == "__main__":
    
    from catalog_extraction.a_data_loader import load_fashion_catalog
    import catalog_extraction.c_gender_extractor as regex_extractor
    
    print("="*80)
    print("GENDER EXTRACTION: EMBEDDING SIMILARITY vs REGEX COMPARISON")
    print("="*80)
    
    print("\nLoading dataset...")
    df_products, df_validation = load_fashion_catalog(
        data_path="data/myntra_products_catalog.csv",
        sample_size=1500
    )
    
    print("\nExtracting with REGEX approach (baseline)...")
    print("-"*80)
    df_products_regex = df_products.copy()
    df_products_regex = regex_extractor.extract_gender(df_products_regex)
    
    print("\nExtracting with EMBEDDING SIMILARITY approach...")
    print("-"*80)
    df_products_embedding = df_products.copy()
    df_products_embedding = extract_gender(df_products_embedding)
    
    print("\n" + "="*80)
    print("SAMPLE RESULTS COMPARISON (First 15 products)")
    print("="*80)
    
    for idx in range(min(15, len(df_products))):
        row_regex = df_products_regex.iloc[idx]
        row_emb = df_products_embedding.iloc[idx]
        row_val = df_validation[df_validation['product_id'] == row_regex['product_id']].iloc[0]
        
        print(f"\nProduct {idx+1}: {row_regex['name'][:60]}...")
        print(f"  Ground Truth: {row_val['gender']}")
        print(f"  Regex:        {row_regex['all_genders']}")
        print(f"  Embedding:    {row_emb['all_genders']}")
        print("-"*80)
    
    print("\nEvaluating REGEX approach...")
    results_regex = regex_extractor.evaluate_gender_extraction(df_products_regex, df_validation)
    
    print("\nEvaluating EMBEDDING approach...")
    results_embedding = evaluate_gender_extraction(df_products_embedding, df_validation)
    
    print("\n" + "="*80)
    print("ACCURACY COMPARISON")
    print("="*80)
    print(f"\n{'Approach':<25} {'Total':<10} {'Correct':<10} {'Wrong':<10} {'Accuracy':<15}")
    print("-"*80)
    print(f"{'Regex (baseline)':<25} {results_regex['total']:<10} {results_regex['correct']:<10} {results_regex['wrong']:<10} {results_regex['accuracy']:.2%}")
    print(f"{'Embedding Similarity':<25} {results_embedding['total']:<10} {results_embedding['correct']:<10} {results_embedding['wrong']:<10} {results_embedding['accuracy']:.2%}")
    
    improvement = (results_embedding['accuracy'] - results_regex['accuracy']) * 100
    if improvement > 0:
        print(f"\nEmbedding is {improvement:.2f}% better than Regex")
    elif improvement < 0:
        print(f"\nRegex is {-improvement:.2f}% better than Embedding")
    else:
        print(f"\nBoth approaches have equal accuracy")
    
    print("\n" + "="*80)
    print("COVERAGE COMPARISON")
    print("="*80)
    
    regex_coverage = (df_products_regex['all_genders'].apply(len).gt(0).sum() / len(df_products_regex)) * 100
    embedding_coverage = (df_products_embedding['all_genders'].apply(len).gt(0).sum() / len(df_products_embedding)) * 100
    
    print(f"Regex Coverage:     {regex_coverage:.2f}% ({df_products_regex['all_genders'].apply(len).gt(0).sum()}/{len(df_products_regex)} products)")
    print(f"Embedding Coverage: {embedding_coverage:.2f}% ({df_products_embedding['all_genders'].apply(len).gt(0).sum()}/{len(df_products_embedding)} products)")
    
    print("\n" + "="*80)
    print("EMBEDDING WRONG PREDICTIONS (showing first 20):")
    print("="*80)
    
    df_wrong = results_embedding['df_wrong']
    for idx in range(min(20, len(df_wrong))):
        row = df_wrong.iloc[idx]
        print(f"\nProduct ID: {row['product_id']}")
        print(f"Name: {row['name'][:70]}...")
        print(f"Ground Truth: {row['gender']}")
        print(f"Embedding Extracted: {row['all_genders']}")
        print("-" * 80)

