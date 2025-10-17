import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from typing import List
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

from .c_gender_extractor import extract_gender as extract_gender_regex


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


def predict_gender_by_similarity(product_name: str, tokenizer, model, gender_embeddings, threshold: float = 0.7) -> List[str]:
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
    
    if best_score < threshold:
        return []
    
    return [best_gender]


def extract_gender_hybrid(df_products: pd.DataFrame) -> pd.DataFrame:
    print("Extracting genders using HYBRID approach (Regex + Embedding fallback)...")
    
    df_regex = extract_gender_regex(df_products.copy())
    
    tokenizer, model = load_embedding_model()
    gender_embeddings = precompute_gender_embeddings(tokenizer, model)
    
    df_products['all_genders'] = df_regex['all_genders'].copy()
    
    empty_mask = df_products['all_genders'].apply(lambda x: len(x) == 0)
    empty_count = empty_mask.sum()
    
    print(f"Regex found genders for {len(df_products) - empty_count}/{len(df_products)} products")
    print(f"Applying embedding fallback to {empty_count} products...")
    
    for idx in df_products[empty_mask].index:
        name = df_products.loc[idx, 'name']
        embedding_genders = predict_gender_by_similarity(name, tokenizer, model, gender_embeddings, threshold=0.7)
        df_products.at[idx, 'all_genders'] = embedding_genders
    
    return df_products


def evaluate_gender_extraction(df_products: pd.DataFrame, df_validation: pd.DataFrame) -> dict:
    df_merged = df_products.merge(df_validation, on='product_id', how='inner')
    
    df_merged['is_correct'] = df_merged.apply(
        lambda row: check_gender_match(row['all_genders'], row['gender']), axis=1
    )
    
    valid_ground_truth = df_merged['gender'].notna()
    df_valid = df_merged[valid_ground_truth]
    
    correct = df_valid['is_correct'].sum()
    total = len(df_valid)
    wrong = total - correct
    
    has_extraction = df_products['all_genders'].apply(lambda x: len(x) > 0).sum()
    coverage = has_extraction / len(df_products) * 100
    
    df_wrong = df_valid[~df_valid['is_correct']]
    
    return {
        'correct': int(correct),
        'total': int(total),
        'wrong': int(wrong),
        'accuracy': correct / total if total > 0 else 0,
        'coverage': coverage,
        'df_wrong': df_wrong
    }


def check_gender_match(extracted_genders: List[str], ground_truth: str) -> bool:
    if pd.isna(ground_truth) or not extracted_genders:
        return False
    
    gt_lower = str(ground_truth).lower().strip()
    
    normalization_map = {
        'boys': ['boys', 'boy'],
        'girls': ['girls', 'girl'],
        'men': ['men', 'man', 'male'],
        'women': ['women', 'woman', 'female'],
        'unisex': ['unisex'],
        'unisex kids': ['unisex', 'kids', 'children', 'child']
    }
    
    for base_gender, variants in normalization_map.items():
        if gt_lower in variants or gt_lower == base_gender:
            for extracted in extracted_genders:
                if extracted.lower() in variants or extracted.lower() == base_gender:
                    return True
    
    return False


if __name__ == "__main__":
    from .a_data_loader import load_fashion_catalog
    from .c_gender_extractor import extract_gender as extract_gender_regex
    
    print("=" * 80)
    print("GENDER EXTRACTION: HYBRID (REGEX + EMBEDDING) COMPARISON")
    print("=" * 80)
    print()
    
    print("Loading dataset...")
    df_products, df_validation = load_fashion_catalog(sample_size=1500)
    print()
    
    print("Extracting with REGEX approach (baseline)...")
    print("-" * 80)
    df_regex = extract_gender_regex(df_products.copy())
    print()
    
    print("Extracting with HYBRID approach (Regex + Embedding fallback)...")
    print("-" * 80)
    df_hybrid = extract_gender_hybrid(df_products.copy())
    print()
    
    print("=" * 80)
    print("SAMPLE RESULTS COMPARISON (First 15 products)")
    print("=" * 80)
    print()
    
    for i in range(min(15, len(df_products))):
        row = df_products.iloc[i]
        validation_row = df_validation[df_validation['product_id'] == row['product_id']].iloc[0]
        
        print(f"Product {i+1}: {row['name'][:60]}...")
        print(f"  Ground Truth: {validation_row['gender']}")
        print(f"  Regex:        {df_regex.iloc[i]['all_genders']}")
        print(f"  Hybrid:       {df_hybrid.iloc[i]['all_genders']}")
        print("-" * 80)
        print()
    
    print("Evaluating REGEX approach...")
    metrics_regex = evaluate_gender_extraction(df_regex, df_validation)
    print()
    
    print("Evaluating HYBRID approach...")
    metrics_hybrid = evaluate_gender_extraction(df_hybrid, df_validation)
    print()
    
    print("=" * 80)
    print("ACCURACY COMPARISON")
    print("=" * 80)
    print()
    print(f"{'Approach':<25} {'Total':<10} {'Correct':<10} {'Wrong':<10} {'Accuracy':<10}")
    print("-" * 80)
    print(f"{'Regex (baseline)':<25} {metrics_regex['total']:<10} {metrics_regex['correct']:<10} {metrics_regex['wrong']:<10} {metrics_regex['accuracy']:.2f}%")
    print(f"{'Hybrid (Regex+Embed)':<25} {metrics_hybrid['total']:<10} {metrics_hybrid['correct']:<10} {metrics_hybrid['wrong']:<10} {metrics_hybrid['accuracy']:.2f}%")
    print()
    
    diff = metrics_hybrid['accuracy'] - metrics_regex['accuracy']
    if diff > 0:
        print(f"Hybrid is {diff:.2f}% better than Regex")
    else:
        print(f"Regex is {abs(diff):.2f}% better than Hybrid")
    
    print()
    print("=" * 80)
    print("COVERAGE COMPARISON")
    print("=" * 80)
    print(f"Regex Coverage:  {metrics_regex['coverage']:.2f}% ({int(metrics_regex['coverage'] * len(df_products) / 100)}/{len(df_products)} products)")
    print(f"Hybrid Coverage: {metrics_hybrid['coverage']:.2f}% ({int(metrics_hybrid['coverage'] * len(df_products) / 100)}/{len(df_products)} products)")
    print()
    
    print("=" * 80)
    print("HYBRID WRONG PREDICTIONS (showing first 20):")
    print("=" * 80)
    
    df_merged = df_hybrid.merge(df_validation, on='product_id', how='inner')
    df_merged['is_correct'] = df_merged.apply(
        lambda row: check_gender_match(row['all_genders'], row['gender']), axis=1
    )
    
    df_wrong = df_merged[~df_merged['is_correct'] & df_merged['gender'].notna()].head(20)
    
    for _, row in df_wrong.iterrows():
        print(f"\nProduct ID: {row['product_id']}")
        print(f"Name: {row['name'][:70]}...")
        print(f"Ground Truth: {row['gender']}")
        print(f"Hybrid Extracted: {row['all_genders']}")
        print("-" * 80)

