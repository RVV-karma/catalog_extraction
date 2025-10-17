import pandas as pd
import spacy
from typing import Optional


def extract_brand_with_spacy(name: str, nlp) -> Optional[str]:
    doc = nlp(name)
    
    for ent in doc.ents:
        if ent.label_ == 'ORG':
            return ent.text
    
    return None


def extract_brands(df_products: pd.DataFrame) -> pd.DataFrame:
    print("Loading spaCy transformer model...")
    nlp = spacy.load("en_core_web_trf")
    
    print("Extracting brands with spaCy NER (transformer)...")
    df_products['extracted_brand_spacy'] = df_products['name'].apply(
        lambda x: extract_brand_with_spacy(x, nlp)
    )
    
    return df_products


def evaluate_brand_extraction(df_products: pd.DataFrame, df_validation: pd.DataFrame) -> dict:
    df_merged = df_products.merge(df_validation, on='product_id', how='inner')
    
    df_merged = df_merged[df_merged['brand'].notna()].copy()
    
    df_merged['brand_lower'] = df_merged['brand'].str.lower().str.strip()
    df_merged['extracted_brand_lower'] = df_merged['extracted_brand_spacy'].fillna('').str.lower().str.strip()
    
    df_merged['is_correct'] = df_merged['brand_lower'] == df_merged['extracted_brand_lower']
    
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
    from catalog_extraction.b_color_extractor import extract_colors
    from catalog_extraction.e_gender_extractor_hybrid import extract_gender_hybrid
    import catalog_extraction.f_brand_extractor as regex_extractor
    
    print("="*80)
    print("BRAND EXTRACTION: SPACY NER vs REGEX COMPARISON")
    print("="*80)
    
    print("\nLoading dataset...")
    df_products, df_validation = load_fashion_catalog(
        data_path="data/myntra_products_catalog.csv",
        sample_size=1500
    )
    
    print("\nExtracting with REGEX approach (baseline)...")
    print("-"*80)
    df_products_regex = df_products.copy()
    df_products_regex = extract_colors(df_products_regex)
    df_products_regex = extract_gender_hybrid(df_products_regex)
    df_products_regex = regex_extractor.extract_brands(df_products_regex)
    
    print("\nExtracting with SPACY NER approach...")
    print("-"*80)
    df_products_spacy = df_products.copy()
    df_products_spacy = extract_brands(df_products_spacy)
    
    print("\n" + "="*80)
    print("SAMPLE RESULTS COMPARISON (First 10 products)")
    print("="*80)
    
    for idx in range(min(10, len(df_products))):
        row_regex = df_products_regex.iloc[idx]
        row_spacy = df_products_spacy.iloc[idx]
        row_val = df_validation[df_validation['product_id'] == row_regex['product_id']].iloc[0]
        
        print(f"\nProduct {idx+1}: {row_regex['name'][:60]}...")
        print(f"  Ground Truth:  {row_val['brand']}")
        print(f"  Regex Extract: {row_regex['extracted_brand']}")
        print(f"  spaCy Extract: {row_spacy['extracted_brand_spacy']}")
        print("-"*80)
    
    print("\nEvaluating REGEX approach...")
    results_regex = regex_extractor.evaluate_brand_extraction(df_products_regex, df_validation)
    
    print("\nEvaluating SPACY approach...")
    results_spacy = evaluate_brand_extraction(df_products_spacy, df_validation)
    
    print("\n" + "="*80)
    print("ACCURACY COMPARISON")
    print("="*80)
    print(f"\n{'Approach':<20} {'Total':<10} {'Correct':<10} {'Wrong':<10} {'Accuracy':<15}")
    print("-"*80)
    print(f"{'Regex (baseline)':<20} {results_regex['total']:<10} {results_regex['correct']:<10} {results_regex['wrong']:<10} {results_regex['accuracy']:.2%}")
    print(f"{'spaCy NER':<20} {results_spacy['total']:<10} {results_spacy['correct']:<10} {results_spacy['wrong']:<10} {results_spacy['accuracy']:.2%}")
    
    improvement = (results_spacy['accuracy'] - results_regex['accuracy']) * 100
    if improvement > 0:
        print(f"\nspaCy is {improvement:.2f}% better than Regex")
    elif improvement < 0:
        print(f"\nRegex is {-improvement:.2f}% better than spaCy")
    else:
        print(f"\nBoth approaches have equal accuracy")
    
    print("\n" + "="*80)
    print("SPACY WRONG PREDICTIONS (showing first 20):")
    print("="*80)
    
    df_wrong = results_spacy['df_wrong']
    for idx in range(min(20, len(df_wrong))):
        row = df_wrong.iloc[idx]
        print(f"\nProduct ID: {row['product_id']}")
        print(f"Name: {row['name'][:70]}...")
        print(f"Ground Truth: {row['brand']}")
        print(f"spaCy Extracted: {row['extracted_brand_spacy']}")
        print("-" * 80)
    
    print("\n" + "="*80)
    print("COVERAGE COMPARISON")
    print("="*80)
    
    regex_coverage = (df_products_regex['extracted_brand'].notna().sum() / len(df_products_regex)) * 100
    spacy_coverage = (df_products_spacy['extracted_brand_spacy'].notna().sum() / len(df_products_spacy)) * 100
    
    print(f"Regex Coverage: {regex_coverage:.2f}% ({df_products_regex['extracted_brand'].notna().sum()}/{len(df_products_regex)} products)")
    print(f"spaCy Coverage: {spacy_coverage:.2f}% ({df_products_spacy['extracted_brand_spacy'].notna().sum()}/{len(df_products_spacy)} products)")
