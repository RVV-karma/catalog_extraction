import pandas as pd
import spacy
from typing import Optional
from catalog_extraction.f_brand_extractor import extract_brand_from_name


def extract_brand_with_spacy(name: str, nlp) -> Optional[str]:
    doc = nlp(name)
    
    for ent in doc.ents:
        if ent.label_ == 'ORG':
            return ent.text
    
    return None


def extract_brand_hybrid(row, nlp) -> Optional[str]:
    regex_brand = extract_brand_from_name(
        row['name'],
        row.get('all_colors', []),
        row.get('all_genders', [])
    )
    
    if regex_brand is not None:
        return regex_brand
    
    spacy_brand = extract_brand_with_spacy(row['name'], nlp)
    return spacy_brand


def extract_brands(df_products: pd.DataFrame) -> pd.DataFrame:
    print("Loading spaCy transformer model for fallback...")
    nlp = spacy.load("en_core_web_trf")
    
    print("Extracting brands with HYBRID approach (Regex + spaCy fallback)...")
    df_products['extracted_brand_hybrid'] = df_products.apply(
        lambda row: extract_brand_hybrid(row, nlp),
        axis=1
    )
    
    return df_products


def evaluate_brand_extraction(df_products: pd.DataFrame, df_validation: pd.DataFrame) -> dict:
    df_merged = df_products.merge(df_validation, on='product_id', how='inner')
    
    df_merged = df_merged[df_merged['brand'].notna()].copy()
    
    df_merged['brand_lower'] = df_merged['brand'].str.lower().str.strip()
    df_merged['extracted_brand_lower'] = df_merged['extracted_brand_hybrid'].fillna('').str.lower().str.strip()
    
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
    import catalog_extraction.g_brand_extractor_spacy as spacy_extractor
    
    print("="*80)
    print("BRAND EXTRACTION: HYBRID APPROACH (Regex + spaCy Fallback)")
    print("="*80)
    
    print("\nLoading dataset...")
    df_products, df_validation = load_fashion_catalog(
        data_path="data/myntra_products_catalog.csv",
        sample_size=1500
    )
    
    print("\nExtracting colors and genders (needed for all approaches)...")
    print("-"*80)
    df_products = extract_colors(df_products)
    df_products = extract_gender_hybrid(df_products)
    
    print("\nExtracting with REGEX approach...")
    print("-"*80)
    df_products_regex = df_products.copy()
    df_products_regex = regex_extractor.extract_brands(df_products_regex)
    
    print("\nExtracting with SPACY TRF approach...")
    print("-"*80)
    df_products_spacy = df_products.copy()
    df_products_spacy = spacy_extractor.extract_brands(df_products_spacy)
    
    print("\nExtracting with HYBRID approach (Regex + spaCy fallback)...")
    print("-"*80)
    df_products_hybrid = df_products.copy()
    df_products_hybrid = extract_brands(df_products_hybrid)
    
    print("\n" + "="*80)
    print("SAMPLE RESULTS COMPARISON (First 15 products)")
    print("="*80)
    
    for idx in range(min(15, len(df_products))):
        row_regex = df_products_regex.iloc[idx]
        row_spacy = df_products_spacy.iloc[idx]
        row_hybrid = df_products_hybrid.iloc[idx]
        row_val = df_validation[df_validation['product_id'] == row_regex['product_id']].iloc[0]
        
        print(f"\nProduct {idx+1}: {row_regex['name'][:60]}...")
        print(f"  Ground Truth:  {row_val['brand']}")
        print(f"  Regex:         {row_regex['extracted_brand']}")
        print(f"  spaCy TRF:     {row_spacy['extracted_brand_spacy']}")
        print(f"  HYBRID:        {row_hybrid['extracted_brand_hybrid']}")
        print("-"*80)
    
    print("\nEvaluating REGEX approach...")
    results_regex = regex_extractor.evaluate_brand_extraction(df_products_regex, df_validation)
    
    print("\nEvaluating SPACY approach...")
    results_spacy = spacy_extractor.evaluate_brand_extraction(df_products_spacy, df_validation)
    
    print("\nEvaluating HYBRID approach...")
    results_hybrid = evaluate_brand_extraction(df_products_hybrid, df_validation)
    
    print("\n" + "="*80)
    print("ACCURACY COMPARISON")
    print("="*80)
    print(f"\n{'Approach':<25} {'Total':<10} {'Correct':<10} {'Wrong':<10} {'Accuracy':<15}")
    print("-"*80)
    print(f"{'Regex (baseline)':<25} {results_regex['total']:<10} {results_regex['correct']:<10} {results_regex['wrong']:<10} {results_regex['accuracy']:.2%}")
    print(f"{'spaCy TRF':<25} {results_spacy['total']:<10} {results_spacy['correct']:<10} {results_spacy['wrong']:<10} {results_spacy['accuracy']:.2%}")
    print(f"{'HYBRID (Regex+spaCy)':<25} {results_hybrid['total']:<10} {results_hybrid['correct']:<10} {results_hybrid['wrong']:<10} {results_hybrid['accuracy']:.2%}")
    
    improvement_over_regex = (results_hybrid['accuracy'] - results_regex['accuracy']) * 100
    improvement_over_spacy = (results_hybrid['accuracy'] - results_spacy['accuracy']) * 100
    
    print("\n" + "="*80)
    print("IMPROVEMENT ANALYSIS")
    print("="*80)
    if improvement_over_regex > 0:
        print(f"HYBRID is {improvement_over_regex:.2f}% better than Regex")
    elif improvement_over_regex < 0:
        print(f"Regex is {-improvement_over_regex:.2f}% better than HYBRID")
    else:
        print(f"HYBRID and Regex have equal accuracy")
    
    if improvement_over_spacy > 0:
        print(f"HYBRID is {improvement_over_spacy:.2f}% better than spaCy")
    
    print("\n" + "="*80)
    print("COVERAGE COMPARISON")
    print("="*80)
    
    regex_coverage = (df_products_regex['extracted_brand'].notna().sum() / len(df_products_regex)) * 100
    spacy_coverage = (df_products_spacy['extracted_brand_spacy'].notna().sum() / len(df_products_spacy)) * 100
    hybrid_coverage = (df_products_hybrid['extracted_brand_hybrid'].notna().sum() / len(df_products_hybrid)) * 100
    
    print(f"Regex Coverage:  {regex_coverage:.2f}% ({df_products_regex['extracted_brand'].notna().sum()}/{len(df_products_regex)} products)")
    print(f"spaCy Coverage:  {spacy_coverage:.2f}% ({df_products_spacy['extracted_brand_spacy'].notna().sum()}/{len(df_products_spacy)} products)")
    print(f"HYBRID Coverage: {hybrid_coverage:.2f}% ({df_products_hybrid['extracted_brand_hybrid'].notna().sum()}/{len(df_products_hybrid)} products)")
    
    print("\n" + "="*80)
    print("FALLBACK STATISTICS")
    print("="*80)
    
    regex_extracted = df_products_regex['extracted_brand'].notna().sum()
    regex_failed = len(df_products_regex) - regex_extracted
    
    print(f"Products where Regex succeeded: {regex_extracted} ({regex_coverage:.2f}%)")
    print(f"Products where Regex failed (would use spaCy): {regex_failed} ({100-regex_coverage:.2f}%)")
    
    print("\n" + "="*80)
    print("HYBRID WRONG PREDICTIONS (showing first 20):")
    print("="*80)
    
    df_wrong = results_hybrid['df_wrong']
    for idx in range(min(20, len(df_wrong))):
        row = df_wrong.iloc[idx]
        regex_val = df_products_regex[df_products_regex['product_id'] == row['product_id']]['extracted_brand'].iloc[0]
        spacy_val = df_products_spacy[df_products_spacy['product_id'] == row['product_id']]['extracted_brand_spacy'].iloc[0]
        
        print(f"\nProduct ID: {row['product_id']}")
        print(f"Name: {row['name'][:70]}...")
        print(f"Ground Truth: {row['brand']}")
        print(f"Regex: {regex_val}")
        print(f"spaCy: {spacy_val}")
        print(f"HYBRID: {row['extracted_brand_hybrid']}")
        print("-" * 80)

