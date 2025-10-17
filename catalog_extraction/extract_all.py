import pandas as pd
from .data_loader import load_fashion_catalog
from .color_extractor import extract_colors, evaluate_color_extraction
from .gender_extractor import extract_gender, evaluate_gender_extraction
from .brand_extractor import extract_brands, evaluate_brand_extraction


def extract_all_attributes(df_products: pd.DataFrame) -> pd.DataFrame:
    print("Extracting colors...")
    df_products = extract_colors(df_products)
    
    print("Extracting genders...")
    df_products = extract_gender(df_products)
    
    print("Extracting brands...")
    df_products = extract_brands(df_products)
    
    return df_products


def evaluate_all_extractions(df_products: pd.DataFrame, df_validation: pd.DataFrame) -> dict:
    print("\nEvaluating color extraction...")
    color_results = evaluate_color_extraction(df_products, df_validation)
    
    print("Evaluating gender extraction...")
    gender_results = evaluate_gender_extraction(df_products, df_validation)
    
    print("Evaluating brand extraction...")
    brand_results = evaluate_brand_extraction(df_products, df_validation)
    
    return {
        'color': color_results,
        'gender': gender_results,
        'brand': brand_results
    }


if __name__ == "__main__":
    print("="*80)
    print("CATALOG ATTRIBUTE EXTRACTION - FULL PIPELINE")
    print("="*80)
    
    print("\nLoading dataset...")
    df_products, df_validation = load_fashion_catalog(
        data_path="data/myntra_products_catalog.csv",
        sample_size=1500
    )
    
    df_products = extract_all_attributes(df_products)
    
    print("\n" + "="*80)
    print("SAMPLE EXTRACTIONS")
    print("="*80)
    
    for idx in range(min(5, len(df_products))):
        row = df_products.iloc[idx]
        print(f"\nProduct ID: {row['product_id']}")
        print(f"Name: {row['name'][:70]}...")
        print(f"Extracted Colors: {row['all_colors']}")
        print(f"Extracted Genders: {row['all_genders']}")
        print(f"Extracted Brand: {row['extracted_brand']}")
        print("-" * 80)
    
    results = evaluate_all_extractions(df_products, df_validation)
    
    print("\n" + "="*80)
    print("OVERALL EVALUATION SUMMARY")
    print("="*80)
    
    print(f"\n{'Attribute':<15} {'Total':<10} {'Correct':<10} {'Wrong':<10} {'Accuracy':<10}")
    print("-" * 80)
    print(f"{'Color':<15} {results['color']['total']:<10} {results['color']['correct']:<10} {results['color']['wrong']:<10} {results['color']['accuracy']:.2%}")
    print(f"{'Gender':<15} {results['gender']['total']:<10} {results['gender']['correct']:<10} {results['gender']['wrong']:<10} {results['gender']['accuracy']:.2%}")
    print(f"{'Brand':<15} {results['brand']['total']:<10} {results['brand']['correct']:<10} {results['brand']['wrong']:<10} {results['brand']['accuracy']:.2%}")
    
    print("\n" + "="*80)
    print("COLOR EXTRACTION - Wrong Predictions (first 5):")
    print("="*80)
    df_wrong = results['color']['df_wrong']
    for idx in range(min(5, len(df_wrong))):
        row = df_wrong.iloc[idx]
        print(f"\nProduct ID: {row['product_id']}")
        print(f"Name: {row['name'][:60]}...")
        print(f"Ground Truth: {row['colour']}")
        print(f"Extracted: {row['all_colors']}")
    
    print("\n" + "="*80)
    print("GENDER EXTRACTION - Wrong Predictions (first 5):")
    print("="*80)
    df_wrong = results['gender']['df_wrong']
    for idx in range(min(5, len(df_wrong))):
        row = df_wrong.iloc[idx]
        print(f"\nProduct ID: {row['product_id']}")
        print(f"Name: {row['name'][:60]}...")
        print(f"Ground Truth: {row['gender']}")
        print(f"Extracted: {row['all_genders']}")
    
    print("\n" + "="*80)
    print("BRAND EXTRACTION - Wrong Predictions (first 5):")
    print("="*80)
    df_wrong = results['brand']['df_wrong']
    for idx in range(min(5, len(df_wrong))):
        row = df_wrong.iloc[idx]
        print(f"\nProduct ID: {row['product_id']}")
        print(f"Name: {row['name'][:60]}...")
        print(f"Ground Truth: {row['brand']}")
        print(f"Extracted: {row['extracted_brand']}")
    
    print("\n" + "="*80)
    print("EXTRACTION COMPLETE!")
    print("="*80)

