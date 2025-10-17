import pandas as pd
import re
from typing import Optional, List


def extract_brand_from_name(name: str, extracted_colors: List[str], extracted_genders: List[str]) -> Optional[str]:
    if extracted_genders:
        gender_pattern = r'\b(' + '|'.join(re.escape(term) for term in extracted_genders) + r')\b'
        match = re.search(gender_pattern, name, re.IGNORECASE)
        
        if match:
            brand = name[:match.start()].strip()
            return brand if brand else None
    
    words = name.split()
    brand_end_idx = None
    
    for idx, word in enumerate(words):
        word_lower = word.lower().strip('.,;:!?')
        
        sub_words = word_lower.replace('-', ' ').split()
        
        if extracted_colors:
            colors_lower = [c.lower() for c in extracted_colors]
            for sub_word in sub_words:
                if sub_word in colors_lower:
                    brand_end_idx = idx
                    break
            
            if brand_end_idx is not None:
                break
        
        word_clean = word_lower.strip('-')
        if word_clean in ['pack', 'set'] and idx > 0:
            brand_end_idx = idx
            break
    
    if brand_end_idx is not None and brand_end_idx > 0:
        brand = ' '.join(words[:brand_end_idx])
        return brand.strip() if brand else None
    
    return None


def extract_brands(df_products: pd.DataFrame) -> pd.DataFrame:
    df_products['extracted_brand'] = df_products.apply(
        lambda row: extract_brand_from_name(row['name'], row.get('all_colors', []), row.get('all_genders', [])),
        axis=1
    )
    
    return df_products


def evaluate_brand_extraction(df_products: pd.DataFrame, df_validation: pd.DataFrame) -> dict:
    df_merged = df_products.merge(df_validation, on='product_id', how='inner')
    
    df_merged = df_merged[df_merged['brand'].notna()].copy()
    
    df_merged['brand_lower'] = df_merged['brand'].str.lower().str.strip()
    df_merged['extracted_brand_lower'] = df_merged['extracted_brand'].fillna('').str.lower().str.strip()
    
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
    
    from catalog_extraction.data_loader import load_fashion_catalog
    from catalog_extraction.color_extractor import extract_colors
    from catalog_extraction.gender_extractor import extract_gender
    
    print("Loading dataset...")
    df_products, df_validation = load_fashion_catalog(
        data_path="data/myntra_products_catalog.csv",
        sample_size=1500
    )
    
    print("Extracting colors...")
    df_products = extract_colors(df_products)
    
    print("Extracting genders...")
    df_products = extract_gender(df_products)
    
    print("\nExtracting brands...")
    df_products = extract_brands(df_products)
    
    print("\nSample Results:")
    print("=" * 80)
    for idx in range(min(10, len(df_products))):
        row = df_products.iloc[idx]
        print(f"\nProduct ID: {row['product_id']}")
        print(f"Name: {row['name'][:70]}...")
        print(f"Extracted Brand: {row['extracted_brand']}")
        print("-" * 80)
    
    print("\nEvaluating accuracy...")
    results = evaluate_brand_extraction(df_products, df_validation)
    
    print("\n" + "=" * 80)
    print("BRAND EXTRACTION EVALUATION")
    print("=" * 80)
    print(f"Total products: {results['total']}")
    print(f"Correct extractions: {results['correct']}")
    print(f"Wrong extractions: {results['wrong']}")
    print(f"Accuracy: {results['accuracy']:.2%}")
    
    if results['wrong'] > 0:
        print("\n" + "=" * 80)
        print(f"WRONG PREDICTIONS (showing first 20):")
        print("=" * 80)
        
        df_wrong = results['df_wrong']
        for idx in range(min(20, len(df_wrong))):
            row = df_wrong.iloc[idx]
            print(f"\nProduct ID: {row['product_id']}")
            print(f"Name: {row['name'][:70]}...")
            print(f"Ground Truth Brand: {row['brand']}")
            print(f"Extracted Brand: {row['extracted_brand']}")
            print("-" * 80)
