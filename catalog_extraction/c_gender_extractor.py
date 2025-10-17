import pandas as pd
from typing import List, Set


def load_gender_terms() -> Set[str]:
    gender_terms = [
        'men', 'women', 'boys', 'girls', 'kids', 'unisex',
        'male', 'female', 'boy', 'girl', 'child', 'children',
        'mens', 'womens', 'man', 'woman', 'males', 'females'
    ]
    return set(gender_terms)


def extract_gender_from_text(text: str, gender_terms: Set[str]) -> List[str]:
    if pd.isna(text):
        return []
    
    import re
    text_lower = text.lower()
    found_genders = []
    
    for gender in sorted(gender_terms, key=len, reverse=True):
        if re.search(r'\b' + re.escape(gender) + r'\b', text_lower):
            found_genders.append(gender)
    
    return found_genders


def normalize_gender(gender: str) -> str:
    normalization_map = {
        'man': 'men',
        'woman': 'women',
        'boy': 'boys',
        'girl': 'girls',
        'male': 'men',
        'female': 'women',
        'males': 'men',
        'females': 'women',
        'mens': 'men',
        'womens': 'women',
        'child': 'kids',
        'children': 'kids'
    }
    return normalization_map.get(gender, gender)


def extract_gender(df_products: pd.DataFrame) -> pd.DataFrame:
    gender_terms = load_gender_terms()
    print(f"Loaded {len(gender_terms)} gender terms")
    
    df_products['genders_from_name'] = df_products['name'].apply(
        lambda x: extract_gender_from_text(x, gender_terms)
    )
    
    df_products['genders_from_description'] = df_products['description'].apply(
        lambda x: extract_gender_from_text(x, gender_terms)
    )
    
    df_products['all_genders'] = df_products.apply(
        lambda row: list(set([normalize_gender(g) for g in row['genders_from_name'] + row['genders_from_description']])),
        axis=1
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
    
    print("Loading dataset...")
    df_products, df_validation = load_fashion_catalog(
        data_path="data/myntra_products_catalog.csv",
        sample_size=1500
    )
    
    print("\nExtracting genders...")
    df_products = extract_gender(df_products)
    
    print("\nSample Results:")
    print("=" * 80)
    for idx in range(min(5, len(df_products))):
        row = df_products.iloc[idx]
        print(f"\nProduct ID: {row['product_id']}")
        print(f"Name: {row['name'][:60]}...")
        print(f"Genders from name: {row['genders_from_name']}")
        print(f"Genders from description: {row['genders_from_description']}")
        print(f"All genders: {row['all_genders']}")
        print("-" * 80)
    
    print("\nEvaluating accuracy...")
    results = evaluate_gender_extraction(df_products, df_validation)
    
    print("\n" + "=" * 80)
    print("GENDER EXTRACTION EVALUATION")
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
            print(f"Name: {row['name'][:60]}...")
            print(f"Description: {row['description'][:100]}...")
            print(f"Ground Truth Gender: {row['gender']}")
            print(f"Extracted Genders: {row['all_genders']}")
            print("-" * 80)

