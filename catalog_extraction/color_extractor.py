import pandas as pd
from typing import List, Set


def load_color_names(color_data_path: str = "data/color_names.csv") -> Set[str]:
    df_colors = pd.read_csv(color_data_path)
    color_names = df_colors['Name'].str.replace(r'\([^)]*\)', '', regex=True).str.strip().str.lower().tolist()
    color_set = set(color_names)
    color_set.add('grey')  # Add British spelling (dataset has 'gray')
    return color_set


def extract_colors_from_text(text: str, color_names: Set[str]) -> List[str]:
    if pd.isna(text):
        return []
    
    text_lower = text.lower()
    found_colors = []
    
    for color in color_names:
        if color in text_lower:
            found_colors.append(color)
    
    return found_colors


def extract_colors(df_products: pd.DataFrame, color_data_path: str = "data/color_names.csv") -> pd.DataFrame:
    color_names = load_color_names(color_data_path)
    print(f"Loaded {len(color_names)} color names")
    
    df_products['colors_from_name'] = df_products['name'].apply(
        lambda x: extract_colors_from_text(x, color_names)
    )
    
    df_products['colors_from_description'] = df_products['description'].apply(
        lambda x: extract_colors_from_text(x, color_names)
    )
    
    df_products['all_colors'] = df_products.apply(
        lambda row: list(set(row['colors_from_name'] + row['colors_from_description'])),
        axis=1
    )
    
    return df_products


def evaluate_color_extraction(df_products: pd.DataFrame, df_validation: pd.DataFrame) -> dict:
    df_merged = df_products.merge(df_validation, on='product_id', how='inner')
    
    df_merged = df_merged[df_merged['colour'].notna()].copy()
    
    df_merged['colour_lower'] = df_merged['colour'].str.lower().str.strip()
    
    df_merged['is_correct'] = df_merged.apply(
        lambda row: row['colour_lower'] in row['all_colors'],
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
    
    from catalog_extraction.data_loader import load_fashion_catalog
    
    print("Loading dataset...")
    df_products, df_validation = load_fashion_catalog(
        data_path="data/myntra_products_catalog.csv",
        sample_size=1500
    )
    
    print("\nExtracting colors...")
    df_products = extract_colors(df_products, color_data_path="data/color_names.csv")
    
    print("\nSample Results:")
    print("=" * 80)
    for idx in range(min(5, len(df_products))):
        row = df_products.iloc[idx]
        print(f"\nProduct ID: {row['product_id']}")
        print(f"Name: {row['name'][:60]}...")
        print(f"Colors from name: {row['colors_from_name']}")
        print(f"Colors from description: {row['colors_from_description']}")
        print(f"All colors: {row['all_colors']}")
        print("-" * 80)
    
    print("\nEvaluating accuracy...")
    results = evaluate_color_extraction(df_products, df_validation)
    
    print("\n" + "=" * 80)
    print("COLOR EXTRACTION EVALUATION")
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
            print(f"Ground Truth Color: {row['colour']}")
            print(f"Extracted Colors: {row['all_colors']}")
            print("-" * 80)
