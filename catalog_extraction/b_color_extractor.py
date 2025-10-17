import pandas as pd
from typing import List, Set, Dict


def load_color_names(color_data_path: str = "data/color_names.csv") -> Set[str]:
    df_colors = pd.read_csv(color_data_path)
    color_names = df_colors['Name'].str.replace(r'\([^)]*\)', '', regex=True).str.strip().str.lower().tolist()
    color_set = set(color_names)
    color_set.add('grey')
    return color_set


def load_color_data(color_data_path: str = "data/color_names.csv") -> pd.DataFrame:
    return pd.read_csv(color_data_path)


def normalize_color_by_hsl(color_name: str, df_colors: pd.DataFrame) -> str:
    color_row = df_colors[df_colors['Name'].str.lower() == color_name.lower()]
    
    if color_row.empty:
        clean_name = color_name.split('(')[0].strip().lower()
        color_row = df_colors[df_colors['Name'].str.lower() == clean_name]
        if color_row.empty:
            return color_name.lower()
    
    color_row = color_row.iloc[0]
    
    hue = color_row['Hue (degrees)']
    saturation = color_row['HSL.S (%)']
    lightness = color_row['HSL.L (%), HSV.S (%), HSV.V (%)']
    
    if saturation < 10:
        if lightness < 20:
            return 'black'
        elif lightness > 85:
            return 'white'
        else:
            return 'grey'
    
    if (30 <= hue <= 60) and (20 <= lightness <= 50):
        return 'brown'
    
    if hue < 30 or hue >= 330:
        return 'red'
    elif 30 <= hue < 45:
        if lightness < 40 or saturation < 40:
            return 'brown'
        return 'orange'
    elif 45 <= hue < 75:
        return 'yellow'
    elif 75 <= hue < 90:
        return 'orange'
    elif 90 <= hue < 150:
        return 'green'
    elif 150 <= hue < 270:
        return 'blue'
    elif 270 <= hue < 300:
        return 'purple'
    elif 300 <= hue < 330:
        if lightness > 60:
            return 'pink'
        return 'purple'
    
    return color_name.lower()


def create_normalization_map(df_colors: pd.DataFrame) -> Dict[str, str]:
    normalization_map = {}
    for idx, row in df_colors.iterrows():
        color_name = row['Name']
        clean_name = color_name.split('(')[0].strip().lower()
        normalized = normalize_color_by_hsl(color_name, df_colors)
        normalization_map[clean_name] = normalized
    
    normalization_map['grey'] = normalization_map.get('gray', 'grey')
    return normalization_map


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
    
    df_colors = load_color_data(color_data_path)
    normalization_map = create_normalization_map(df_colors)
    
    df_products['all_normalized_colors'] = df_products['all_colors'].apply(
        lambda colors: list(set([normalization_map.get(c.lower(), c.lower()) for c in colors]))
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
    
    from catalog_extraction.a_data_loader import load_fashion_catalog
    
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
        print(f"Normalized colors: {row['all_normalized_colors']}")
        print("-" * 80)
    
    all_extracted_colors = []
    all_normalized_colors = []
    for colors in df_products['all_colors']:
        all_extracted_colors.extend(colors)
    for colors in df_products['all_normalized_colors']:
        all_normalized_colors.extend(colors)
    
    print("\n" + "=" * 80)
    print("COLOR NORMALIZATION SUMMARY")
    print("=" * 80)
    print(f"Unique extracted colors: {len(set(all_extracted_colors))}")
    print(f"Unique normalized colors: {len(set(all_normalized_colors))}")
    print(f"\nNormalized colors list: {sorted(set(all_normalized_colors))}")
    
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
