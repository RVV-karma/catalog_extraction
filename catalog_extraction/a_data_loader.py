import pandas as pd
from typing import Tuple, Optional


def load_fashion_catalog(
    data_path: str = "data/myntra_products_catalog.csv",
    sample_size: Optional[int] = None,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    cols = ['ProductID', 'ProductName', 'ProductBrand', 'Gender', 'Description', 'PrimaryColor']
    df = pd.read_csv(data_path, usecols=cols)
    df = df.dropna(subset=['ProductName', 'Description']).copy()
    
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=random_state)
    
    df = df.reset_index(drop=True)
    
    df_products = pd.DataFrame({
        'product_id': df['ProductID'],
        'name': df['ProductName'],
        'description': df['Description']
    })
    
    df_validation = pd.DataFrame({
        'product_id': df['ProductID'],
        'brand': df['ProductBrand'],
        'gender': df['Gender'],
        'colour': df['PrimaryColor']
    })
    
    return df_products, df_validation


if __name__ == "__main__":
    
    try:
        df_products, df_validation = load_fashion_catalog(
            data_path="data/myntra_products_catalog.csv",
            sample_size=1500
        )
        
        print("\n" + "="*60)
        print("DATA LOADING SUMMARY")
        print("="*60)
        print(f"Total products: {len(df_products)}")
        print(f"Non-null brands: {df_validation['brand'].notna().sum()}")
        print(f"Non-null colours: {df_validation['colour'].notna().sum()}")
        
        if df_validation['brand'].notna().sum() > 0:
            print(f"\nTop 5 brands: {df_validation['brand'].value_counts().head(5).to_dict()}")
        
        if df_validation['colour'].notna().sum() > 0:
            print(f"\nTop 5 colours: {df_validation['colour'].value_counts().head(5).to_dict()}")
        
        print("="*60)
        
        print("\nSample Products:")
        print(df_products.head())
        
        print("\nSample Validation Data:")
        print(df_validation.head())
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
