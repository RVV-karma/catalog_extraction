from .data_loader import load_fashion_catalog
from .color_extractor import extract_colors, evaluate_color_extraction
from .gender_extractor import extract_gender, evaluate_gender_extraction
from .brand_extractor import extract_brands, evaluate_brand_extraction
from .extract_all import extract_all_attributes, evaluate_all_extractions

__version__ = "1.0.0"
__all__ = ['load_fashion_catalog', 'extract_colors', 'evaluate_color_extraction', 'extract_gender', 'evaluate_gender_extraction', 'extract_brands', 'evaluate_brand_extraction', 'extract_all_attributes', 'evaluate_all_extractions']
