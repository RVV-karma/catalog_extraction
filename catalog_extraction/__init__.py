from .a_data_loader import load_fashion_catalog
from .b_color_extractor import extract_colors, evaluate_color_extraction
from .c_gender_extractor import extract_gender as extract_gender_regex
from .i_gender_extractor_hybrid import extract_gender_hybrid as extract_gender, evaluate_gender_extraction
from .f_brand_extractor_hybrid import extract_brands, evaluate_brand_extraction
from .g_extract_all import extract_all_attributes, evaluate_all_extractions

__version__ = "1.0.0"
__all__ = ['load_fashion_catalog', 'extract_colors', 'evaluate_color_extraction', 'extract_gender', 'extract_gender_regex', 'evaluate_gender_extraction', 'extract_brands', 'evaluate_brand_extraction', 'extract_all_attributes', 'evaluate_all_extractions']
