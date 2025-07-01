import os
from data_loader import load_and_clean_data
from features import create_features

def run_pipeline():
    """
    Runs the full data preprocessing pipeline.
    """
    # Get the directory of the current script to build absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # Define file paths
    raw_data_path = os.path.join(project_root, 'data', 'raw', 'Orders_Master_Data(in).xlsx')
    cleaned_data_path = os.path.join(project_root, 'data', 'interim', 'orders_cleaned.csv')
    processed_data_dir = os.path.join(project_root, 'data', 'processed')

    # Step 1: Load and clean data
    print("Starting data loading and cleaning...")
    load_and_clean_data(raw_data_path, cleaned_data_path)
    print("Data loading and cleaning complete.")

    # Step 2: Create features
    print("Starting feature creation...")
    create_features(cleaned_data_path, processed_data_dir)
    print("Feature creation complete.")

if __name__ == "__main__":
    run_pipeline()