

import pandas as pd
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def clean_medicine_data(input_file=os.path.join(PROJECT_ROOT, 'data', 'raw', 'Medicine_Details.csv'),
                        output_file=os.path.join(PROJECT_ROOT, 'data', 'processed', 'cleaned_medicine_data.csv')):
    """
    Cleans the medicine dataset by:
    - Removing unnecessary columns (Image URL, Manufacturer)
    - Handling missing values
    - Standardizing column formats
    """
    print("Cleaning medicine data...")
    
    # Read the raw data
    try:
        med_data = pd.read_csv(input_file)
        print(f"Successfully loaded medicine data with {med_data.shape[0]} rows and {med_data.shape[1]} columns")
    except FileNotFoundError:
        print(f"Error: File {input_file} not found. Please check the file path.")
        return None
    
    # Display original columns
    print("Original columns:", med_data.columns.tolist())
    
    # Remove unnecessary columns
    cols_to_remove = ['Image URL', 'Manufacturer']
    med_data = med_data.drop(columns=cols_to_remove, errors='ignore')
    
    # Handle missing values
    med_data = med_data.fillna({'Uses': 'Unknown', 'Side_effects': 'None reported'})
    
    # Standardize column names
    med_data.columns = [col.strip().lower().replace(' ', '_') for col in med_data.columns]
    
    # Display cleaned data info
    print("Cleaned columns:", med_data.columns.tolist())
    print(f"Cleaned medicine data shape: {med_data.shape}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save cleaned data
    med_data.to_csv(output_file, index=False)
    print(f"Cleaned medicine data saved to {output_file}")
    
    return med_data

def clean_repurposing_data(input_file=os.path.join(PROJECT_ROOT, 'data', 'raw', 'Drug Discovery and Repurposing Tool.csv'),
                           output_file=os.path.join(PROJECT_ROOT, 'data', 'processed', 'cleaned_repurposing_data.csv')):
    """
    Cleans the drug repurposing dataset by:
    - Standardizing column formats
    - Converting confidence level to numerical format if needed
    """
    print("\nCleaning repurposing data...")
    
    # Read the raw data
    try:
        rep_data = pd.read_csv(input_file)
        print(f"Successfully loaded repurposing data with {rep_data.shape[0]} rows and {rep_data.shape[1]} columns")
    except FileNotFoundError:
        print(f"Error: File {input_file} not found. Please check the file path.")
        return None
    
    # Display original columns
    print("Original columns:", rep_data.columns.tolist())
    
    # Standardize column names
    rep_data.columns = [col.strip().lower().replace(' ', '_') for col in rep_data.columns]
    
    # Convert confidence level to numerical if not already
    if 'confidence_level' in rep_data.columns and rep_data['confidence_level'].dtype == 'object':
        # This assumes confidence is in format like "High", "Medium", "Low"
        confidence_map = {'high': 3, 'medium': 2, 'low': 1}
        rep_data['confidence_level_numeric'] = rep_data['confidence_level'].str.lower().map(confidence_map)
    
    # Display cleaned data info
    print("Cleaned columns:", rep_data.columns.tolist())
    print(f"Cleaned repurposing data shape: {rep_data.shape}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save cleaned data
    rep_data.to_csv(output_file, index=False)
    print(f"Cleaned repurposing data saved to {output_file}")
    
    return rep_data

def main():
    # Clean both datasets
    med_data = clean_medicine_data()
    rep_data = clean_repurposing_data()
    
    print("\nData cleaning complete!")
    
    # Return both datasets for potential further processing
    return med_data, rep_data

if __name__ == "__main__":
    main()