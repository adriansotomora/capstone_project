import os
import pandas as pd
import numpy as np

def load_and_clean_data(input_path, output_path):
    """
    Loads the raw data, cleans it, and saves it to an intermediate file.
    """
    # Load data
    raw_data = pd.read_excel(input_path)

    # Clean column names
    df = raw_data.copy()
    df.columns = df.columns.str.replace(' ', '_')
    df.columns = df.columns.str.lower()
    df = df.drop_duplicates()

    # Correct city names
    df['city'] = df['city'].replace({
        'C√°diz': 'Cadiz',
        'Castell√≥n': 'Castellon',
        'C√≥rdoba': 'Cordoba'
    })

    # Convert date and set as index
    df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y')
    df.set_index('date', inplace=True)

    # Rename columns
    df.rename(columns={
        "median_ticket_(‚ç¨)": "median_ticket",
        "prom_contacts_month": "promotor_visits",
        "tel_contacts_month": "promotor_calls"
    }, inplace=True)

    # Filter out zero volume and income
    filter_condition = (df['volume'] == 0) & (df['income'] == 0)
    df = df[~filter_condition]

    # Create interim directory if it doesn't exist
    interim_dir = os.path.dirname(output_path)
    if not os.path.exists(interim_dir):
        os.makedirs(interim_dir)
        print(f"Created directory: {interim_dir}")

    # Save cleaned data
    df.to_csv(output_path, index=True)
    print(f"Cleaned data saved to {output_path}")