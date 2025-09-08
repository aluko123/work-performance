import pandas as pd

def deduplicate_master_file(input_path='master_training_data.csv'):
    """
    Loads the master training data, performs aggressive cleaning, removes
    duplicate utterances, and saves a new, cleaned file.
    """
    try:
        df = pd.read_csv(input_path)
        print(f"✅ Successfully loaded '{input_path}'.")
        print(f"Original number of rows: {len(df)}")
    except FileNotFoundError:
        print(f"❌ Error: '{input_path}' not found.")
        return

    # Define the columns that identify a unique utterance
    duplicate_check_subset = ['Date', 'Time Stamp', 'Team Member', 'Utterance']
    
    # Create a copy to avoid modifying the original DataFrame during checks
    df_to_clean = df.copy()

    # --- NEW: Aggressive Cleaning and Normalization ---
    print("Performing aggressive cleaning on key columns...")
    
    # Create a temporary, cleaned column for comparison
    # 1. Convert to lowercase
    # 2. Replace multiple whitespace characters (spaces, tabs, newlines) with a single space
    # 3. Strip leading/trailing whitespace
    df_to_clean['Utterance_normalized'] = df_to_clean['Utterance'].astype(str).str.lower().str.replace(r'\s+', ' ', regex=True).str.strip()
    
    # Also clean the speaker column just in case
    df_to_clean['Speaker_normalized'] = df_to_clean['Team Member'].astype(str).str.lower().str.strip()
    # --------------------------------------------------

    # The new subset for checking duplicates uses the normalized columns
    normalized_subset = ['Date', 'Time Stamp', 'Speaker_normalized', 'Utterance_normalized']

    # Drop duplicates based on the normalized columns, keeping the first occurrence's index
    deduplicated_indices = df_to_clean.drop_duplicates(subset=normalized_subset, keep='first').index
    
    # Use the indices to select the correct rows from the ORIGINAL DataFrame
    deduplicated_df = df.loc[deduplicated_indices]
    
    print(f"Number of rows after removing duplicates: {len(deduplicated_df)}")
    rows_removed = len(df) - len(deduplicated_df)
    print(f"✅ {rows_removed} duplicate rows were removed.")

    # Save the cleaned data to a new file
    output_path = 'master_training_data_deduplicated.csv'
    deduplicated_df.to_csv(output_path, index=False)
    
    print(f"\nClean, deduplicated file saved as '{output_path}'")

# --- Main Execution ---
if __name__ == '__main__':
    deduplicate_master_file()