import pandas as pd
from functools import reduce
import os
import re
import json



def clean_and_prefix_columns(df: pd.DataFrame, prefix) -> pd.DataFrame:
    """
    cleans all potential score columns by extracting only the leading number
    """
    new_columns = {}
    merge_keys = ['Date', 'Time Stamp', 'Team Member', 'Utterance', 'Timestamp', 'Speaker']

    for col in df.columns:
        if col in merge_keys:
            #standardize
            if col == 'Time Stamp': new_columns[col] = 'Timestamp'
            elif col == 'Team Member': new_columns[col] = 'Speaker'
            else: new_columns[col] = col
        else:
            sanitized_col = re.sub(r'[^a-zA-Z0-9]', '_', col)
            new_columns[col] = f"{prefix}_{sanitized_col}"

    df.rename(columns=new_columns, inplace=True)
    return df


def clean_score_columns(df: pd.DataFrame) -> pd.DataFrame:

    print("Cleaning score columns...")
    for col in df.columns:
        #clean score columns
        if df[col].dtype == 'object' and col not in ['Utterance', 'Speaker', 'Date', 'Timestamp']:
            #convert to string
            #df[col] = df[col].astype(str)
            #extract numeric lead and convert to number
            df[col] = pd.to_numeric(df[col].astype(str).str.split(r'\s*[-–]\s*').str[0], errors='coerce')

    return df


def consolidate_training_data():
    """
    load all seperate labeled data files, merge into a single master df, clean and save
    """
    data_dir = '../tf-idf/'
    files_to_load = {
        'comm':'comm_perf/comm_data.csv',
        'feedback': 'feedback/feedback_data.csv',
        'deviation': 'deviation_data.csv',
        'sqdcp': 'sqdcp/SQDCP_Data.csv'
    }

    dataframes = []
    column_name_map = {}

    for prefix, file_path in files_to_load.items():
        try:
            df = pd.read_csv(data_dir + file_path)
            original_columns = df.columns.tolist()
            
            #clean and prefix
            df = clean_and_prefix_columns(df, prefix)
            dataframes.append(df)

            #create mapping to help header names
            for orig_col, new_col in zip(original_columns, df.columns):
                if orig_col not in ['Date', 'Time Stamp', 'Team Member', 'Utterance']:
                    column_name_map[new_col] = {
                        "original_name": orig_col,
                        "source_file": file_path
                    }
            print(f"Successfully loaded '{file_path}' with {len(df)} rows.")
        except FileNotFoundError:
            print(f"Warning: File not found: '{file_path}'. Skipping.")

    if len(dataframes) < 2:
        print("Error: Need at least two data files to merge. Stopping.")
        return


    merge_keys = ['Date', 'Timestamp', 'Speaker', 'Utterance']
    print(f"\nMerging dataframes on keys: {merge_keys}...")

    # ensures no rows are lost if utterance is in one file and not the other
    merged_df = reduce(lambda left, right: pd.merge(left, right, on=merge_keys, how='outer'), dataframes)

    print(f"Merging complete. The consolidated table has {len(merged_df)} rows.")


    #clean score columns
    merged_df = clean_score_columns(merged_df)

    #deduplicate final dataframe
    df_to_clean = merged_df.copy()
    df_to_clean['Utterance_normalized'] = df_to_clean['Utterance'].astype(str).str.lower().str.replace(r'\s+', ' ', regex=True).str.strip()
    df_to_clean['Speaker_normalized'] = df_to_clean['Speaker'].astype(str).str.lower().str.strip()
    normalized_subset = ['Date', 'Timestamp', 'Speaker_normalized', 'Utterance_normalized']
    deduplicated_indices = df_to_clean.drop_duplicates(subset=normalized_subset, keep='first').index
    final_df = merged_df.loc[deduplicated_indices]
    print(f"\n\nDeduplication complete. Data has {len(final_df)} rows.")

    #master training file
    output_filename = 'master_training_data.csv'
    final_df.to_csv(output_filename, index=False)

    print(f"\nMaster training file saved to '{output_filename}'")
    print("Please review this file to ensure the data has been merged correctly.")

    output_map_path = 'column_name_mapping.json'
    with open(output_map_path, 'w') as f:
        json.dump(column_name_map, f, indent=4)
    print(f"Column name mapping saved as '{output_map_path}'")

# main execution
if __name__ == '__main__':
    consolidate_training_data()

