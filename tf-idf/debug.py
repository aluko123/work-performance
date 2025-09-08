import pandas as pd

def debug_week_calculation():
    """
    Loads the data and prints each step of the week calculation
    to diagnose the issue.
    """
    try:
        df = pd.read_csv('predicted_sqdcp.csv')
        print("✅ Successfully loaded 'predicted_sqdcp.csv'.")
    except FileNotFoundError:
        print("❌ Error: 'predicted_sqdcp.csv' not found.")
        return

    print("\n--- Step 1: Parsing 'Timestamp' column ---")
    # We will try to convert the Timestamp to a datetime object
    # The 'errors=coerce' will turn any timestamp that can't be read into 'NaT' (Not a Time)
    df['datetime'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    print("First 5 entries of the new 'datetime' column:")
    print(df['datetime'].head())
    print(f"\nNumber of rows where timestamp could not be parsed (NaT): {df['datetime'].isna().sum()}")

    # Stop if all timestamps failed to parse
    if df['datetime'].isna().all():
        print("\n❌ CRITICAL ERROR: Could not parse any timestamps. Please check the format in your CSV.")
        return

    print("\n--- Step 2: Creating 'adjusted_date' (shifting for 6:30 AM) ---")
    df['adjusted_date'] = (df['datetime'] - pd.to_timedelta('6 hours 30 minutes')).dt.date
    print("First 5 entries of the new 'adjusted_date' column:")
    print(df['adjusted_date'].head())


    print("\n--- Step 3: Assigning 'Work_Week' number ---")
    unique_workdays = sorted(df['adjusted_date'].dropna().unique())
    workday_map = {day: i for i, day in enumerate(unique_workdays)}
    df['day_num'] = df['adjusted_date'].map(workday_map)
    df['Work_Week'] = (df['day_num'] // 5) + 1
    print("First 5 entries of the final 'Work_Week' column:")
    print(df['Work_Week'].head())
    print(f"\nUnique Work_Week values calculated: {df['Work_Week'].unique()}")


# Run the debug function
debug_week_calculation()