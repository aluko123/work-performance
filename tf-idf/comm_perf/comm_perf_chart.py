import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates

def generate_speaker_trend_chart():
    """
    loads predicted communication data and creates a line chart of average
    score for each individual speaker
    """
    try:
        df = pd.read_csv('predicted_comm_perf.csv')
    except FileNotFoundError:
        print("Error: 'predicted_comm_perf.csv' not found")
        return
    
    #verify column name
    aggregate_score_column = 'Agg_Comm_Score_Pred'
    if aggregate_score_column not in df.columns:
        print(f"Error: The score column '{aggregate_score_column}' was not found in the file.")
        print(f"Available columns are: {list(df.columns)}")
        return
    
    #use 'date' column to convert to datetime objects
    df['date_obj'] = pd.to_datetime(df['Date'], errors='coerce')
    df.dropna(subset=['date_obj', aggregate_score_column], inplace=True)
    

    if df.empty:
        print("No valid dates found in the data. Cannot generate trend chart.")
        return
    
    daily_speaker_scores = df.groupby(['date_obj', 'Speaker'])[aggregate_score_column].mean()

    #make speakers become columns
    pivot_df = daily_speaker_scores.unstack(level='Speaker')


    #create plot
    plt.figure(figsize=(15, 8))

    for speaker in pivot_df.columns:
        plt.plot(pivot_df.index, pivot_df[speaker], marker='o', linestyle='-', label=speaker)

    plt.title('Daily Average Communication Score by Speaker', fontsize=16)
    plt.ylabel('Daily Average Communication Score (1-5)', fontsize=12)
    plt.xlabel('Date', fontsize=12)
    plt.legend(title='Speaker')
    plt.grid(True, which='both',linestyle='--', linewidth=0.5)

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()

    output_filename = 'comm_speaker_trends.png'
    plt.savefig(output_filename)
    plt.close()
    print(f"Communication speaker trends chart saved as '{output_filename}'")

#run the function
if __name__ == "__main__":
    generate_speaker_trend_chart()
    
