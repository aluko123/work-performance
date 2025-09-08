import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates

def generate_daily_trend_chart():
    """
    loads predicted communication data and creates a line chart of average
    Tier 1 and Tier 2 score for each individual speaker
    """

    try:
        df = pd.read_csv('predicted_feedback_behavior.csv')
    except FileNotFoundError:
        print("Error: 'predicted_feedback_behavior.csv' not found")
        return
    
    #verify column name
    score_cols = ['Tier1_Aggregate_Pred', 'Tier2_Aggregate_Pred']
    df['date_obj'] = pd.to_datetime(df['Date'], errors='coerce')
    df.dropna(subset=['date_obj'] + score_cols, inplace=True)

    if df.empty:
        print("No valid data found to generate chart")
        return

    # group  by data and speaker and find mean for each day for each tier
    daily_speaker_scores = df.groupby(['date_obj', 'Speaker'])[score_cols].mean()

    # make speakers become columns
    pivot_df = daily_speaker_scores.unstack(level='Speaker')


    # we'll make a two-panel plot for Tier 1 and Tier 2 aggregates alike
    fig, axes = plt.subplots(2, 1, figsize=(15, 12), sharex=True)

    # Plot 1: Tier 1 Scores
    tier1_df = pivot_df['Tier1_Aggregate_Pred']
    for speaker in tier1_df.columns:
        axes[0].plot(tier1_df.index, tier1_df[speaker], marker='o', linestyle='-', label=speaker)
    
    axes[0].set_title('Daily Average Tier 1 Feedback Scores by Speaker', fontsize=16)
    axes[0].set_ylabel('Daily Average Score', fontsize=12)
    axes[0].legend(title='Speaker')
    axes[0].grid(True, which='both', linestyle='--', linewidth=0.5)


    # Plot 2: Tier 2 Scores
    tier2_df = pivot_df['Tier2_Aggregate_Pred']
    for speaker in tier2_df.columns:
        axes[1].plot(tier2_df.index, tier2_df[speaker], marker='o', linestyle='-', label=speaker)
    
    axes[1].set_title('Daily Average Tier 2 Feedback Scores by Speaker', fontsize=16)
    axes[1].set_ylabel('Daily Average Score', fontsize=12)
    axes[1].set_xlabel('Date', fontsize=12)
    axes[1].legend(title='Speaker')
    axes[1].grid(True, which='both', linestyle='--', linewidth=0.5)

    #format x-axis
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()

    plt.tight_layout()

    output_filename = 'feeback_trends.png'
    plt.savefig(output_filename)
    plt.close()
    print(f"Feedback trends chart saved as '{output_filename}'")

#run the function
if __name__ == "__main__":
    generate_daily_trend_chart()