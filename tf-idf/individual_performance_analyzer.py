import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def normalize_score(score, min_val, max_val, new_min=1, new_max=5):
    """
    normalizes a score from its original scale to a new scale (default 1-5).
    """
    if max_val == min_val:
        return new_min
    return new_min + (score - min_val) * (new_max - new_min) / (max_val - min_val)

def generate_normalized_spider_plot(speaker_name: str):
    """
    loads all prediction data, normalizes scores for a specific speaker,
    and generates a spider plot of performance profile
    """

    #load data sources
    try:
        feedback_df = pd.read_csv('feedback/predicted_feedback_behavior.csv')
        comm_df = pd.read_csv('comm_perf/predicted_comm_perf.csv')
        sqdcp_df = pd.read_csv('sqdcp/predicted_sqdcp.csv')
    except FileNotFoundError as e:
        print(f"Error: Could not find required data file. {e}")
        return
    

    # define key metrics to plot
    metrics_to_plot = {
        # Label: (DataFrame, ColumnName, MinScore, MaxScore)
        'Timely': (feedback_df, 'Timely_Pred', 1, 5),
        'Action-Oriented': (feedback_df, 'Action-Oriented_Pred', 1, 5),
        'Communication': (comm_df, 'Agg_Comm_Score_Pred', 10, 50),
        'Tier1_Communication': (feedback_df, 'Tier1_Aggregate_Pred', 5, 25),
        'Tier2_Communication': (feedback_df, 'Tier2_Aggregate_Pred', 5, 25),
        'Safety Score': (sqdcp_df, 'Saftey_Score_Pred', 8, 40),
        'Quality Score': (sqdcp_df, 'Quality_Score_Pred', 7, 35),
        'People Score': (sqdcp_df, 'People_Score_Pred', 10, 50),
        'Delivery Score': (sqdcp_df, 'Delivery_Score_Pred', 9, 45),
        'Cost Score': (sqdcp_df, 'Cost_Score_Pred', 11, 55)
    }

    #calculate average scores
    normalized_scores = []
    labels = []

    for label, (df, col_name, min_val, max_val) in metrics_to_plot.items():
        if speaker_name not in df['Speaker'].unique():
            print(f"Warning: Speaker '{speaker_name} not found in data for metric '{label}'. Skipping.")
            continue

        if col_name not in df.columns:
            print(f"Warning: Column '{col_name}' not found  for metric '{label}'. Skipping.")
            continue

        avg_raw_score = df[df['Speaker'] == speaker_name][col_name].mean()
        
        #Normalize score
        normalized = normalize_score(avg_raw_score, min_val, max_val)
        
        normalized_scores.append(normalized)
        labels.append(label)

    if not normalized_scores:
        print(f"No data found for speaker '{speaker_name}'. Cannont generate plot.")
        return
    

    #generate plot
    num_vars = len(labels)

    #angles for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    normalized_scores += normalized_scores[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    
    ax.plot(angles, normalized_scores, color='blue', linewidth=2)
    ax.fill(angles, normalized_scores, color='blue', alpha=0.25)

    #draw the plot and fill the area
    ax.set_rlim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["1", "2", "3", "4", "5"], color="grey", size=10)

    #set the labels for each axis
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=12)

    #set radial grid and limits
    #ax.set_rlim(0, 5)
    #plt.yticks([1, 2, 3, 4, 5], ["1", "2", "3", "4", "5"], color="grey", size=10)

    plt.title(f"Normalized Performance Profile for {speaker_name}", size=20, color='blue', y=1.1)

    output_filename = f'normalized_profile_{speaker_name.lower()}.png'
    plt.savefig(output_filename)
    plt.close()

    print(f"Normalized individual performance spider plot saved as '{output_filename}'")


SPEAKER = "Carlos"

if __name__ == '__main__':
    generate_normalized_spider_plot(SPEAKER)