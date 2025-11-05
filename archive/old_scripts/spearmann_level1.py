import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def run_level1_correlation_sqdcp():
    """
    loads level 1 hard performance metrics, runs a Spearman correlation,
    and generates a heatmap to visualize the relationships.
    """

    try:
        #load Level 1 Performance Metrics file
        df = pd.read_csv('Performance_Labels.xlsx - Level 1 Performance Metrics.csv')
    except FileNotFoundError:
        print("Error: 'Performance_Labels.xlsx - Level 1 Performance Metrics.csv' not found.")
        return


    # select only the numeric columns for correlation analysis
    numeric_df = df.select_dtypes(include=['number'])

    if numeric_df.empty:
        print("No numeric data found in the file to correlate.")
        return

    # --- Calculate Spearman Correlation ---
    corr_matrix = numeric_df.corr(method='spearman')
    print(corr_matrix)

    # --Visualize as a Heatmap --
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Spearman Correlation of Level 1 Performance Metrics', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    output_filename = 'level1_correlation_heatmap.png'
    plt.savefig(output_filename)
    plt.close()

    print(f"\nCorrelation heatmap saved as '{output_filename}'")


if __name__ == '__main__':
    run_level1_correlation_sqdcp()