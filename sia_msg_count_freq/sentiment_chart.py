import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import ast


# parse sentiment column to extract compound score
def extract_compound_score(sentiment_string):
    try:
        sentiment_dict = ast.literal_eval(sentiment_string)
        return sentiment_dict.get('compound', None)
    except (ValueError, SyntaxError, TypeError):
        return None

def generate_speaker_sentiment_trend_chart():
    """
    loads communication data, parsing the sentiment column to extract the compound score
    and create a line chart of daily average sentiment for each speaker    
    """

    try:
        df = pd.read_csv('final_communication_analysis.csv')
    except FileNotFoundError:
        print("Error: 'final_communication_analysis.csv' not found.")
        return
    
    #verify column names
    date_col = 'date'
    speaker_col = 'speaker'
    sentiment_col = 'sentiment'

    required_cols = [date_col, speaker_col, sentiment_col]
    if not all(col in df.columns for col in required_cols):
        print(f"Error: The CSV must contain the following columns: {required_cols}")
        return
    
    df['compound_score'] = df[sentiment_col].apply(extract_compound_score)

    # Aggregate by Day and Speaker
    df['date_obj'] = pd.to_datetime(df[date_col], errors='coerce')
    df.dropna(subset=['date_obj', 'compound_score'], inplace=True)

    if df.empty:
        print("No valid data found to generate trend chart.")
        return

    daily_speaker_sentiment = df.groupby(['date_obj', speaker_col])['compound_score'].mean() 

    #pivot data
    pivot_df = daily_speaker_sentiment.unstack(level=speaker_col)


    #Create line chart
    plt.figure(figsize=(15, 8))

    for speaker in pivot_df.columns:
        plt.plot(pivot_df.index, pivot_df[speaker], marker='o', linestyle='-', label=speaker)

    daily_average = df.groupby('date_obj')['compound_score'].mean()
    plt.plot(daily_average.index, daily_average, linestyle='--', color='gray', label='Team Average')

    #format
    plt.title('Daily Average Sentiment Trend by Speaker', fontsize=16)
    plt.ylabel('Average Compound Sentiment Score (-1 to 1)', fontsize=12)
    plt.xlabel('Date', fontsize=12)
    plt.ylim(-1, 1)
    plt.legend(title='Speaker')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()

    output_filename = 'speaker_sentiment_trends.png'
    plt.savefig(output_filename)
    plt.close()
    print(f"Speaker sentiment trends chart saved as '{output_filename}'")   


if __name__ == '__main__':
    generate_speaker_sentiment_trend_chart()