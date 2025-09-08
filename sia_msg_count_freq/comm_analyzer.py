import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from input import meeting
import pandas as pd

def download_nltk_data():
    # Download necessary NLTK data with simpler, more robust error handling
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger')
    try:
        nltk.data.find('sentiment/vader_lexicon')
    except LookupError:
        nltk.download('vader_lexicon')

def analyze_communication(text):
    download_nltk_data()

    # Corrected Regex to find messages
    message_pattern = re.compile(r"^(\d{1,2}:\d{2}\s*(?:AM|PM))\s*—\s*([^:]+):\s*(.*)", re.MULTILINE)
    date_pattern = re.compile(r"^(Monday|Tuesday|Wednesday|Thursday|Friday), (\w+ \d{1,2}, \d{4})")

    messages = []
    current_date = None

    for line in text.split('\n'):
        date_match = date_pattern.match(line)
        if date_match:
            current_date = date_match.group(2)
            continue

        message_match = message_pattern.match(line)
        if message_match and current_date:
            time, speaker, content = message_match.groups()
            speaker = speaker.strip()
            # Clean up speaker name if it includes role
            if '(' in speaker:
                speaker = speaker.split('(')[0].strip()
            
            messages.append({
                "date": current_date,
                "time": time,
                "speaker": speaker,
                "message": content.strip()
            })

    if not messages:
        print("No messages found. Please check the input format.")
        return

    # Initialize sentiment analyzer
    sid = SentimentIntensityAnalyzer()

    # --- Analysis ---
    
    # 1. Message Count & Frequency
    df = pd.DataFrame(messages)
    total_messages = len(df)
    messages_per_speaker = df['speaker'].value_counts()
    messages_per_day = df['date'].value_counts().sort_index()

    # 2. POS Tagging and Sentiment Analysis
    analysis_results = []
    for index, row in df.iterrows():
        message = row['message']
        
        # Sentiment
        sentiment_scores = sid.polarity_scores(message)
        
        # POS Tagging
        tokens = word_tokenize(message)
        pos_tags = pos_tag(tokens)
        
        analysis_results.append({
            "date": row['date'],
            "speaker": row['speaker'],
            "message": message,
            "sentiment": sentiment_scores,
            "pos_tags": pos_tags
        })

    # --- Output ---
    
    print("--- Communication Analysis ---")
    print("\n1. Message Count & Frequency")
    print("-----------------------------")
    print(f"Total Messages: {total_messages}")
    print("\nMessages per Speaker:")
    print(messages_per_speaker.to_string())
    print("\n\nMessages per Day:")
    print(messages_per_day.to_string())
    
    print("\n\n2. Detailed Message Analysis (Sentiment & POS)")
    print("-------------------------------------------------")
    
    # Create a DataFrame for detailed results
    detailed_df = pd.DataFrame(analysis_results)
    
    # Format for better readability
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_colwidth', 80)
    
    # Print selected columns for brevity in the console
    display_df = detailed_df[['date', 'speaker', 'message', 'sentiment']].copy()
    display_df['sentiment_compound'] = display_df['sentiment'].apply(lambda x: x['compound'])
    
    print(display_df[['date', 'speaker', 'message', 'sentiment_compound']].to_string())
    
    # Save full results to CSV for easier review
    detailed_df.to_csv("communication_analysis_full.csv", index=False)
    print("\n\nFull analysis including POS tags saved to 'communication_analysis_full.csv'")


if __name__ == "__main__":
    analyze_communication(meeting)