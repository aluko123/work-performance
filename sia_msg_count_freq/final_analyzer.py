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

def parse_correctly(text):
    """
    Parses multi-line messages from the meeting transcript.
    A message starts after a speaker line and ends before the next speaker line or date.
    """
    messages = []
    current_date = None
    current_message_lines = []
    current_speaker_info = None

    # Regex to identify a line with a speaker and timestamp
    speaker_pattern = re.compile(r"^(\d{1,2}:\d{2}\s*(?:AM|PM))\s*—\s*([^:]+):")
    date_pattern = re.compile(r"^(Monday|Tuesday|Wednesday|Thursday|Friday), (\w+ \d{1,2}, \d{4})")

    for line in text.split('\n'):
        date_match = date_pattern.match(line)
        speaker_match = speaker_pattern.match(line)

        # If we find a new date or a new speaker, the previous message has ended.
        if (date_match or speaker_match) and current_speaker_info and current_message_lines:
            # Join the collected lines to form the full message
            full_message = ' '.join(current_message_lines).strip()
            
            # Clean up the message from quotes and extra whitespace
            full_message = re.sub(r'^"|"$', '', full_message)
            full_message = re.sub(r'\s+', ' ', full_message)

            if full_message: # Only save if there is content
                messages.append({
                    "date": current_speaker_info['date'],
                    "speaker": current_speaker_info['speaker'],
                    "message": full_message
                })
            
            # Reset for the next message
            current_message_lines = []

        # --- Start of a new entry ---
        if date_match:
            current_date = date_match.group(2)
            current_speaker_info = None # A new day starts
        
        if speaker_match and current_date:
            time, speaker_name = speaker_match.groups()
            
            # Clean up speaker name
            if '(' in speaker_name:
                speaker_name = speaker_name.split('(')[0].strip()

            current_speaker_info = {'date': current_date, 'speaker': speaker_name}
        
        # If we are after a speaker but before the next one, it's part of the message
        elif current_speaker_info and not speaker_match and not date_match:
            current_message_lines.append(line.strip())

    # Add the very last message in the file
    if current_speaker_info and current_message_lines:
        full_message = ' '.join(current_message_lines).strip()
        full_message = re.sub(r'^"|"$', '', full_message)
        full_message = re.sub(r'\s+', ' ', full_message)
        if full_message:
            messages.append({
                "date": current_speaker_info['date'],
                "speaker": current_speaker_info['speaker'],
                "message": full_message
            })

    return messages

def analyze_communication(messages):
    download_nltk_data()
    sid = SentimentIntensityAnalyzer()
    analysis_results = []

    for msg in messages:
        message_text = msg['message']
        sentiment_scores = sid.polarity_scores(message_text)
        tokens = word_tokenize(message_text)
        pos_tags = pos_tag(tokens)
        
        analysis_results.append({
            "date": msg['date'],
            "speaker": msg['speaker'],
            "message": message_text,
            "sentiment": sentiment_scores,
            "pos_tags": pos_tags
        })
    
    return analysis_results

if __name__ == "__main__":
    import io
    import sys

    parsed_messages = parse_correctly(meeting)
    if parsed_messages:
        analysis_results = analyze_communication(parsed_messages)
        
        # Create a DataFrame for detailed results
        detailed_df = pd.DataFrame(analysis_results)
        
        # Save full results to CSV for easier review
        detailed_df.to_csv("final_communication_analysis.csv", index=False)
        print("Full analysis including POS tags saved to 'final_communication_analysis.csv'")

        # Capture summary output to a string buffer
        summary_buffer = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = summary_buffer

        print("--- Communication Analysis Summary ---")
        print(f"Total Messages: {len(detailed_df)}")
        print("\nMessages per Speaker:")
        print(detailed_df['speaker'].value_counts().to_string())
        print("\n\nMessages per Day:")
        print(detailed_df['date'].value_counts().sort_index().to_string())
        print("\n\nSentiment Analysis (Compound Score Average):")
        print(detailed_df.groupby('speaker')['sentiment'].apply(lambda s: s.apply(lambda x: x['compound']).mean()))
        
        # Restore stdout
        sys.stdout = old_stdout
        
        # Get summary string and write to file
        summary_output = summary_buffer.getvalue()
        with open("analysis_summary.txt", "w") as f:
            f.write(summary_output)
        
        # Print summary to console as well
        print(summary_output)
        print("\nSummary also saved to 'analysis_summary.txt'")

    else:
        print("Failed to parse any messages.")
