import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from input import meeting

def train_comm_perf_models(df: pd.DataFrame, dimensions: list) -> dict:
    """
    trains a seperate classifier for each of the 10 communication dimensions
    """

    trained_models = {}

    for dim in dimensions:
        dim_df = df[['Utterance', dim]].dropna()
        if dim_df.empty:
            continue
        model_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english')),
            ('clf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
        ])
        model_pipeline.fit(dim_df['Utterance'], dim_df[dim])
        trained_models[dim] = model_pipeline
    return trained_models


def process_meeting_comm_perf(meeting_text: str, models: dict, dimensions: list) -> pd.DataFrame:
    """
    parse a meeting, predicting scores for all dimensions, and aggregate given scores
    """
    pattern = re.compile(r'((?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday), (?:January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2}, \d{4})|(\d{1,2}:\d{2}\sAM)\s*[-—]\s*(.*?):\s*[""](.*?)[""]', re.DOTALL)
    matches = pattern.findall(meeting_text)

    if not matches:
        return pd.DataFrame()

    parsed_data = []
    current_date = None
    for match in matches:
        if match[0]:
            current_date = match[0]
        else:
            timestamp, speaker, utterance = match[1], match[2], match[3]
            clean_utterance = utterance.replace('\n', ' ').strip()

            prediction = {
                "Date": current_date,
                "Timestamp": timestamp,
                "Speaker": speaker.replace("(Team Leader)", "").strip(),
                "Utterance": clean_utterance,
            } 

            #Get a prediction from each model
            predicted_scores = []
            for dim_name in dimensions:
                if dim_name in models:
                    score = models[dim_name].predict([clean_utterance])[0]
                    prediction[f"{dim_name}_Pred"] = score
                    predicted_scores.append(score)
        
            if predicted_scores:
                #print("Predicted scores: ", predicted_scores)
                prediction['Agg_Comm_Score_Pred'] = sum(predicted_scores)

            parsed_data.append(prediction)

    return pd.DataFrame(parsed_data)

# -- clean score columns --
def clean_score_columns(df: pd.DataFrame, dimensions: list) -> pd.DataFrame:
    """
    Cleans the score columns by extracting only the leading number
    """
    print("Cleaning score columns")
    for dim in dimensions:
        if dim in df.columns:
            # convert column to string to handle different data types
            # split by '-' and take the first part
            # convert result to numeric type, and make errors NaN
            df[dim] = pd.to_numeric(df[dim].astype(str).str.split(r'\s*[-–]\s*').str[0], errors='coerce')
    return df


# dimensions list
COMM_PERF_DIMENSIONS = [
    'Pausing', 'Verbal Affirmation', 'Continuation Prompts', 'Paraphrasing', 'Clarifying Questions',
     'Probing Questions', 'Open-Ended Questions', 'Coaching Questions', 'Acknowledgment of Emotions', 'Summary Statements'
]

#load training dataset
try:
    train_df = pd.read_csv("comm_data.csv")
except FileNotFoundError:
    print("Error: File not found")
    exit()


# if 'Pausing' in train_df.columns:
#     print("\n--- Before Cleaning ---")
#     print("Unique values in 'Pausing' column:")
#     print(train_df['Pausing'].unique())

# data cleaning on score columns
train_df = clean_score_columns(train_df, COMM_PERF_DIMENSIONS)


# if 'Pausing' in train_df.columns:
#     print("\n--- Before Cleaning ---")
#     print("Unique values in 'Pausing' column:")
#     print(train_df['Pausing'].unique())

# train the models
comm_perf_models = train_comm_perf_models(train_df, COMM_PERF_DIMENSIONS)


# process a new meeting
meeting_input = meeting


if not comm_perf_models:
    print("No models were trained, couldn't process meeting")
else:
    #comm_perf_df = pd.DataFrame() 
    comm_perf_df = process_meeting_comm_perf(meeting_input, comm_perf_models, COMM_PERF_DIMENSIONS)
    print("Meeting Analysis: Communication Performance")
    if not comm_perf_df.empty:
        #reorder columns to show aggregated score first
        cols = ['Timestamp', 'Speaker']
        if 'Agg_Comm_Score_Pred' in comm_perf_df.columns:
            cols.append('Agg_Comm_Score_Pred')
        cols.extend([f"{dim}_Pred" for dim in COMM_PERF_DIMENSIONS if f"{dim}_Pred" in comm_perf_df.columns])

        #print(comm_perf_df[cols].to_string())

        #save to csv
        output_path = "predicted_comm_perf.csv"
        comm_perf_df.to_csv(output_path, index=False)
        print(f"Predicted communication performance saved to {output_path}")
    else:
        print("No dialogue was processed from the input text")


