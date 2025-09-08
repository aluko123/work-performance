import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from input import meeting

def train_deviation_behavior(df: pd.DataFrame, dimensions: list) -> dict:
    """
    trains a seperate classifier for each of the 17 behavior dimensions
    """

    trained_models = {}

    for dim in dimensions:
        dim_df = df[['Utterance', dim]].dropna()
        if dim_df.empty:
            continue
        model_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english')),
            ('clf', RandomForestClassifier(n_estimators=100, random_state = 42, n_jobs=-1))
        ])
        model_pipeline.fit(dim_df['Utterance'], dim_df[dim])
        trained_models[dim] = model_pipeline
    return trained_models

def process_meeting_deviation_behavior(meeting_text: str, models: dict, dimensions: list) -> pd.DataFrame:
    """
    parse a meeting, predicting scores for all dimensions, and aggregate given scores
    """
    pattern = re.compile(r"(\d{1,2}:\d{2}\sAM)\s*[-—]\s*(.*?):\s*[\"“”](.*?)[\"“”]", re.DOTALL)
    matches = pattern.findall(meeting_text)
    #print(matches[0])
    if not matches:
        return pd.DataFrame()
    
    parsed_data = []
    for timestamp, speaker, utterance, in matches:
        clean_utterance = utterance.replace('\n', ' ').strip() 
        prediction = {
            #"Date": date,
            "Timestamp": timestamp, 
            "Speaker": speaker.replace("(Team Leader)", "").strip(), 
            "Utterance": clean_utterance
        }

        predicted_scores = []
        for dim_name in dimensions:
            if dim_name in models:
                score = models[dim_name].predict([clean_utterance])[0]
                prediction[f"{dim_name}_Pred"] = score
                predicted_scores.append(score)
        if predicted_scores:
            prediction['Agg_Deviation_Score_Pred'] = sum(predicted_scores)
        parsed_data.append(prediction)
    return pd.DataFrame(parsed_data)

# dimensions list
DEV_BEHAVIOR_DIMENSIONS = [
    'Current Target', 'Deviation from Target', 'Process Location & Impact', 'Reason for Deviation (“Why”)', 'Standard Confirmation',
    'Containment / Short-Term Fix', 'Actions Taken to Understand Deviation', 'Trend Recognition', 'Decision Made', 
    'Define Next Actions', 'Report Back Plan', 'Follow-Up on Open Actions', 'Preventive Action / System Fix', 'Spillover / Risk Awareness', 
    'Operator Voice Acknowledged', 'Summary & Prioritization of Issues', 'Go&See (Gemba Walk)'
]


# load training dataset
try:
    train_df = pd.read_csv("deviation_data.csv")
    # col_headers = list(train_df.columns)
    # print("Columns found in file:")
    # print(col_headers)
except FileNotFoundError:
    print("Error: File not found")
    exit()


#train model
dev_behavior_models = train_deviation_behavior(train_df, DEV_BEHAVIOR_DIMENSIONS)


# process a new meeting
meeting_input = meeting

if not dev_behavior_models:
    print("No models were trained, couldn't process meeting")
else:
    dev_behavior_df = process_meeting_deviation_behavior(meeting_input, dev_behavior_models, DEV_BEHAVIOR_DIMENSIONS)
    print("Meeting Analysis: Deviation Behavior")
    if not dev_behavior_df.empty:
        cols = ['Timestamp', 'Speaker']
        if 'Agg_Deviation_Score_Pred' in dev_behavior_df.columns:
            cols.append('Agg_Deviation_Score_Pred')
        cols.extend([f"{dim}_Pred" for dim in DEV_BEHAVIOR_DIMENSIONS if f"{dim}_Pred" in dev_behavior_df.columns])

        #save to csv
        output_path = "predicted_deviation_behavior.csv"
        dev_behavior_df.to_csv(output_path, index=False)
        print(f"Predicted deviation behavior saved to {output_path}")
    else:
        print("No dialogue was processed from the input text")