import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from input import meeting


def train_feedback_behavior(df: pd.DataFrame, dimensions: list) -> dict:
    """
    trains a seperate classifier for each of the 10 dimensions
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


def process_meeting_feedback_behavior(meeting_text: str, models: dict, dimensions: list) -> pd.DataFrame:
    """
    parse a meeting, predicting scores for all dimensions, and aggregate tier 1 and tier sums
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
                "Utterance": clean_utterance
            }

            #aggregation
            tier1_score = 0
            tier2_score = 0

            #Use enumerate to get both index(i) and dimension name
            for i, dim_name in enumerate(dimensions):
                if dim_name in models:
                    score = models[dim_name].predict([clean_utterance])[0]
                    prediction[f"{dim_name}_Pred"] = score
                    
                    if i < 5:
                        tier1_score += score
                    else:
                        tier2_score += score

            prediction['Tier1_Aggregate_Pred'] = tier1_score
            prediction['Tier2_Aggregate_Pred'] = tier2_score


            parsed_data.append(prediction)
    return pd.DataFrame(parsed_data)

#dimensions list
FEEDBACK_BEHAVIOR_DIMENSIONS = [
    'Timely', 'Neutral & Specific', 'Impact + Emotion', 'Action-Oriented', 'Clarity of Situation',
    'Feedback for Reinforcement', 'Feedback for Improvement', '"I" + Sensory Statements', 
    'Objective Framing', 'Avoids Mind Reading', 'Behavior + Pattern Awareness', 'Invites Dialogue'
]


#load training dataset
try:
    train_df = pd.read_csv("feedback_data.csv")
except FileNotFoundError:
    print("Error: File not found")
    exit()

#train model
feedback_behavior_models = train_feedback_behavior(train_df, FEEDBACK_BEHAVIOR_DIMENSIONS)

#process a new meeting
meeting_input = meeting

if not feedback_behavior_models:
    print("No models were trained, couldn't process meeting")
else:
    feedback_behavior_df = process_meeting_feedback_behavior(meeting_input, feedback_behavior_models, FEEDBACK_BEHAVIOR_DIMENSIONS)
    print("Meeting Analysis: Feedback Behavior")
    if not feedback_behavior_df.empty:
        cols = ['Date', 'Timestamp', 'Speaker']
        cols.append('Tier1_Aggregate_Pred')
        cols.append('Tier2_Aggregate_Pred')
        cols.extend([f"{dim}_Pred" for dim in FEEDBACK_BEHAVIOR_DIMENSIONS if f"{dim}_Pred" in feedback_behavior_df.columns])


        #save to csv
        output_path = "predicted_feedback_behavior.csv"
        feedback_behavior_df.to_csv(output_path, index=False)
        print(f"Predicted feedback behavior saved to {output_path}")
    else:
        print("No dialogue was processed from the input text")