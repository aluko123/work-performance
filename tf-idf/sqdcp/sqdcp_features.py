import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from input import meeting


SAFETY_FEATURES = {'Hazard Identification', 'Escalation Criteria Stated', 'Escalation Decision Made', 'Proactive Phrasing', 'PPE & Compliance Visibility', 'Safety Coaching Language', 'Emotional Awareness',
    'Silence on Safety'}

QUALITY_FEATURES = {'Clear Defect Description', 'Root Cause Exploration', 'Trend Recognition', 'Voice of Operator Reflected', 'Turn-Taking Balance', 'Containment vs. Root Fix Split', 
    'Quality Tradeoff Framing'}

DELIVERY_FEATURES = {'Clear Deviation Stated', 'Projection Statement', 'Proactive Phrasing', 'Containment Action Identified', 'Accountability Assigned', 'Escalation Decision Made',
    'Trend Recognition', 'Voice of Operator Reflected', 'Voice of Customer Reflected'}

COST_FEATURES = {'Scrap/Waste Acknowledged', 'Rework Time Stated', 'Overtime Justified or Flagged', 'Downtime Cost Noted',
    'Labor Allocation Awareness', 'Material Waste/Inventory Excess', 'Tool/Equipment Cost Impact', 'Cost vs. Risk Tradeoff Framing', 'Prioritization Based on Cost', 'Coaching Language on Efficiency', 
    'Voice of Customer Cost'}

PEOPLE_FEATURES = {'Feedback Quality (Tier 1/2)', 'Participation Inclusivity', 'Emotional Intelligence in Language', 'Dialogue Invitation', 'Recognition of Contributions', 'New Hire Status Shared',
    'Training Progress Shared', 'Training Matrix Awareness', 'Mentoring/Support Mentioned', 'Workload or Morale Reflected'}


SQDCP_FEATURES = [
    'Hazard Identification', 'Escalation Criteria Stated', 'Escalation Decision Made', 'Proactive Phrasing', 'PPE & Compliance Visibility', 'Safety Coaching Language', 'Emotional Awareness',
    'Silence on Safety', 'Clear Defect Description', 'Root Cause Exploration', 'Trend Recognition', 'Voice of Operator Reflected', 'Turn-Taking Balance', 'Containment vs. Root Fix Split', 
    'Quality Tradeoff Framing', 'Clear Deviation Stated', 'Projection Statement', 'Proactive Phrasing', 'Containment Action Identified', 'Accountability Assigned', 'Escalation Decision Made',
    'Trend Recognition', 'Voice of Operator Reflected', 'Voice of Customer Reflected', 'Scrap/Waste Acknowledged', 'Rework Time Stated', 'Overtime Justified or Flagged', 'Downtime Cost Noted',
    'Labor Allocation Awareness', 'Material Waste/Inventory Excess', 'Tool/Equipment Cost Impact', 'Cost vs. Risk Tradeoff Framing', 'Prioritization Based on Cost', 'Coaching Language on Efficiency', 
    'Voice of Customer Cost', 'Feedback Quality (Tier 1/2)', 'Participation Inclusivity', 'Emotional Intelligence in Language', 'Dialogue Invitation', 'Recognition of Contributions', 'New Hire Status Shared',
    'Training Progress Shared', 'Training Matrix Awareness', 'Mentoring/Support Mentioned', 'Workload or Morale Reflected'
]

def train_sqdcp_classification(df: pd.DataFrame, features: list) -> dict:
    """
    trains a seperate classifier for each of the features
    """

    trained_models = {}

    for feature in features:
        feature_df = df[['Utterance', feature]].dropna()
        if feature_df.empty:
            continue
        model_pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english')),
            ('clf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
        ])
        model_pipeline.fit(feature_df['Utterance'], feature_df[feature])
        trained_models[feature] = model_pipeline
    return trained_models 

def process_meeting_feedback_behavior(meeting_text: str, models: dict, features: list) -> pd.DataFrame:
    """
    parse a meeting, predicting scores for all features, and aggregate safety, quality, delivery, cost, people
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

            sqdcp_scores = [0] * 5

            for i, feature_name in enumerate(features):
                if feature_name in models:
                    score = models[feature_name].predict([clean_utterance])[0]
                    prediction[f"{feature_name}_Pred"] = score


                    if feature_name in SAFETY_FEATURES:
                        sqdcp_scores[0] += score
                    elif feature_name in QUALITY_FEATURES:
                        sqdcp_scores[1] += score
                    elif feature_name in DELIVERY_FEATURES:
                        sqdcp_scores[2] += score
                    elif feature_name in COST_FEATURES:
                        sqdcp_scores[3] += score
                    elif feature_name in PEOPLE_FEATURES:
                        sqdcp_scores[4] += score

            prediction['Saftey_Score_Pred'] = sqdcp_scores[0]
            prediction['Quality_Score_Pred'] = sqdcp_scores[1]
            prediction['Delivery_Score_Pred'] = sqdcp_scores[2]
            prediction['Cost_Score_Pred'] = sqdcp_scores[3]
            prediction['People_Score_Pred'] = sqdcp_scores[4]

            parsed_data.append(prediction)
    return pd.DataFrame(parsed_data)

#load training dataset
try:
    train_df = pd.read_csv("sqdcp_data.csv")
except FileNotFoundError:
    print("Error: File not found")
    exit()


#train model
sqdcp_feature_models = train_sqdcp_classification(train_df, SQDCP_FEATURES)

#process meeting
meeting_input = meeting

if not sqdcp_feature_models:
    print("No models were trained, couldn't process meeting")
else:
    sqdcp_df = process_meeting_feedback_behavior(meeting_input, sqdcp_feature_models, SQDCP_FEATURES)
    print("Meeting Analysis: SQDCP")
    if not sqdcp_df.empty:
        cols = ['Date', 'Timestamp', 'Speaker']
        cols.append('Saftey_Score_Pred')
        cols.append('Quality_Score_Pred')
        cols.append('Delivery_Score_Pred')
        cols.append('Cost_Score_Pred')
        cols.append('People_Score_Pred')
        cols.extend([f"{dim}_Pred" for dim in SQDCP_FEATURES if f"{dim}_Pred" in sqdcp_df.columns])

        #save to csv
        output_path = "predicted_sqdcp_v2.csv"
        sqdcp_df.to_csv(output_path, index=False)
        print(f"Predicted SQDCP saved to {output_path}")
    else:
        print("No dialogue was processed from the input text")





