import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier

from input import meeting


def preprocess_labels(df: pd.DataFrame, class_list: list) -> pd.DataFrame:
    """
    Transforms the SA Category column for multi-label classification.
    Handles missing values and converts labeled categories into multi-hot encoded format
    """
    # drop rows with no SA Category label
    df.dropna(subset=['SA Category'], inplace=True)

    #pre-process SA Category column
    processed_labels = df['SA Category'].astype(str).str.split(',').apply(
        lambda codes: [int(float(c.strip())) for c in codes]
    )

    #multilabelBinarizer for multi-hot encoded columns
    mlb = MultiLabelBinarizer(classes=class_list)
    encoded_labels = mlb.fit_transform(processed_labels)

    #new df with encoded_labels
    encoded_df = pd.DataFrame(encoded_labels, columns=mlb.classes_, index=df.index)

    #concatenate utterance with new encoded label columns
    return pd.concat([df['Utterance'], encoded_df], axis=1)

def train_multilabel_sa_model(training_df: pd.DataFrame, label_cols: list) -> Pipeline:

    """
    Trains a multi-label Random Forest model using One-vs-Rest startegy
    """

    #base classifier
    base_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

    #wrap base classifiet in onevsrest
    ovr_classifier = OneVsRestClassifier(base_classifier)

    # Create pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('clf', ovr_classifier)
        ])

    #train model
    pipeline.fit(training_df['Utterance'], training_df[label_cols])

    return pipeline


def process_meeting(meeting_text: str,  model: Pipeline, label_cols: list) -> pd.DataFrame:

    """
    Parses raw meeting transcript and uses trained model for SA categorization
    """

    pattern = re.compile(r"(\d{1,2}:\d{2}\sAM)\s*[-—]\s*(.*?):\s*[\"“”](.*?)[\"“”]", re.DOTALL)
    matches = pattern.findall(meeting_text)

    parsed_data = []
    for timestamp, person, utterance in matches:
        clean_utterance = utterance.replace('\n', ' ').strip()
        predicted_array = model.predict([clean_utterance])[0]
        predicted_labels = [label_cols[i] for i, val in enumerate(predicted_array) if val == 1]
        
        parsed_data.append({
            #'Date': meeting_date,
            'Timestamp': timestamp,
            'Person': person.replace(" (Team Leader)", ""),
            'Utterance': clean_utterance,
            'SA_Code': predicted_labels
        })
    sa_df = pd.DataFrame(parsed_data)
    return sa_df


#define SA categories
SA_CLASSES = [1, 2, 3, 4]
LABEL_COLUMNS = SA_CLASSES

#load training data
try:
    raw_df = pd.read_csv('training_data.csv')
    initial = len(raw_df)
    print(f"Training data loaded successfully from CSV. Found {initial} labeled examples.")

    #preprocess dataset
    train_df = preprocess_labels(raw_df.copy(), SA_CLASSES)
    print(f"Data cleaned and preprocessed. Ready for training with {len(train_df)} labeled examples.")

    #training
    sa_model = train_multilabel_sa_model(train_df, LABEL_COLUMNS)
    print("SA classification model trained successfully")

    # #debugging
    # print(f"Type of sa_model is: {type(sa_model)}")

except FileNotFoundError:
    print("Training data file not found")
    exit()


print("-"*50)


meeting_input = meeting

# #debugging
# print(f"Type of sa_model is: {type(sa_model)}")

#meeting_date = "2024-06-10"
sa_df_predicted = process_meeting(meeting_input, sa_model, LABEL_COLUMNS)
print(f"Processed {len(sa_df_predicted)} SA events from meeting")

print("Meeting Analysis with Random Forest Model:")
print(sa_df_predicted.to_string())

#save to csv
output_path = "predicted_sa_events_3.csv"
sa_df_predicted.to_csv(output_path, index=False)
print(f"Predicted SA events saved to {output_path}")




