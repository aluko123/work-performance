import pandas as pd
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# --- Configuration ---
MODEL_PATH = '/mnt/c/Users/adeda/Dropbox/PC/Downloads/Work/bert_classification/multi_task_bert_model.pth'
DATA_PATH = '/mnt/c/Users/adeda/Dropbox/PC/Downloads/Work/bert_classification/master_training_data.csv'
TEXT_COLUMN = 'Utterance'
TEST_SET_SIZE = 0.2
RANDOM_STATE = 42

# --- Model Definition (copied from your script for self-containment) ---
class MultiTaskBertModel(nn.Module):
    def __init__(self, n_classes_dict):
        super(MultiTaskBertModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifiers = nn.ModuleDict({
            task_name: nn.Linear(self.bert.config.hidden_size, num_labels)
            for task_name, num_labels in n_classes_dict.items()
        })

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        bert_output = self.bert(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled_output = bert_output.pooler_output
        outputs = {
            task_name: classifier(pooled_output)
            for task_name, classifier in self.classifiers.items()
        }
        return outputs

# --- Main Functions ---
def load_data(path):
    """Loads and prepares the dataset."""
    print(f"Loading data from {path}...")
    df = pd.read_csv(path)
    # Identify all metric columns to be used as labels
    metric_cols = [col for col in df.columns if col.startswith(('comm_', 'feedback_', 'deviation_', 'sqdcp_'))]
    return df, metric_cols

def predict_scores(text, model, tokenizer, metric_cols):
    """Generates predictions for a single text utterance."""
    model.eval()
    inputs = tokenizer(
        text,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=128
    )
    with torch.no_grad():
        outputs = model(**inputs)

    predictions = {}
    for col_name in metric_cols:
        # Model predicts logits for 5 classes (0-4). Argmax finds the most likely class.
        pred_index = torch.argmax(outputs[col_name], dim=1).item()
        predictions[col_name] = pred_index
    return predictions

def main():
    """Main function to run the evaluation."""
    df, metric_cols = load_data(DATA_PATH)

    # Split data into training and a held-out test set
    _, test_df = train_test_split(
        df,
        test_size=TEST_SET_SIZE,
        random_state=RANDOM_STATE
    )
    print(f"Data split complete. Test set size: {len(test_df)} rows.")

    # Load the trained model
    n_classes = {col: 5 for col in metric_cols}
    model = MultiTaskBertModel(n_classes)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    print(f"Model loaded successfully from {MODEL_PATH}.")

    # Store true labels and predictions
    all_true_labels = {col: [] for col in metric_cols}
    all_predictions = {col: [] for col in metric_cols}

    print("\nGenerating predictions on the test set...")
    for _, row in test_df.iterrows():
        text = row[TEXT_COLUMN]
        if not isinstance(text, str):
            continue # Skip rows where utterance is not a string

        # Get model predictions (as 0-4)
        predicted_labels = predict_scores(text, model, tokenizer, metric_cols)

        for col in metric_cols:
            true_label = row[col]
            # The original labels are 1-5. We subtract 1 to match the model's 0-4 output.
            # We only evaluate on labels that are not NaN.
            if pd.notna(true_label):
                all_true_labels[col].append(int(true_label) - 1)
                all_predictions[col].append(predicted_labels[col])

    print("Prediction generation complete.")
    print("\n--- Model Performance Report ---")

    # Generate and print a classification report for each metric
    for col in metric_cols:
        true_labels_for_col = all_true_labels[col]
        predictions_for_col = all_predictions[col]

        if len(true_labels_for_col) > 0:
            print(f"\n--- METRIC: {col} ---")
            # Set zero_division=0 to handle cases where a class has no predictions
            report = classification_report(
                true_labels_for_col,
                predictions_for_col,
                zero_division=0
            )
            print(report)
        else:
            print(f"\n--- METRIC: {col} ---")
            print("No labeled data available in the test set for this metric.")

if __name__ == '__main__':
    main()
