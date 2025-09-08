import pandas as pd
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split

# --- Configuration ---
MODEL_PATH = '/mnt/c/Users/adeda/Dropbox/PC/Downloads/Work/bert_classification/multi_task_bert_model.pth'
DATA_PATH = '/mnt/c/Users/adeda/Dropbox/PC/Downloads/Work/bert_classification/master_training_data.csv'
TEXT_COLUMN = 'Utterance'
TEST_SET_SIZE = 0.2
RANDOM_STATE = 42
N_SAMPLES_TO_REVIEW = 5 # Number of random samples to review

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
        pred_index = torch.argmax(outputs[col_name], dim=1).item()
        # Convert from 0-4 index to 1-5 score
        predictions[col_name] = pred_index + 1
    return predictions

def main():
    """Main function to run the qualitative review."""
    df, metric_cols = load_data(DATA_PATH)

    # Split data to get the same test set as the evaluation script
    _, test_df = train_test_split(
        df,
        test_size=TEST_SET_SIZE,
        random_state=RANDOM_STATE
    )

    # Take a random sample from the test set
    if len(test_df) > N_SAMPLES_TO_REVIEW:
        sample_df = test_df.sample(n=N_SAMPLES_TO_REVIEW, random_state=RANDOM_STATE)
    else:
        sample_df = test_df
    print(f"Selected {len(sample_df)} random samples for review.")

    # Load the trained model
    n_classes = {col: 5 for col in metric_cols}
    model = MultiTaskBertModel(n_classes)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    print(f"Model loaded successfully from {MODEL_PATH}.")

    print("\n--- Qualitative Review ---")
    for index, row in sample_df.iterrows():
        text = row[TEXT_COLUMN]
        if not isinstance(text, str):
            continue

        predicted_labels = predict_scores(text, model, tokenizer, metric_cols)

        print(f"\n=================== SAMPLE {index} ===================")
        print(f"UTTERANCE: \"{text}\"")
        print("--------------------------------------------------")
        print(f"{'Metric':<40} | {'True Label':<12} | {'Predicted'}")
        print("--------------------------------------------------")

        for col in metric_cols:
            true_label = row[col]
            if pd.notna(true_label):
                predicted_label = predicted_labels[col]
                true_label_int = int(true_label)
                is_match = "✅" if true_label_int == predicted_label else "❌"
                print(f"{col:<40} | {true_label_int:<12} | {predicted_label} {is_match}")
        print("==================================================\n")

if __name__ == '__main__':
    main()
