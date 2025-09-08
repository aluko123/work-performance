import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch


def create_torch_dataset(encodings, labels):
    """format data to fit trainer"""

    class TorchDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels.astype(float)

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    return TorchDataset(encodings, labels)


def fine_tune_multilabel_sa_model():
    """
    load training_data.csv, prepare it and fine-tune on BERT for SA classification 
    """
    try:
        df = pd.read_csv('training_data.csv')
    except FileNotFoundError:
        print("Error: 'training_data.csv' not found.")
        return
    
    text_col = 'Utterance'
    label_col = 'SA Category'

    if text_col not in df.columns or label_col not in df.columns:
        print(f"Error: CSV must contain '{text_col}' and '{label_col}' columns.")
        return

    df.dropna(subset=[text_col, label_col], inplace=True)

    processed_labels = df[label_col].astype(str).str.split(',').apply(
        lambda codes: [int(float(c.strip())) for c in codes]
    )

    mlb = MultiLabelBinarizer(classes=[1, 2, 3, 4])
    encoded_labels = mlb.fit_transform(processed_labels)

    texts = df[text_col].tolist()
    
    

    # #mapping sa_codes to bert zero-index
    # unique_labels = sorted(df[label_col].unique())
    # label_map = {label: i for i, label in enumerate(unique_labels)}
    # df['label'] = df[label_col].map(label_map)
    # num_labels = len(unique_labels)

    # print(f"Labels mapped: {label_map}")

    #load BERT model
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=4,
        problem_type = "multi_label_classification"
    )

    #prep data
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, encoded_labels, test_size=0.2, random_state=42)

    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

    train_dataset = create_torch_dataset(train_encodings, train_labels)
    val_dataset = create_torch_dataset(val_encodings, val_labels)


    #define training settings
    training_args = TrainingArguments(
        output_dir = './results_multilabel',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        #warmup_steps=500,
        #weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_dir='./logs_multilabel',
        #logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    print("\nStarting the Multi-Label fine-tuning process...")
    trainer.train()
    print("Fine-tuning completed.")

    #save model for use later
    output_model_dir = './sa_bert_model_multilabel'
    model.save_pretrained(output_model_dir)
    tokenizer.save_pretrained(output_model_dir)
    print(f"Multi-label model saved to '{output_model_dir}'")


if __name__ == '__main__':
    fine_tune_multilabel_sa_model()