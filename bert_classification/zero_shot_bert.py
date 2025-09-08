import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split


#define list of dimensions for each category
COMM_PERF_DIMENSIONS = [
'Pausing', 'Verbal Affirmation', 'Continuation Prompts', 'Paraphrasing', 'Clarifying Questions',
    'Probing Questions', 'Open-Ended Questions', 'Coaching Questions', 'Acknowledgment of Emotions', 'Summary Statements'
]

SAFETY_FEATURES = [
    'Hazard Identification', 'Escalation Criteria Stated', 'Escalation Decision Made', 'Proactive Phrasing', 'PPE & Compliance Visibility', 'Safety Coaching Language', 'Emotional Awareness',
'Silence on Safety']

QUALITY_FEATURES = ['Clear Defect Description', 'Root Cause Exploration', 'Trend Recognition_y', 'Voice of Operator Reflected', 'Turn-Taking Balance', 'Containment vs. Root Fix Split', 
    'Quality Tradeoff Framing']

DELIVERY_FEATURES = ['Clear Deviation Stated', 'Projection Statement', 'Proactive Phrasing', 'Containment Action Identified', 'Accountability Assigned', 'Escalation Decision Made',
    'Trend Recognition.1', 'Voice of Operator Reflected', 'Voice of Customer Reflected']

COST_FEATURES = ['Scrap/Waste Acknowledged', 'Rework Time Stated', 'Overtime Justified or Flagged', 'Downtime Cost Noted',
    'Labor Allocation Awareness', 'Material Waste/Inventory Excess', 'Tool/Equipment Cost Impact', 'Cost vs. Risk Tradeoff Framing', 'Prioritization Based on Cost', 'Coaching Language on Efficiency', 
    'Voice of Customer Cost']

PEOPLE_FEATURES = ['Feedback Quality (Tier 1/2)', 'Participation Inclusivity', 'Emotional Intelligence in Language', 'Dialogue Invitation', 'Recognition of Contributions', 'New Hire Status Shared',
    'Training Progress Shared', 'Training Matrix Awareness', 'Mentoring/Support Mentioned', 'Workload or Morale Reflected']


SQDCP_FEATURES = [
    'Hazard Identification', 'Escalation Criteria Stated', 'Escalation Decision Made', 'Proactive Phrasing', 'PPE & Compliance Visibility', 'Safety Coaching Language', 'Emotional Awareness',
    'Silence on Safety', 'Clear Defect Description', 'Root Cause Exploration', 'Trend Recognition', 'Voice of Operator Reflected', 'Turn-Taking Balance', 'Containment vs. Root Fix Split', 
    'Quality Tradeoff Framing', 'Clear Deviation Stated', 'Projection Statement', 'Proactive Phrasing', 'Containment Action Identified', 'Accountability Assigned', 'Escalation Decision Made',
    'Trend Recognition', 'Voice of Operator Reflected', 'Voice of Customer Reflected', 'Scrap/Waste Acknowledged', 'Rework Time Stated', 'Overtime Justified or Flagged', 'Downtime Cost Noted',
    'Labor Allocation Awareness', 'Material Waste/Inventory Excess', 'Tool/Equipment Cost Impact', 'Cost vs. Risk Tradeoff Framing', 'Prioritization Based on Cost', 'Coaching Language on Efficiency', 
    'Voice of Customer Cost', 'Feedback Quality (Tier 1/2)', 'Participation Inclusivity', 'Emotional Intelligence in Language', 'Dialogue Invitation', 'Recognition of Contributions', 'New Hire Status Shared',
    'Training Progress Shared', 'Training Matrix Awareness', 'Mentoring/Support Mentioned', 'Workload or Morale Reflected'
]

FEEDBACK_BEHAVIOR_DIMENSIONS = [
'Timely', 'Neutral & Specific', 'Impact + Emotion', 'Action-Oriented', 'Clarity of Situation',
'Feedback for Reinforcement', 'Feedback for Improvement', '"I" + Sensory Statements', 
'Objective Framing', 'Avoids Mind Reading', 'Behavior + Pattern Awareness', 'Invites Dialogue'
]

DEV_BEHAVIOR_DIMENSIONS = [
'Current Target', 'Deviation from Target', 'Process Location & Impact', 'Reason for Deviation (“Why”)', 'Standard Confirmation',
'Containment / Short-Term Fix', 'Actions Taken to Understand Deviation', 'Trend Recognition_x', 'Decision Made', 
'Define Next Actions', 'Report Back Plan', 'Follow-Up on Open Actions', 'Preventive Action / System Fix', 'Spillover / Risk Awareness', 
'Operator Voice Acknowledged', 'Summary & Prioritization of Issues', 'Go&See (Gemba Walk)'
]


#ALL_METRIC_COLS = COMM_PERF_DIMENSIONS + SAFETY_FEATURES + QUALITY_FEATURES + DELIVERY_FEATURES + COST_FEATURES + PEOPLE_FEATURES + FEEDBACK_BEHAVIOR_DIMENSIONS + DEV_BEHAVIOR_DIMENSIONS

# create pytorch dataset
class MultiTaskDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        labels = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_attention_mask=True,
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(labels, dtype=torch.long)
        }
    
#class for custome multi-task model architecture
class MutliTaskBertModel(nn.Module):
    def __init__(self, n_classes_dict):
        super(MutliTaskBertModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifiers = nn.ModuleDict({
            task_name: nn.Linear(self.bert.config.hidden_size, num_labels)
            for task_name, num_labels in n_classes_dict.items()
        })
    
    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask        
        )

        pooled_output = bert_output.pooler_output

        outputs = {
            task_name: classifier(pooled_output)
            for task_name, classifier in self.classifiers.items()
        }
        return outputs
    

#main execution
if __name__ == '__main__':
    # load data
    df = pd.read_csv('master_training_data.csv')
    
    ALL_METRIC_COLS = [col for col in df.columns if col.startswith(('comm_', 'feedback_', 'deviation_', 'sqdcp_'))]


    #prepare labels
    #for each metric, determine the number of unique labels
    n_classes = {col: 5 for col in ALL_METRIC_COLS}

    #Fill NaN labels with -100 for loss function to ignore
    for col in ALL_METRIC_COLS:
        df[col] = df[col] - 1 #map metric range to 0-4
        df[col] = df[col].fillna(-100).astype(int)

    #divide into train/validation split
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)


    # init model, tokenizer, loaders
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = MutliTaskBertModel(n_classes)
    
    # labels df
    #df_labels = df[ALL_METRIC_COLS]


    train_dataset = MultiTaskDataset(
        texts=train_df['Utterance'].to_numpy(),
        labels=train_df[ALL_METRIC_COLS].to_numpy(),
        tokenizer=tokenizer,
    )

    val_dataset = MultiTaskDataset(
        texts=val_df['Utterance'].to_numpy(),
        labels=val_df[ALL_METRIC_COLS].to_numpy(),
        tokenizer=tokenizer,
    )

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=8)

    #loop
    EPOCHS = 30
    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_dataloader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n\nUsing device: {device}")
    model = model.to(device)

    print("Starting Multi-Task Model Training")
    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0

        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            optimizer.zero_grad()

            #forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            #calculate combined loss
            combined_loss = torch.tensor(0.0, device=device)

            for i, task_name in enumerate(ALL_METRIC_COLS):
                task_logits = outputs[task_name]
                task_labels = labels[:, i]
                if not (task_labels == -100).all():
                    combined_loss += loss_fn(task_logits, task_labels)

            if combined_loss > 0:
                combined_loss.backward()
                optimizer.step()
                scheduler.step()
                total_train_loss += combined_loss.item()

        avg_train_loss = total_train_loss / len(train_dataloader)


        #validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

                #calculate combined loss
                combined_loss = torch.tensor(0.0, device=device)
                for i, task_name in enumerate(ALL_METRIC_COLS):
                    task_logits = outputs[task_name]
                    task_labels = labels[:, i]
                    if not (task_labels == -100).all():
                        combined_loss += loss_fn(task_logits, task_labels)

                total_val_loss += combined_loss.item()

        avg_val_loss = total_val_loss / len(val_dataloader)
        
        print(f"Epoch {epoch + 1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    output_model_path = './multi_task_bert_model.pth'
    torch.save(model.state_dict(), output_model_path)    
    print(f"Multi-Task Model Training Complete and saved to '{output_model_path}'")