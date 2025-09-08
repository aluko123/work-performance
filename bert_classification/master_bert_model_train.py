import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
import os

def create_torch_dataset(encodings, labels):
    """
    A helper class to format data for the trainer
    """
    class TorchDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.lables[idx])
            return item
        
        def __len__(self):
            return len(self.labels)
        
    return TorchDataset(encodings, labels)

def fine_tune_model(job_config):
    """
    A generic function to fine-tnue a BERT model based on a configuration dictionary
    """
    print(f"\n{'='*20} Starting Training Job: {job_config['model_name']} {'='*20}")

    # load and prepare data
    try:
        df = pd.read_csv(job_config['csv_path'])
    except FileNotFoundError:
        print(f"Error: File not found at {job_config['csv_path']}. Skipping job.")
        return

    df.dropna(subset=[job_config['text_col'], job_config['label_col']], inplace=True)


    #clean scores if they are in string format like "5 - Exemplary"
    if df[job_config['label_col']].dtype == 'object':
        df[job_config['label_col']] = pd.to_numeric(df[job_config['label_col']].astype(str).str.split(r'\s*[--]\s*').str[0], errors='coerce')
        df.dropna(subset=[job_config['label_col']], inplace=True)
    
    unique_labels = sorted(df[job_config['label_col']].unique())
    label_map = {label: i for i, label in enumerate(unique_labels)}
    df['label'] = df[job_config['label_col']].map(label_map)
    num_labels = len(unique_labels)

    if num_labels < 2:
        print(f"Error: Not enough unique labels ({num_labels}) for '{job_config['label_col']}'. Skipping job.")
        return

    print(f"Training '{job_config['label_col']}' with {num_labels} unique labels: {label_map}")


    #load pre-trained BERT model and tokenizer
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)


    #prepare datasets for training
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df[job_config['label_col']].tolist(), df['label'].tolist(), test_size=0.2, random_state=42
    )

    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

    train_dataset = create_torch_dataset(train_encodings, train_labels)
    val_dataset = create_torch_dataset(val_encodings, val_labels)


    #define training arguments and train
    training_args = TrainingArguments(
        output_dir=f"./results_{job_config['model_name']}",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",          
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,   
    )


    print(f"Starting fine-tuning for '{job_config['label_col']}'...")
    trainer.train()
    print(f"Fine-tuning complete")


    # save model
    os.makedirs(job_config['model_save_path'], exist_ok=True)
    model.save_pretrained(job_config['model_save_path'])
    tokenizer.save_pretrained(job_config['model_save_path'])
    print(f"Model for '{job_config['model_name']}' saved to '{job_config['model_save_path']}'")


#main
if __name__ == '__main__':
    #CONFIGURING TRAINING JOBS
    data_dir = '../tf-idf/'

    #define list of dimensions for each category
    COMM_PERF_DIMENSIONS = [
    'Pausing', 'Verbal Affirmation', 'Continuation Prompts', 'Paraphrasing', 'Clarifying Questions',
     'Probing Questions', 'Open-Ended Questions', 'Coaching Questions', 'Acknowledgment of Emotions', 'Summary Statements'
    ]

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
    
    FEEDBACK_BEHAVIOR_DIMENSIONS = [
    'Timely', 'Neutral & Specific', 'Impact + Emotion', 'Action-Oriented', 'Clarity of Situation',
    'Feedback for Reinforcement', 'Feedback for Improvement', '"I" + Sensory Statements', 
    'Objective Framing', 'Avoids Mind Reading', 'Behavior + Pattern Awareness', 'Invites Dialogue'
    ]

    DEV_BEHAVIOR_DIMENSIONS = [
    'Current Target', 'Deviation from Target', 'Process Location & Impact', 'Reason for Deviation (“Why”)', 'Standard Confirmation',
    'Containment / Short-Term Fix', 'Actions Taken to Understand Deviation', 'Trend Recognition', 'Decision Made', 
    'Define Next Actions', 'Report Back Plan', 'Follow-Up on Open Actions', 'Preventive Action / System Fix', 'Spillover / Risk Awareness', 
    'Operator Voice Acknowledged', 'Summary & Prioritization of Issues', 'Go&See (Gemba Walk)'
    ]

    training_jobs = []

    #add jobs for Communication Performance
    for dim in COMM_PERF_DIMENSIONS:
        training_jobs.append({
            'model_name': f'Comm_Perf_{dim}',
            'csv_path': os.path.join(data_dir, 'comm_perf', 'comm_data.csv'),
            'text_col': 'Utterance',
            'label_col': dim,
            'model_save_path': f'./models/comm_perf_{dim.lower().replace(" ", "_")}_bert'     
        })  

    #add jobs for feedback behavior
    for dim in FEEDBACK_BEHAVIOR_DIMENSIONS:
        training_jobs.append({
            'model_name':f'Feeback_Behavior_{dim}',
            'csv_path': os.path.join(data_dir, 'feedback', 'feedback_data.csv'),
            'text_col': 'Utterance',
            'label_col': dim,
            'model_save_path': f'./models/feedback_behavior_{dim.lower().replace(" ", "_").replace("&", "_").replace("+", "_")}_bert'
        })


    # add jobs for deviation data
    for dim in DEV_BEHAVIOR_DIMENSIONS:
        training_jobs.append({
            'model_name': f'Deviation_Behavior_{dim}',
            'csv_path': os.path.join(data_dir, 'deviation_data.csv'),
            'text_col': 'Utterance',
            'label_col': dim,
            'model_save_path': f'./models/deviation_behavior_{dim.lower().replace(" ", "_").replace("&", "_").replace("+", "_")}_bert'  
        })

    # add jobs for safety data
    for dim in SAFETY_FEATURES:
        training_jobs.append({
            'model_name': f'Safety_features_{dim}',
            'csv_path': os.path.join(data_dir, 'sqdcp', 'SQDCP_Data.csv'),
            'text_col': 'Utterance',
            'label_col': dim,
            'model_save_path': f'./models/safety_features_{dim.lower().replace(" ", "_").replace("&", "_").replace("+", "_")}_bert'  
        })

    # add jobs for people data
    for dim in PEOPLE_FEATURES:
        training_jobs.append({
            'model_name': f'People_features_{dim}',
            'csv_path': os.path.join(data_dir, 'sqdcp', 'SQDCP_Data.csv'),
            'text_col': 'Utterance',
            'label_col': dim,
            'model_save_path': f'./models/people_features_{dim.lower().replace(" ", "_").replace("&", "_").replace("+", "_")}_bert'  
        })
 
    # add jobs for quality data
    for dim in QUALITY_FEATURES:
        training_jobs.append({
            'model_name': f'Quality_features_{dim}',
            'csv_path': os.path.join(data_dir, 'sqdcp', 'SQDCP_Data.csv'),
            'text_col': 'Utterance',
            'label_col': dim,
            'model_save_path': f'./models/quality_features_{dim.lower().replace(" ", "_").replace("&", "_").replace("+", "_")}_bert'  
        })
    
    # add jobs for cost data
    for dim in COST_FEATURES:
        training_jobs.append({
            'model_name': f'Cost_features_{dim}',
            'csv_path': os.path.join(data_dir, 'sqdcp', 'SQDCP_Data.csv'),
            'text_col': 'Utterance',
            'label_col': dim,
            'model_save_path': f'./models/cost_features_{dim.lower().replace(" ", "_").replace("&", "_").replace("+", "_")}_bert'  
        })
    
    # add jobs for delivery data
    for dim in DELIVERY_FEATURES:
        training_jobs.append({
            'model_name': f'Delivery_features_{dim}',
            'csv_path': os.path.join(data_dir, 'sqdcp', 'SQDCP_Data.csv'),
            'text_col': 'Utterance',
            'label_col': dim,
            'model_save_path': f'./models/delivery_features_{dim.lower().replace(" ", "_").replace("&", "_").replace("+", "_")}_bert'  
        })
    
    print(f"Found {len(training_jobs)} models to train")

    for job in training_jobs:
        fine_tune_model(job)
    
    print("\nAll training jobs complete!")


"""
Issues with this: Our model doesn't have enough labels to go off, preventing it from labeling correctly
Trying oneshotting classifier in another file
"""