import pandas as pd
import torch
import torch.nn as nn
import re
from transformers import BertTokenizer, BertModel
from input import meeting


#create class for custom model
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

#function to load multi-task model
def load_multi_task_model(model_path, n_classes_dict):
    print(f"Loading model from '{model_path}'...")
    try:
        model = MultiTaskBertModel(n_classes_dict)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        print("Model loaded successfully.")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None
    
#predict scores for a given utterance
def predict_all_scores(text, model, tokenizer, metric_cols):
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
    for col_name  in metric_cols:
        pred_index = torch.argmax(outputs[col_name], dim=1).item()
        predictions[col_name] = pred_index + 1
    return predictions



#analyze meeting texts
def analyze_meeting_with_multitask_bert(meeting_text, model, tokenizer, metric_cols):
    pattern = re.compile(r'((?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday), (?:January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2}, \d{4})|(\d{1,2}:\d{2}\sAM)\s*[-—]\s*(.*?):\s*[""](.*?)[""]', re.DOTALL)
    matches = pattern.findall(meeting_text)

    SAFETY_COLS = [col for col in metric_cols if col.startswith('sqdcp_') and any(feat in col for feat in ['Hazard_Identification', 'Escalation_Criteria_Stated', 'Escalation_Decision_Made', 'Proactive_Phrasing', 'PPE___Compliance_Visibility', 'Safety_Coaching_Language', 'Emotional_Awareness', 'Silence_on_Safety'])]
    QUALITY_COLS = [col for col in metric_cols if col.startswith('sqdcp_') and any(feat in col for feat in ['Clear_Defect_Description', 'Root_Cause_Exploration', 'Trend_Recognition', 'Voice_of_Operator_Reflected', 'Turn_Taking_Balance', 'Containment_vs__Root_Fix_Split', 'Quality_Tradeoff_Framing'])]
    DELIVERY_COLS = [col for col in metric_cols if col.startswith('sqdcp_') and any(feat in col for feat in ['Clear_Deviation_Stated', 'Projection_Statement', 'Containment_Action_Identified', 'Accountability_Assigned', 'Escalation_Decision_Made', 'Trend_Recognition', 'Voice_of_Operator_Reflected', 'Voice_of_Customer_Reflected'])]
    COST_COLS = [col for col in metric_cols if col.startswith('sqdcp_') and any(feat in col for feat in ['Scrap_Waste_Acknowledged', 'Rework_Time_Stated', 'Overtime_Justified_or_Flagged', 'Downtime_Cost_Noted', 'Labor_Allocation_Awareness', 'Material_Waste_Inventory_Excess', 'Tool_Equipment_Cost_Impact', 'Cost_vs__Risk_Tradeoff_Framing', 'Prioritization_Based_on_Cost', 'Coaching_Language_on_Efficiency', 'Voice_of_Customer_Cost'])]
    PEOPLE_COLS = [col for col in metric_cols if col.startswith('sqdcp_') and any(feat in col for feat in ['Feedback_Quality__Tier_1_2_', 'Participation_Inclusivity', 'Emotional_Intelligence_in_Language', 'Dialogue_Invitation', 'Recognition_of_Contributions', 'New_Hire_Status_Shared', 'Training_Progress_Shared', 'Training_Matrix_Awareness', 'Mentoring_Support_Mentioned', 'Workload_or_Morale_Reflected'])]


    parsed_data = []
    current_date = None
    for match in matches:
        if match[0]:
            current_date = match[0]
        else:
            timestamp, speaker, utterance = match[1], match[2], match[3]
            clean_utterance = utterance.replace('\n', ' ').strip()

            #predict base scores
            predicted_scores = predict_all_scores(clean_utterance, model, tokenizer, metric_cols)

            final_prediction_row = {
                "Date": current_date,
                "Timestamp": timestamp,
                "Speaker": speaker.replace("(Team Leader)", "").strip(),
                "Utterance": clean_utterance,
                **predicted_scores
            }

            #Aggregate as needed
            #define dimension lists for feedback, deviation, and communication
            COMM_DIMS = [col for col in metric_cols if col.startswith('comm_')]
            FEEDBACK_DIMS = [col for col in metric_cols if col.startswith('feedback_')]
            DEV_DIMS = [col for col in metric_cols if col.startswith('deviation_')]
            

            final_prediction_row['Total_Deviation_Score'] = sum(predicted_scores.get(dim, 0) for dim in DEV_DIMS)
            final_prediction_row['Total_Comm_Score'] = sum(predicted_scores.get(dim, 0) for dim in COMM_DIMS)
            final_prediction_row['Feedback_Tier1_Score'] = sum(predicted_scores.get(dim, 0) for dim in FEEDBACK_DIMS[:6])
            final_prediction_row['Feedback_Tier2_Score'] = sum(predicted_scores.get(dim, 0) for dim in FEEDBACK_DIMS[6:])
            final_prediction_row['Total_Safety_Score'] = sum(final_prediction_row.get(col, 0) for col in SAFETY_COLS)
            final_prediction_row['Total_Quality_Score'] = sum(final_prediction_row.get(col, 0) for col in QUALITY_COLS)
            final_prediction_row['Total_Delivery_Score'] = sum(final_prediction_row.get(col, 0) for col in DELIVERY_COLS)
            final_prediction_row['Total_Cost_Score'] = sum(final_prediction_row.get(col, 0) for col in COST_COLS)
            final_prediction_row['Total_People_Score'] = sum(final_prediction_row.get(col, 0) for col in PEOPLE_COLS)

            parsed_data.append(final_prediction_row)

    return pd.DataFrame(parsed_data)


if __name__ == '__main__':
    #load datafile model
    df = pd.read_csv('master_training_data.csv')
    ALL_METRIC_COLS = [col for col in df.columns if col.startswith(('comm_', 'feedback_', 'deviation_', 'sqdcp_'))]

    #dictionary to define model's structure
    n_classes = {col: 5 for col in ALL_METRIC_COLS}

    #load fine-tuned model
    multi_task_model, mt_tokenizer = load_multi_task_model('./multi_task_bert_model.pth', n_classes)

    if multi_task_model and mt_tokenizer:
        meeting_input = meeting

        #analysis
        results_df = analyze_meeting_with_multitask_bert(meeting_input, multi_task_model, mt_tokenizer, ALL_METRIC_COLS)

        print("\n -- Meeting Analysis Results ---")
        display_cols = ['Speaker', 'Utterance', 'comm_Pausing', 'feedback_Timely', 'deviation_Current_Target', 'Feedback_Tier1_Score','Total_Comm_Score']
        display_cols = [col for col in display_cols if col in results_df.columns]
        
        if display_cols:
            print(results_df[display_cols].to_string())
            results_df[display_cols].to_csv("test_analysis.csv", index=False)
        

        #save to CSV
        output_path = "final_meeting_bert_analysis.csv"
        results_df.to_csv(output_path, index=False)
        print(f"\nFull meeting analysis saved to {output_path}")