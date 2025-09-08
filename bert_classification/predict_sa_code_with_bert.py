import numpy as np
import pandas as pd
import re
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from input import meeting


def load_sa_model(model_dir='./sa_bert_model_multilabel'):
    """
    loads fine-tuned model and tokenizer
    """
    print(f"Loading model from '{model_dir}'...")
    try:
        tokenizer = BertTokenizer.from_pretrained(model_dir)
        model = BertForSequenceClassification.from_pretrained(model_dir)
        print("Model loaded successfully.")
        return model, tokenizer
    except OSError:
        print(f"Error: Model not found in directory")
        return None, None
    

def predict_sa_utterance(text, model, tokenizer, labels, threshold=0.5):
    """
    predicts SA category for single utterance using fine-tuned model
    """
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    #predicted_class_idx = torch.argmax(outputs.logits, dim=1).item()
    probabilities = torch.sigmoid(logits).squeeze().cpu().numpy()

    predicted_label_indices = np.where(probabilities > threshold)[0]
    predicted_labels = [labels[i] for i in predicted_label_indices]

    return predicted_labels



def analyze_meeting_with_bert(meeting_text, model, tokenizer, label_map) -> pd.DataFrame:
    """
    parses full meeting transcript, and predicts SA category using fine-tuned BERT model
    """

    #regex
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
            
            #predict SA category using BERT
            predicted_sa_category = predict_sa_utterance(clean_utterance, model, tokenizer, label_map)

            prediction = {
                "Date": current_date,
                "Timestamp": timestamp,
                "Speaker": speaker.replace("(Team Leader)", "").strip(),
                "Utterance": clean_utterance,
                "SA_Category_Predicted": predicted_sa_category
            }

            parsed_data.append(prediction)

    return pd.DataFrame(parsed_data)


if __name__ == '__main__':
    SA_LABELS = [1, 2, 3, 4]


    #load fine-tuned model
    sa_model, sa_tokenizer = load_sa_model()

    if sa_model and sa_tokenizer:
        meeting_input = meeting

        #analysis
        results_df = analyze_meeting_with_bert(meeting_input, sa_model, sa_tokenizer, SA_LABELS)

        print("\n -- Meeting Analysis Results ---")
        print(results_df.to_string())

        #save to csv
        output_path = "predicted_bert_sa_category.csv"
        results_df.to_csv(output_path, index=False)
        print(f"BERT meeting analysis results saved to {output_path}")


