import os
import json
import requests
from openai import OpenAI

# --- Configuration ---
GOLDEN_DATASET_PATH = "data/golden_dataset.json"
EVALUATION_PROMPT_PATH = "evaluation_prompt.txt"
RAG_API_URL = os.getenv("RAG_API_URL")

# --- Main Functions ---
def load_golden_dataset(path):
    """Loads the golden dataset from a JSON file."""
    with open(path, 'r') as f:
        return json.load(f)

def load_evaluation_prompt(path):
    """Loads the evaluation prompt from a text file."""
    with open(path, 'r') as f:
        return f.read()

def get_rag_answer(question):
    """Queries the RAG API to get a generated answer."""
    payload = {"question": question}
    try:
        response = requests.post(RAG_API_URL, json=payload, stream=True)
        response.raise_for_status()

        answer = ""
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith('data:'):
                    try:
                        json_data = json.loads(decoded_line[len('data:'):])
                        if 'answer_token' in json_data:
                            answer += json_data['answer_token']
                    except json.JSONDecodeError:
                        # Handle cases where a line is not valid JSON
                        pass
        return answer
    except requests.exceptions.RequestException as e:
        print(f"Error calling RAG API: {e}")
        return None

def evaluate_answer(question, ideal_answer, generated_answer, prompt_template):
    """Uses an LLM to evaluate the generated answer."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    prompt = prompt_template.format(
        question=question,
        ideal_answer=ideal_answer,
        generated_answer=generated_answer
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert evaluator for a RAG system."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return None

def main():
    """Main function to run the RAG evaluation."""
    golden_dataset = load_golden_dataset(GOLDEN_DATASET_PATH)
    evaluation_prompt = load_evaluation_prompt(EVALUATION_PROMPT_PATH)

    for i, item in enumerate(golden_dataset):
        question = item["question"]
        ideal_answer = item["ideal_answer"]

        print(f"--- Evaluating Question {i+1}/{len(golden_dataset)} ---")
        print(f"Question: {question}")

        generated_answer = get_rag_answer(question)
        if generated_answer is None:
            continue

        print(f"Generated Answer: {generated_answer}")

        evaluation = evaluate_answer(question, ideal_answer, generated_answer, evaluation_prompt)
        if evaluation is None:
            continue

        print(f"Evaluation: {evaluation}")
        print("\n")

if __name__ == "__main__":
    main()
