import json
import time
import argparse
import os
from tqdm import tqdm
from openai import OpenAI
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Tuple


USER_PROMPT_TEMPLATE = """You are a rigorous evaluator for an out-of-view (OOV) multi-choice visual question answering task.

Your job is to determine whether the model's response is logically consistent with the ground truth and factual correct. Notice that the image is not provided for this evaluation. Evaluate both the option-level rationale and the final answer justification holistically.

Case of INCORRECT:
1. Any answers based primarily on guessing, imagination or scene stereotypes without referring to image evidence.
2. Any rationale that only say "not visible", "unlikely", "probably", or other vague statements without supporting evidence.
3. Missing the rationale for any five options.
4. Responses that omit essential evidence, or only provide the final choice should be considered incorrect. 

Important reminders:
- Long explanations do NOT necessarily mean correct reasoning.
- Be strict with each criterion.

Question: {question}

Options: {options}

Ground Truth Answer: {answer_reasoning} {answer}

Response: {response}

Respond ONLY with one of the following:

<answer>Yes</answer>
or
<answer>No</answer>
"""

def parse_response(raw_response):
    if raw_response is None:
        return "Bad format"
    if len(raw_response) == 1:
        return raw_response
    try:
        patterns = [
            ("<answer>", "</answer>"),
            ("\\boxed{", "}"),
            ("<|begin_of_box|>", "<|end_of_box|>")
        ]
        response = "Bad format"
        for start, end in patterns:
            if start in raw_response and end in raw_response:
                candidate = raw_response.split(start, 1)[1].split(end, 1)[0].strip()
                if len(candidate) == 1:
                    response = candidate
                    break
    except:
        response = "Bad format"
    return response

class OpenAI_model:
    def __init__(self, api_key, base_url, model_name):
        self.model = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

    def chat(self, query=None, max_tokens=128, temperature=0.0, top_p=0.95):
        trials = 0
        messages = [
            {
                "role": "system", 
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": query
            }
        ]

        while trials < 3:
            try:
                response = self.model.chat.completions.create(
                    model=self.model_name,
                    messages=messages, 
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"Retrying... {trials + 1}/3; Error: {e}")
                # sleep for 1 second
                time.sleep(2 ** trials)
                trials += 1
        return None

class ThreadSafeResultHandler:
    def __init__(self, output_json):
        self.output_json = output_json
        self.results = []
        self.lock = threading.Lock()

        if os.path.exists(output_json):
            with open(output_json) as f:
                self.results = json.load(f)
        
    def add_result(self, result):
        with self.lock:
            self.results.append(result)
            # Save incrementally to avoid losing progress
            with open(self.output_json, "w") as f:
                json.dump(self.results, f, indent=4)

def evaluate_single_item(item: Dict, model: OpenAI_model) -> Tuple[str, str, bool]:
    """
    Evaluate a single item and return (qa_id, response, is_correct).
    This function is thread-safe as each thread has its own model instance.
    """
    qa_id = item["id"]
    question = item["question"]
    options = item["options"]
    option_string = '\n'.join([letter + '. ' + option for letter, option in zip('ABCDE', options)])
    answer_rationale = item["answer_reasoning"] + f" <answer>{item['answer']}</answer>"
    response = item["response"]

    if "<think>" in response and "</think>" in response:
        # skip the thinking part
        response = "".join(response.split("</think>")[1:]).strip()

    if len(response) < 30:
        return qa_id, "Skipped call", False

    user_query = USER_PROMPT_TEMPLATE.format(
        question=question, options=option_string,
        answer_rationale=answer_rationale, response=response
    )
    
    api_response = model.chat(query=user_query)
    
    # parse the response
    try:
        answer = api_response.split("<answer>")[1].split("</answer>")[0].strip()
    except:
        answer = "Bad format"
    
    is_correct = answer.lower() == "yes"
    return qa_id, api_response, is_correct

def process_eval_item(model, item, result_handler):
    """Process a single evaluation item in a thread"""
    qa_id, api_response, is_correct = evaluate_single_item(item, model)
        
    # Create result object similar to inference_api structure
    result = {
        "id": qa_id,
        "category": item.get("category", None),
        "outdoor": item.get("outdoor", None),
        "question_type": item.get("question_type", None),
        "question": item["question"],
        "options": item["options"],
        "answer_reasoning": item["answer_reasoning"],
        "response_reasoning": item["response"],
        "eval_response": api_response,
        "is_correct": is_correct
    }
    
    result_handler.add_result(result)
    return True

def main(args):
    output_path = args.output_path
    eval_output_path = args.eval_output_json
    
    with open(output_path, "r") as f:
        data = json.load(f)
    total = len(data)
    
    # Initialize result handler
    result_handler = ThreadSafeResultHandler(eval_output_path)
    
    # skip the items that are already in the eval json file
    existings = set()
    if os.path.exists(eval_output_path):
        with open(eval_output_path) as f:
            eval_data = json.load(f)
            for item in eval_data:
                existings.add(item["id"])
    
    # Filter out already processed items
    remaining_items = []
    for item in data:
        if item["id"] in existings: continue
        
        response = parse_response(item["response"])
        if response == "Bad format":
            response = parse_response(item.get("response_answer", None))
        if item["answer"].lower() == response.lower():
            remaining_items.append(item)
    
    if not remaining_items:
        print("All items have already been processed!")
        # Calculate accuracy from existing results
    
        if os.path.exists(eval_output_path):
            with open(eval_output_path) as f:
                eval_data = json.load(f)
                
                directional_acc = sum(item.get("is_correct", 0) for item in eval_data if item["question_type"] == "directional")
                contextual_acc = sum(item.get("is_correct", 0) for item in eval_data if item["question_type"] == "contextual")
                
                acc = sum(item.get("is_correct", 0) for item in eval_data)

                directional_total = sum(1 for item in eval_data if item["question_type"] == "directional") + 1e-5
                contextual_total = sum(1 for item in eval_data if item["question_type"] == "contextual") + 1e-5
                total = len(eval_data)

        print(f"Total: {total}")
        print(f"Directional Reasoning Accuracy: {directional_acc / directional_total * 100}%")
        print(f"Contextual Reasoning Accuracy: {contextual_acc / contextual_total * 100}%")
        print(f"Reasoning Accuracy: {acc / total * 100}%")
        return
    print(f"Processing {len(remaining_items)} remaining items with {args.num_threads} threads...")
    
    def process_item(item):
        """Process a single item with its own model instance"""
        model = OpenAI_model(api_key=args.api_key, base_url=args.base_url, model_name=args.model)
        return process_eval_item(model, item, result_handler)
    
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        future_to_item = {executor.submit(process_item, item): item for item in remaining_items}
        for future in tqdm(as_completed(future_to_item), total=len(remaining_items), desc="Evaluating"):
            item = future_to_item[future]
            try:
                future.result()
            except Exception as e:
                print(f"Unexpected error processing {item['id']}: {e}")

    # Calculate final accuracy from all results
    if os.path.exists(eval_output_path):
        with open(eval_output_path) as f:
            eval_data = json.load(f)
            directional_acc = sum(1 for item in eval_data if item.get("is_correct", False) and item["question_type"] == "directional")
            contextual_acc = sum(1 for item in eval_data if item.get("is_correct", False) and item["question_type"] == "contextual")
            acc = sum(1 for item in eval_data if item.get("is_correct", False))

            directional_total = sum(1 for item in eval_data if item["question_type"] == "directional")
            contextual_total = sum(1 for item in eval_data if item["question_type"] == "contextual")
            total = len(eval_data)

    print(f"Total: {total}")
    print(f"Directional Reasoning Accuracy: {directional_acc / directional_total * 100}%")
    print(f"Contextual Reasoning Accuracy: {contextual_acc / contextual_total * 100}%")
    print(f"Reasoning Accuracy: {acc / total * 100}%")

    # save the accuracy
    with open(output_path[:-5] + f"/{args.model}_reasoning_eval.txt", "w") as f:
        f.write(f"Total: {total}\n")
        f.write(f"Directional Reasoning Accuracy: {directional_acc / directional_total * 100}%\n")
        f.write(f"Contextual Reasoning Accuracy: {contextual_acc / contextual_total * 100}%\n")
        f.write(f"Reasoning Accuracy: {acc / total * 100}%\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output JSON file.")
    parser.add_argument("--eval_output_json", type=str, help="Path to the evaluation output JSON file.")
    parser.add_argument("--model", type=str, default="gpt-5-mini", help="Model name")
    parser.add_argument("--num_threads", type=int, default=8, help="Number of threads for parallel processing (default: 16)")
    args = parser.parse_args()

    if "gpt" in args.model:
        args.api_key = os.environ.get("OPENAI_API_KEY") or "YOUR_OPENAI_API_KEY"
        args.base_url = "https://api.openai.com/v1"

    if args.eval_output_json is None:
        args.eval_output_json = args.output_path[:-5] + f"/{args.model}_rationale_acc.json"
    os.makedirs(os.path.dirname(args.eval_output_json), exist_ok=True)
    
    main(args)