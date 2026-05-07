import json
import argparse
import os

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

def main(args):
    with open(args.output_path, "r") as f:
        data = json.load(f)

    accuracy, total = 0, len(data)
    directional_acc, directional_total = 0, 0
    contextual_acc, contextual_total = 0, 0
    bad_format = 0
    incorrect_ids = []
    for item in data:
        raw_response = item["response"]
        response = parse_response(raw_response)

        question_id = item["id"]
        question_type = item["question_type"]

        # Try second field only if first response is bad format
        if response == "Bad format":
            response = parse_response(item.get("response_answer"))
            if response == "Bad format":
                bad_format += 1

        ans_correct = item["answer"].lower() == response.lower()
        if question_type == "directional":
            directional_acc += ans_correct
            directional_total += 1
        elif question_type == "contextual":
            contextual_acc += ans_correct
            contextual_total += 1

        accuracy += ans_correct
        if not ans_correct:
            incorrect_ids.append(question_id)

    print(f"Total: {total}")
    print(f"Directional Accuracy: {directional_acc / directional_total * 100}%")
    print(f"Contextual Accuracy: {contextual_acc / contextual_total * 100}%")
    print(f"Choice Accuracy: {accuracy / total * 100}%")
    print(f"Bad format: {bad_format}")

    # save the accuracy
    with open(args.output_path[:-5] + f"/choice_acc.txt", "w") as f:
        f.write(f"Total: {total}\n")
        f.write(f"Directional Accuracy: {directional_acc / directional_total * 100}%\n")
        f.write(f"Contextual Accuracy: {contextual_acc / contextual_total * 100}%\n")
        f.write(f"Choice Accuracy: {accuracy / total * 100}%\n")
        f.write(f"Bad format: {bad_format}\n")
        f.write(f"Incorrect IDs: {incorrect_ids}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate QA results")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output JSON file.")
    args = parser.parse_args()

    os.makedirs(args.output_path[:-5], exist_ok=True)
    main(args)