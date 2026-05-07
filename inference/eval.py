import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run evaluation scripts")
    parser.add_argument("--output_path", type=str, help="Path to the output JSON file.")
    parser.add_argument("--model", type=str, default="deepseek-chat")
    args = parser.parse_args()
    
    output_path = args.output_path
    if not os.path.exists(output_path):
        raise ValueError(f"File does not exist: {output_path}")

    command_choice_accuracy = "python evaluation/eval_acc.py --output_path "
    command_rationale_accuracy = "python evaluation/eval_reasoning.py --output_path "
    
    # choice accuracy evaluation
    command = command_choice_accuracy + output_path
    os.system(command)
    
    # rationale accuracy evaluation
    command = command_rationale_accuracy + output_path + f" --model {args.model}"
    os.system(command)

if __name__ == "__main__":
    main()