import ast
import json
from pathlib import Path
import random
import pandas as pd # type: ignore
from datasets import Dataset
from transformers import ( # type: ignore
    AutoTokenizer)

MCQ_TRAINING_PATH = "datasets/train_dataset_mcq.csv"
SAQ_TRAINING_PATH = "datasets/train_dataset_saq.csv"

def generate_shuffled_variations(options, correct_key):
    """
    Generates 4 variations of the options dict, ensuring the correct answer
    rotates through A, B, C, and D.
    
    Args:
        options (dict): The original options dictionary (e.g., {'A': 'text', ...})
        correct_key (str): The key of the correct answer in the original dict (e.g., 'A')
        
    Returns:
        dict: A dictionary where keys are the NEW correct letters ('A', 'B', 'C', 'D')
              and values are the JSON strings of the shuffled options.
    """
    
    correct_text = options[correct_key]
    distractors = [text for key, text in options.items() if key != correct_key]
    keys = ["A", "B", "C", "D"]
    output_variations = {}

    for target_correct_letter in keys:

        current_distractors = distractors[:]
        random.shuffle(current_distractors)
        new_options = {}
        distractor_index = 0

        for key in keys:
            if key == target_correct_letter:
                new_options[key] = correct_text
            else:
                new_options[key] = current_distractors[distractor_index]
                distractor_index += 1

        output_variations[target_correct_letter] = new_options

    return output_variations

def create_training_dataset_mcq(csv_path=MCQ_TRAINING_PATH, use_all_answers:bool = False, seed:int =42):
    """
    Create a Hugging Face Dataset ready for LoRA training (MCQ).
    
    Args:
        csv_path: Path to the CSV file
        use_all_answers: choose to only use each entry once or shuffle it to multiply the dataset by 4
        
    Returns:
        Hugging Face Dataset object with 'messages' field
    """
    df = pd.read_csv(csv_path)
    training_examples = []

    for row in df.itertuples():
        prompt = row.prompt.strip()
        correct_answer = row.answer_idx.strip()
        choices = json.loads(row.choices)

        if use_all_answers:
            marker = '{"answer_choice":""}'
            index = prompt.find(marker)
            if index != -1:
                prompt = prompt[:index + len(marker)]

            answers = generate_shuffled_variations(choices, correct_answer)

            for letter, answer in answers.items():
                completion = json.dumps({"answer_choice": letter})
                formatted_options_list = []
                for key, value in answer.items():
                    formatted_options_list.append(f"{key}. {value}")
                formatted_options_str = "\n".join(formatted_options_list)
                combined_question = prompt + "\n\n" + formatted_options_str
                training_examples.append({
                    'messages': [
                        {"role": "user", "content": combined_question},
                        {"role": "assistant", "content": completion},
                    ],
                    'mcqid': row.MCQID
                })
        else:
            completion = json.dumps({"answer_choice": correct_answer})
            training_examples.append({
                'messages': [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": completion}
                ],
                'mcqid': row.MCQID
            })

    
    dataset = Dataset.from_list(training_examples).shuffle(seed)
    return dataset

def create_saq_prompt(question: str) -> str:
    """
    Create a formatted prompt for SAQ questions.
    
    Args:
        question: The question text
        
    Returns:
        Formatted prompt string
    """
    return f"{question} Provide ONLY the exact answer without explanation."

def create_training_dataset_saq(csv_path=SAQ_TRAINING_PATH, use_all_answers=False, 
                                 weight_sampling=False, seed=42):
    """
    Create a Hugging Face Dataset ready for LoRA training (SAQ).
    
    Args:
        csv_path: Path to the CSV file
        use_all_answers: If True, create multiple training examples per question 
                        (one for each valid answer). If False, use only best answer.
        weight_sampling: If True and use_all_answers=True, repeat examples based 
                        on their weight/count. If False, each answer appears once.
        seed: Random seed for shuffling
        
    Returns:
        Hugging Face Dataset object with 'messages' field
    """
    df = pd.read_csv(csv_path)
    training_examples = []
    
    for row in df.itertuples():
        en_question = row.en_question.strip()
        annotations = ast.literal_eval(row.annotations)
        idks = ast.literal_eval(row.idks)
        merged_answers = {}
        
        for item in annotations:
            if item.get('en_answers'):
                answer_key = item['en_answers'][0]
                merged_answers[answer_key] = item['count']

        if isinstance(idks, dict):
            merged_answers.update(idks)
        
        prompt = create_saq_prompt(en_question)
        
        if use_all_answers:
            valid_answers = [k for k, v in merged_answers.items() if v > 0]
            
            for answer in valid_answers:
                repeat_count = merged_answers[answer] if weight_sampling else 1
                
                example = {
                    'messages': [
                        {"role": "user", "content": f"{prompt} Target Score: {merged_answers[answer]}"},
                        {"role": "assistant", "content": answer}
                    ],
                    'id': row.ID
                }

                for _ in range(repeat_count):
                    training_examples.append(example.copy())
           
        else:
            best_answer = max(merged_answers.items(), key=lambda x: x[1])[0]
            
            training_examples.append({
                'messages': [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": best_answer}
                ],
                'id': row.ID
            })
    
    dataset = Dataset.from_list(training_examples)
    
    if use_all_answers and weight_sampling:
        dataset = dataset.shuffle(seed=seed)
    
    return dataset


def create_training_data_tokenized(task_type: str, tokenizer, seed: int = 42,
                          use_all_answers: bool = False, weight_sampling: bool = False, debug:bool = False):
    """
    Create train/validation splits for training.
    
    Args:
        task_type: Either 'mcq' or 'saq'
        tokenizer: The tokenizer to use
        seed: Random seed for reproducibility
        use_all_answers: enlarge the datasets
        weight_sampling: (SAQ only) Sample proportionally to answer weights
        debug: Print debug information
    """

    if task_type.lower() == 'mcq':
        dataset = create_training_dataset_mcq(MCQ_TRAINING_PATH, use_all_answers, seed)
    elif task_type.lower() == 'saq':
        dataset = create_training_dataset_saq(
            SAQ_TRAINING_PATH, 
            use_all_answers,
            weight_sampling,
            seed
        )
    else:
        raise ValueError(f"task_type must be 'mcq' or 'saq', got {task_type}")
    
    def tokenize_and_mask(examples):
        """
        Tokenization for INSTRUCT models.
        Uses the chat template and masks everything before assistant's response.
        Also masks the EOT token so model doesn't learn to append it.
        """
        input_ids_list = []
        labels_list = []
        
        for messages in examples["messages"]:
            
            # Apply chat template
            input_ids = tokenizer.apply_chat_template(
                messages,
                truncation=True,
                max_length=2048,
                add_generation_prompt=False,
                padding=False,
            )
            
            labels = list(input_ids)
            full_text = tokenizer.decode(input_ids)
            
            # For LLaMA Instruct models
            if "<|start_header_id|>assistant<|end_header_id|>" in full_text:
                assistant_header = "<|start_header_id|>assistant<|end_header_id|>"
                last_assistant_idx = full_text.rfind(assistant_header)
                
                if last_assistant_idx != -1:
                    content_start = last_assistant_idx + len(assistant_header)
                    while content_start < len(full_text) and full_text[content_start] in '\n\r':
                        content_start += 1
                    
                    prefix_text = full_text[:content_start]
                    prefix_tokens = tokenizer.encode(prefix_text, add_special_tokens=False)
                    mask_length = min(len(prefix_tokens), len(labels))
                    labels[:mask_length] = [-100] * mask_length
                
                # Mask the EOT token at the end
                eot_token = "<|eot_id|>"
                eot_id = tokenizer.encode(eot_token, add_special_tokens=False)
                if len(eot_id) > 0 and len(input_ids) >= len(eot_id):
                    # Check if last tokens are EOT
                    if input_ids[-len(eot_id):] == eot_id:
                        labels[-len(eot_id):] = [-100] * len(eot_id)
            
            # For Mistral Instruct models
            elif "[/INST]" in full_text:
                sep_ids = tokenizer.encode("[/INST]", add_special_tokens=False)
                sep_len = len(sep_ids)
                
                start_idx = -1
                for i in range(len(input_ids) - sep_len, -1, -1):
                    if input_ids[i : i+sep_len] == sep_ids:
                        start_idx = i + sep_len
                        break
                
                if start_idx != -1:
                    labels[:start_idx] = [-100] * start_idx
                
                # Mask the EOS token at the end for Mistral
                eos_token = "</s>"
                eos_id = tokenizer.encode(eos_token, add_special_tokens=False)
                if len(eos_id) > 0 and len(input_ids) >= len(eos_id):
                    if input_ids[-len(eos_id):] == eos_id:
                        labels[-len(eos_id):] = [-100] * len(eos_id)
            
            input_ids_list.append(input_ids)
            labels_list.append(labels)

        return {
            "input_ids": input_ids_list,
            "labels": labels_list
        }

    dataset = dataset.map(
        tokenize_and_mask,
        batched=True,
        remove_columns=dataset.column_names,
        desc=f"Tokenizing {task_type.upper()}"
    )
    
    if debug:
        print("\n--- DEBUGGING DATA MASKING ---")
        sample = dataset[0] 
        input_ids = sample['input_ids']
        labels = sample['labels']

        print(f"Total Input Length: {len(input_ids)}")

        # Decode the full input
        decoded_input = tokenizer.decode(input_ids)
        print(f"\n[FULL INPUT]:\n{decoded_input[:500]}...")

        # Decode only the labels (what model learns)
        valid_labels = [l for l in labels if l != -100]
        decoded_labels = tokenizer.decode(valid_labels)

        print(f"\n[GRADED LABELS] (This is what the model learns):")
        print(f"'{decoded_labels}'")
        
        # Show where masking occurs
        mask_count = sum(1 for l in labels if l == -100)
        print(f"\nMasked tokens: {mask_count}/{len(labels)}")
        print("-------------------------------\n")

    return dataset


if __name__ == "__main__":
    print("="*70)
    print("MCQ Dataset")
    print("="*70)
    
    mcq_dataset = create_training_dataset_mcq(MCQ_TRAINING_PATH, True, 42)
    print(f"Total MCQ examples: {len(mcq_dataset)}")
    print(f"Dataset columns: {mcq_dataset.column_names}")
    print("\nFirst MCQ example:")
    for msg in mcq_dataset[0]['messages']:
        print(f"  {msg['role']}: {msg['content']}")
    

    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Meta-Llama-3-8B-Instruct",
        cache_dir="./Model-Cache",
        trust_remote_code=True
    )
    data = create_training_data_tokenized(task_type='mcq', tokenizer=tokenizer, debug=True)
    
    print("\n" + "="*70)
    print("SAQ Dataset - BEST ANSWER ONLY")
    print("="*70)
    
    saq_best = create_training_dataset_saq(
        SAQ_TRAINING_PATH,
        use_all_answers=False
    )
    print(f"Total SAQ examples (best only): {len(saq_best)}")
    print("\nFirst SAQ example:")
    for msg in saq_best[0]['messages']:
        print(f"  {msg['role']}: {msg['content']}")