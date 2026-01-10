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

def create_training_dataset_mcq(csv_path=MCQ_TRAINING_PATH, use_all_answers:bool = False, debug:bool = False):
    """
    Create a Hugging Face Dataset ready for LoRA training (MCQ).
    
    Args:
        csv_path: Path to the CSV file
        use_all_answers: choose to only use each entry once or shuffel it to multiply the dataset by 4
        
    Returns:
        Hugging Face Dataset object with 'messages' field
    """
    df = pd.read_csv(csv_path)
    training_examples = []

    for idx, row in df.iterrows():
        prompt = row['prompt'].strip()
        correct_answer = row['answer_idx'].strip()
        choices = json.loads(row['choices'])
        #country = json.loads(row['choice_countries'])

        if use_all_answers:
            marker = '{"answer_choice":""}'
            index = prompt.find(marker)
            if index != -1:
                prompt = prompt[:index + len(marker)] # removing the choices to add them manually

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
                        {"role": "user", "content": "Why is this correct"},
                        {"role": "assistant", "content": f"Because '{answer[letter]}' is the correct answer"}
                    ],
                    'mcqid': row['MCQID']
                })
                if debug:
                    print(f"{training_examples[-1]}\n")
        else:
            training_examples.append({
                'messages': [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": completion}
                ],
                'mcqid': row['MCQID']
            })
            if debug:
                print(f"{training_examples[-1]}\n")

    
    dataset = Dataset.from_list(training_examples)
    return dataset

def create_saq_prompt(question: str) -> str:
    """
    Create a formatted prompt for SAQ questions.
    
    Args:
        question: The question text
        
    Returns:
        Formatted prompt string
    """
    return f"{question} Provide ONLY the exact answer without explanation. Provide not more than 4 word answers."

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
                
                # Use messages format
                example = {
                    'messages': [
                        {"role": "user", "content": prompt},
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
        csv_path: Path to the CSV file
        task_type: Either 'mcq' or 'saq'
        test_size: Fraction for validation (default 0.1)
        seed: Random seed for reproducibility
        use_all_answers: enlarge the datasets
        weight_sampling: (SAQ only) Sample proportionally to answer weights
        
    """

    # Llama 3 spacific chat template
    if tokenizer.chat_template is None:
        tokenizer.chat_template = (
            "{% set loop_messages = messages %}"
            "{% for message in loop_messages %}"
            #"{{ message['content'] | trim +'\n' }}"
            "{{ message['role'] + ': ' + message['content'] | trim +'\n' }}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '' }}"
            "{% endif %}"
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if task_type.lower() == 'mcq':
        dataset = create_training_dataset_mcq(MCQ_TRAINING_PATH, use_all_answers, debug)
    elif task_type.lower() == 'saq':
        dataset = create_training_dataset_saq(
            SAQ_TRAINING_PATH, 
            use_all_answers=use_all_answers,
            weight_sampling=weight_sampling,
            seed=seed
        )
    else:
        raise ValueError(f"task_type must be 'mcq' or 'saq', got {task_type}")
    
    def tokenize_function(examples):
        """Safely tokenize messages"""
        input_ids_list = []
        labels_list = []
        
        for messages in examples["messages"]:
            # Tokenize
            tokenized = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                padding=False,
                truncation=False
            )
            
            # Ensure it's a Python list
            if not isinstance(tokenized, list):
                tokenized = tokenized.tolist()
            
            input_ids_list.append(tokenized)
            labels_list.append(tokenized)
        
        return {
            "input_ids": input_ids_list,
            "labels": labels_list
        }

    dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )

    return dataset

    


if __name__ == "__main__":
    print("="*70)
    print("MCQ Dataset")
    print("="*70)
    
    # Create full MCQ training dataset
    mcq_dataset = create_training_dataset_mcq(MCQ_TRAINING_PATH, True, True)
    print(f"Total MCQ examples: {len(mcq_dataset)}")
    print(f"Dataset columns: {mcq_dataset.column_names}")
    print("\nFirst MCQ example:")
    for msg in mcq_dataset[0]['messages']:
        print(f"  {msg['role']}: {msg['content']}")
    

    tokenizer = AutoTokenizer.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.2",
            cache_dir="./Mistral-7B",
            trust_remote_code=True
        )
    data = create_training_data_tokenized(task_type='mcq', tokenizer=tokenizer)
    for _ in data:
        print(tokenizer.decode(_['input_ids']))

    print("\n" + "="*70)
    print("SAQ Dataset - BEST ANSWER ONLY")
    print("="*70)
    
    # Strategy 1: Use only the best answer
    saq_best = create_training_dataset_saq(
        SAQ_TRAINING_PATH,
        use_all_answers=False
    )
    print(f"Total SAQ examples (best only): {len(saq_best)}")
    print("\nFirst SAQ example:")
    for msg in saq_best[0]['messages']:
        print(f"  {msg['role']}: {msg['content']}")
    
    print("\n" + "="*70)
    print("SAQ Dataset - ALL ANSWERS (unweighted)")
    print("="*70)
    
    # Strategy 2: Use all valid answers (each appears once)
    saq_all = create_training_dataset_saq(
        SAQ_TRAINING_PATH,
        use_all_answers=True,
        weight_sampling=False
    )
    print(f"Total SAQ examples (all answers): {len(saq_all)}")
    
    # Show examples for the same question ID
    first_id = saq_all[0]['id']
    same_id_examples = [ex for ex in saq_all if ex['id'] == first_id]
    print(f"\nExamples for question ID '{first_id}': {len(same_id_examples)} different answers")
    for i, ex in enumerate(same_id_examples[:3]):
        answer = ex['messages'][1]['content']
        print(f"  Answer {i+1}: {answer}")
    
    print("\n" + "="*70)
    print("SAQ Dataset - ALL ANSWERS (weighted sampling)")
    print("="*70)
    
    # Strategy 3: Use all valid answers with repetition based on weight
    saq_weighted = create_training_dataset_saq(
        SAQ_TRAINING_PATH,
        use_all_answers=True,
        weight_sampling=True
    )
    print(f"Total SAQ examples (weighted): {len(saq_weighted)}")
    print("Note: Higher-count answers are repeated more in training")
    
    print("\n" + "="*70)
    print("Comparison Summary")
    print("="*70)
    print(f"Best answer only:       {len(saq_best):,} examples")
    print(f"All answers (equal):    {len(saq_all):,} examples")
    print(f"All answers (weighted): {len(saq_weighted):,} examples")
   