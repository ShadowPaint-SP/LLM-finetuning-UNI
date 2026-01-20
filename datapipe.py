import ast
import json
from pathlib import Path
import random
import pandas as pd 
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

# Import RAG module
from rag import WikivoyageRAG, augment_training_data_with_rag, setup_rag_system

MCQ_TRAINING_PATH = "datasets/train_dataset_mcq.csv"
SAQ_TRAINING_PATH = "datasets/train_dataset_saq.csv"
BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "Model-Cache"

# --- HELPER FUNCTIONS ---

def generate_shuffled_variations(options, correct_key):
    """Generates 4 variations of the options dict."""
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

def create_saq_prompt(question: str) -> str:
    return f"{question} Provide ONLY the exact answer without explanation."

def augment_question(question: str, task_type: str, variation_seed: int) -> str:
    """Create paraphrased variations."""
    random.seed(variation_seed)
    if task_type == 'mcq':
        variations = [
            f"Which of the following best answers: {question}?",
            f"Regarding the following question: {question}",
            f"What is the correct response to: {question}?",
            f"Select the best answer for: {question}",
            f"Choose the most appropriate option: {question}",
        ]
    else:
        variations = [
            f"Answer in detail: {question}",
            f"Please explain: {question}",
            f"Provide an answer to: {question}",
            f"Explain the following: {question}",
            f"Respond to this question: {question}",
        ]
    return random.choice(variations)

# --- CORE DATA PROCESSING FUNCTIONS ---

def process_dataframe_mcq(df, use_all_answers=False, seed=42, is_validation=False):
    """
    Process a DataFrame of MCQ questions into training examples.
    is_validation: If True, disables data expansion (use_all_answers) to create a stable benchmark.
    """
    training_examples = []
    
    # Disable augmentation for validation to prevent leakage/noise
    effective_use_all = False if is_validation else use_all_answers

    for row in df.itertuples():
        prompt = row.prompt.strip()
        correct_answer = row.answer_idx.strip()
        choices = json.loads(row.choices)
        country = json.loads(row.choice_countries)

        # 1. Standard Q&A Pairs
        if effective_use_all:
            # CLEANING: Remove existing JSON markers if present
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
                        {"role": "user", "content": "Why is this correct"},
                        {"role": "assistant", "content": f"Because '{answer[letter]}' is the correct answer"}
                    ],
                    'mcqid': row.MCQID
                })
        else:
            # Standard single example (used for Val set or basic training)
            completion = json.dumps({"answer_choice": correct_answer})
            training_examples.append({
                'messages': [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": completion}
                ],
                'mcqid': row.MCQID
            })

        # 2. Auxiliary Task (Reasoning/Country ID)
        # Only add this for training to avoid polluting validation metrics with non-target tasks
        if not is_validation:
            question_only = prompt.split('?')[0] + '?'
            training_examples.append({
                'messages': [
                    {"role": "user", "content": f"Analyze the following question and identify the geographical regions associated with the options: '{question_only} \n {json.dumps(choices)}'"},
                    {"role": "assistant", "content": json.dumps(country)},
                ],
                'mcqid': row.MCQID
            })

    dataset = Dataset.from_list(training_examples)
    if not is_validation:
        dataset = dataset.shuffle(seed)
    return dataset

def process_dataframe_saq(df, use_all_answers=False, seed=42, is_validation=False):
    """
    Process a DataFrame of SAQ questions.
    """
    training_examples = []
    
    # Disable augmentation for validation
    effective_use_all = False if is_validation else use_all_answers
    
    for row in df.itertuples():
        en_question = row.en_question.strip()
        annotations = ast.literal_eval(row.annotations)
        idks = ast.literal_eval(row.idks)
        country = row.country
        merged_answers = {}
        
        for item in annotations:
            if item.get('en_answers'):
                answer_key = item['en_answers'][0]
                merged_answers[answer_key] = item['count']

        if isinstance(idks, dict):
            merged_answers.update(idks)
        
        prompt = create_saq_prompt(en_question)
        
        if effective_use_all:
            valid_answers = [k for k, v in merged_answers.items() if v > 0]
            for answer in valid_answers:
                training_examples.append({
                    'messages': [
                        {"role": "user", "content": f"{prompt} Target Score: {merged_answers[answer]}"},
                        {"role": "assistant", "content": answer},
                        {"role": "user", "content": "Why is this correct"},
                        {"role": "assistant", "content": f"Because it is a cultural question about {country}"}
                    ],
                    'id': row.ID
                })
        else:
            # Validation / Standard: Pick the single most frequent answer
            best_answer = max(merged_answers.items(), key=lambda x: x[1])[0]
            training_examples.append({
                'messages': [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": best_answer},
                    {"role": "user", "content": "Why is this correct"},
                    {"role": "assistant", "content": f"Because it is a cultural question about {country}"}
                ],
                'id': row.ID
            })
        
        # Auxiliary Task (only for training)
        if not is_validation:
            best_answer = max(merged_answers.items(), key=lambda x: x[1])[0]
            if best_answer not in ['idk', 'not-applicable', 'no-answer']:
                training_examples.append({
                    'messages': [
                        {"role": "user", "content": f"Identify the country of origin for this entity: '{best_answer}'"},
                        {"role": "assistant", "content": country}
                    ],
                    'id': row.ID
                })
    
    dataset = Dataset.from_list(training_examples)
    if not is_validation:
        dataset = dataset.shuffle(seed)
    return dataset


# --- AUGMENTATION WRAPPER ---

def augment_dataset_before_tokenization(dataset: Dataset, task_type: str, 
                                        augmentation_factor: float = 1.5,
                                        seed: int = 42) -> Dataset:
    """Paraphrase augmentation."""
    if augmentation_factor <= 1.0:
        return dataset
    
    random.seed(seed)
    original_size = len(dataset)
    num_augmented = int(original_size * (augmentation_factor - 1.0))
    
    augmented_examples = list(dataset)
    
    for i in range(num_augmented):
        idx = random.randint(0, original_size - 1)
        original = dataset[idx]
        augmented = {k: v for k, v in original.items()}
        
        messages = original['messages']
        user_message = messages[0]['content']
        
        # Logic to find where the question text is (especially for MCQs with options appended)
        if task_type == 'mcq':
            if '\n\nA.' in user_message:
                parts = user_message.split('\n\nA.')
                base_question = parts[0]
                options_part = '\n\nA.' + parts[1]
            else:
                base_question = user_message
                options_part = ''
            
            paraphrased = augment_question(base_question, task_type, seed + i)
            new_user_content = paraphrased + options_part
        else:
            new_user_content = augment_question(user_message, task_type, seed + i)
        
        new_messages = [{"role": "user", "content": new_user_content}]
        new_messages.extend(messages[1:])
        
        augmented['messages'] = new_messages
        augmented_examples.append(augmented)
    
    return Dataset.from_list(augmented_examples)


# --- MAIN PIPELINE FUNCTION ---

def create_training_data_tokenized(
    task_type: str, 
    tokenizer, 
    seed: int = 42,
    use_all_answers: bool = False, 
    augment_data: bool = False, 
    augmentation_factor: float = 1.5,
    use_rag: bool = False,
    rag_k: int = 3,
    rag_method: str = "hybrid",
    rag_cache_dir: str = "rag_cache",
    debug: bool = False,
    val_set_size: float = 0.1  # NEW ARGUMENT
):
    """
    Creates LEAKAGE-FREE Train/Test splits by splitting IDs first.
    """
    # 1. Load Data Frame & ID Column
    if task_type.lower() == 'mcq':
        df = pd.read_csv(MCQ_TRAINING_PATH)
        id_col = 'MCQID'
        process_func = process_dataframe_mcq
    elif task_type.lower() == 'saq':
        df = pd.read_csv(SAQ_TRAINING_PATH)
        id_col = 'ID'
        process_func = process_dataframe_saq
    else:
        raise ValueError("task_type must be 'mcq' or 'saq'")

    # 2. Split unique IDs (Prevent Leakage)
    unique_ids = df[id_col].unique()
    if val_set_size and val_set_size > 0:
        train_ids, val_ids = train_test_split(unique_ids, test_size=val_set_size, random_state=seed)
    else:
        train_ids = unique_ids
        val_ids = []

    train_df = df[df[id_col].isin(train_ids)]
    val_df = df[df[id_col].isin(val_ids)]

    print(f"[{task_type.upper()}] Splitting Data by ID: {len(train_ids)} Train IDs, {len(val_ids)} Val IDs")

    # 3. Process Dataframes
    # Train gets all augmentations (use_all_answers, etc.)
    train_dataset = process_func(train_df, use_all_answers=use_all_answers, seed=seed, is_validation=False)
    
    # Val gets NO augmentation (stable benchmark)
    if len(val_df) > 0:
        val_dataset = process_func(val_df, use_all_answers=False, seed=seed, is_validation=True)
    else:
        val_dataset = None

    # 4. Paraphrase Augmentation (Train Only)
    if augment_data:
        original_len = len(train_dataset)
        train_dataset = augment_dataset_before_tokenization(train_dataset, task_type, augmentation_factor, seed)
        print(f"[AUGMENTATION] Train set: {original_len} -> {len(train_dataset)}")

    # 5. RAG Augmentation
    if use_rag:
        print(f"\n[RAG] Initializing RAG system...")
        rag = WikivoyageRAG(
            wikivoyage_xml_path="datasets/wikivoyage.xml",
            cache_dir=rag_cache_dir,
            use_dense=(rag_method in ["dense", "hybrid"]),
            use_sparse=(rag_method in ["sparse", "hybrid"]),
            device=None 
        )
        rag.initialize(force_rebuild=False)

        # Apply to Train (with Randomization)
        print("[RAG] Augmenting Train Set (Randomized Context)...")
        train_list = list(train_dataset)
        train_aug = augment_training_data_with_rag(
            train_list, rag, task_type, k=rag_k, add_context_to_user=True, 
            batch_retrieval=True
        )
        train_dataset = Dataset.from_list(train_aug)

        # Apply to Val (Deterministic - No Randomization)
        if val_dataset:
            print("[RAG] Augmenting Val Set (Deterministic Context)...")
            val_list = list(val_dataset)
            val_aug = augment_training_data_with_rag(
                val_list, rag, task_type, k=rag_k, add_context_to_user=True, 
                batch_retrieval=True
            )
            val_dataset = Dataset.from_list(val_aug)

    # 6. Tokenization
    def tokenize_and_mask(examples):
        input_ids_list = []
        labels_list = []
        for messages in examples["messages"]:
            input_ids = tokenizer.apply_chat_template(
                messages, truncation=True, max_length=2048,
                add_generation_prompt=False, padding=False,
            )
            input_ids_list.append(input_ids)
            labels_list.append(input_ids) # Simple causal masking
        return {"input_ids": input_ids_list, "labels": labels_list}

    print(f"Tokenizing {task_type.upper()}...")
    train_dataset = train_dataset.map(tokenize_and_mask, batched=True, remove_columns=train_dataset.column_names)
    if val_dataset:
        val_dataset = val_dataset.map(tokenize_and_mask, batched=True, remove_columns=val_dataset.column_names)

    # Return DatasetDict
    if val_dataset:
        return DatasetDict({"train": train_dataset, "test": val_dataset})
    else:
        return DatasetDict({"train": train_dataset})