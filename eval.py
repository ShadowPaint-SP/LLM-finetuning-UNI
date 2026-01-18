import json
import os
import re
import ast
import datapipe
import torch # type: ignore
import zipfile
import pandas as pd # type: ignore
from tqdm import tqdm  # type: ignore
from datetime import datetime
from collections import defaultdict


MCQ_TRAINING_PATH = "datasets/train_dataset_mcq.csv"
SAQ_TRAINING_PATH = "datasets/train_dataset_saq.csv"
MCQ_TESTING_PATH = "datasets/test_dataset_mcq.csv"
SAQ_TESTING_PATH = "datasets/test_dataset_saq.csv"

def _mcq_func(query: str, tokenizer, model, debug: bool):
    """
    MCQ (Multiple Choice Questions) with improved prompt using chat template
    """

    def _extract_choice_from_text(text: str) -> str:
        """Extract MCQ choice (A-D) from generated text"""
        # Try to parse as JSON first
        try:
            # Look for JSON object in the text
            json_match = re.search(r'\{[^}]*"answer_choice"[^}]*\}', text)
            if json_match:
                json_str = json_match.group(0)
                data = json.loads(json_str)
                choice = data.get("answer_choice", "").strip().upper()
                if choice in ["A", "B", "C", "D"]:
                    return choice
        except:
            pass
        
        # Fallback: Look for isolated A-D after [/INST]
        # Split on [/INST] to ignore the prompt part
        if "[/INST]" in text:
            answer_part = text.split("[/INST]")[-1]
        else:
            answer_part = text
        
        # Look for A, B, C, or D
        letters = re.findall(r'\b[A-D]\b', answer_part.upper())
        if letters:
            return letters[0]
        
        # Last resort: check first character
        answer_part = answer_part.strip().upper()
        if answer_part and answer_part[0] in "ABCD":
            return answer_part[0]
        
        # Default fallback
        return "A"
    # Create messages format
    #TODO system prompt appying or formatting
    
    messages = [
            {"role": "user", "content": query}
    ]
    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    promt_len = prompt.shape[1]
    # Generate answer
    with torch.no_grad():
        outputs = model.generate(
            prompt,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = outputs[0][promt_len:]
    generated = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True
    )
    
    answer = _extract_choice_from_text(generated)
    if debug:
        tqdm.write(f"\rMCQ Prompt: {tokenizer.decode(prompt[0])}\nMCQ generation: {generated}\nMCQ answer: {answer}", end='\n')
    return answer

def _answer_n_mcq(tokenizer, model, n: int, path, debug: bool):
    """
    Answer n multiple choice questions
    n=-1 <-> Answer ALL questions
    """
    mcq = pd.read_csv(path)
    # Extract sample or all
    if n != -1:
        mcq = mcq.sample(n=n) # TODO evtl random_state=12
    # Keep necessary columns
    mcq = mcq[["MCQID", "prompt"]]
    # Get answers
    preds = []
    debug_i = 0
    for q in tqdm(mcq["prompt"], desc="Processing MCQ"):
        if debug_i >= 10:
            debug = False
        debug_i += 1
        answer = _mcq_func(q, tokenizer, model, debug)
        preds.append(answer)

    mcq["answer"] = preds
    mcq_formatted = pd.DataFrame({
        "MCQID": mcq["MCQID"],
        "A": (mcq["answer"] == "A").astype(bool),
        "B": (mcq["answer"] == "B").astype(bool),
        "C": (mcq["answer"] == "C").astype(bool),
        "D": (mcq["answer"] == "D").astype(bool),
    })

    return mcq_formatted

def _evaluate_mcq_predictions(prediction_file):
    """
    Evaluates MCQ predictions against a ground truth dataset.
    
    Args:
        prediction_file (str): Path to the TSV file with predictions (MCQID, A, B, C, D).
        ground_truth_file (str): Path to the CSV file with ground truth (MCQID, answer_column).
    """
    answer_column = "answer_idx"
    try:
        # Load predictions (Tab-Separated Values)
        preds_df = pd.read_csv(prediction_file, sep='\t')
    except Exception as e:
        print(f"Error loading prediction file: {e}")
        return
    try:
        # Load test dataset (Comma-Separated Values)
        gt_df = pd.read_csv(MCQ_TRAINING_PATH)
    except Exception as e:
        print(f"Error loading ground truth file: {e}")
        return

    # Function to convert boolean columns (A, B, C, D) into a single prediction letter
    def get_predicted_choice(row):
        choices = ['A', 'B', 'C', 'D']
        # Find which column is True
        selected = [c for c in choices if row.get(c) == True]
        
        if len(selected) == 1:
            return selected[0]
        elif len(selected) > 1:
            return "Ambiguous" # Multiple choices marked as True
        else:
            return "None"      # No choice marked as True

    # Validate that prediction file has the required columns
    required_cols = ['MCQID', 'A', 'B', 'C', 'D']
    if not all(col in preds_df.columns for col in required_cols):
        print(f"[ERROR] Prediction file is missing one of the required columns: {required_cols}")
        return

    # Extract single predicted label
    preds_df['predicted_answer'] = preds_df.apply(get_predicted_choice, axis=1)

    # Merge predictions with ground truth on MCQID
    merged_df = pd.merge(preds_df[['MCQID', 'predicted_answer']], 
                         gt_df[['MCQID', answer_column]], 
                         on='MCQID', 
                         how='inner')
    
    if merged_df.empty:
        print("[ERROR] No matching MCQIDs found between prediction and ground truth files.")
        return

    # Normalize values for comparison (trim whitespace, uppercase)
    merged_df['predicted_answer'] = merged_df['predicted_answer'].astype(str).str.strip().str.upper()
    merged_df[answer_column] = merged_df[answer_column].astype(str).str.strip().str.upper()

    # Calculate correctness
    merged_df['is_correct'] = merged_df['predicted_answer'] == merged_df[answer_column]

    # Metrics
    total = len(merged_df)
    correct = merged_df['is_correct'].sum()
    accuracy = correct / total if total > 0 else 0

    # Output Results
    print("\n" + "="*40)
    print("MCQ EVALUATION RESULTS")
    print("="*40)
    print(f"Total Questions Evaluated: {total}")
    print(f"Correct Predictions:       {correct}")
    print(f"Accuracy:                  {accuracy:.2%}")
    print("="*40)

    # Save detailed report
    output_filename = "results/evaluation_report_mcq.csv"
    merged_df.to_csv(output_filename, index=False)
    print(f"Detailed report saved to '{output_filename}'")


def _saq_func(query: str, tokenizer, model, debug: bool):
    """
    SAQ (Short Answer Questions) using chat template
    """
    def _extract_answer_from_text(full_text: str) -> str:
        """Robust answer extraction with fallback strategies"""

        answer_text = full_text.split("\n")[0].strip()  # First line only
        answer_text = answer_text.rstrip(".,!?").strip()  # Remove punctuation
        
        return answer_text
    
    # Create messages format
    #TODO system prompt appying or formatting
    query = datapipe.create_saq_prompt(query) # becasue in mcq the instructions are already there but in saq they are missing
    messages = [
        #{"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ]
    
    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    promt_len = prompt.shape[1]

    with torch.no_grad():
        outputs = model.generate(
            prompt,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][promt_len:]
    generated = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True
    )

    answer_text = _extract_answer_from_text(generated)
    if debug:
        tqdm.write(f"\rSAQ Prompt: {tokenizer.decode(prompt[0])}\nSAQ generation: {generated}\nSAQ answer: {answer_text}", end='\n')
    return answer_text

def _answer_n_saq(tokenizer, model, n: int, path, debug: bool):
    """
    Answer n short answer questions
    n=-1 <-> Answer ALL questions
    """

    saq = pd.read_csv(path)

    # Extract sample or all
    if n != -1:
        saq = saq.sample(n=n) #TODO evtl add random selection

    # Keep necessary columns
    saq = saq[["ID", "en_question"]]

    # Get answers
    preds = []
    debug_i = 0
    for q in tqdm(saq["en_question"], desc="Processing SAQ"):
        if debug_i >= 10:
            debug = False
        debug_i += 1
        answer = _saq_func(q, tokenizer, model, debug)
        preds.append(answer)

    saq["answer"] = preds
    return saq[["ID", "answer"]]

def _evaluate_saq_predictions(prediction_file):
    """
    Evaluates SAQ (Short Answer Question) predictions against a ground truth dataset.
    
    Args:
        prediction_file (str): Path to the TSV file with predictions (ID, answer).
    """
    try:
        # Load predictions (Tab-Separated Values)
        preds_df = pd.read_csv(prediction_file, sep='\t')
    except Exception as e:
        print(f"Error loading prediction file: {e}")
        return
            
    try:
        # Load ground truth dataset (Comma-Separated Values)
        gt_df = pd.read_csv(SAQ_TRAINING_PATH)
    except Exception as e:
        print(f"Error loading ground truth file: {e}")
        return

    results_by_id = defaultdict(list)

    for row in gt_df.itertuples():
        question_id = row.ID
        annotations = ast.literal_eval(row.annotations)
        idks = ast.literal_eval(row.idks)
        merged_answers = {}
        
        for item in annotations:
            if item.get('en_answers'):
                answer_key = item['en_answers'][0]
                merged_answers[answer_key] = item['count']
                
        if isinstance(idks, dict):
            merged_answers.update(idks)

        results_by_id[question_id].append(merged_answers)

    best_possible_score = 0
    pred_achieved_score = 0
    result_data = []

    for row in preds_df.itertuples():
        current_id = row.ID
        candidate_answer = row.answer
        
        valid_answers = {}
        score = 0
        max_score = 0

        if current_id in results_by_id:
            possible_answer_sets = results_by_id[current_id]
            
            best_set = None
            best_achieved = -1
            best_set_max = 0

            for ans_set in possible_answer_sets:
                current_achieved = ans_set.get(candidate_answer, 0)
                current_max = max(ans_set.values()) if ans_set else 0
                #if current_achieved > 0:
                #    print(f"[DEBUG] ID '{current_id}' candidate answer '{candidate_answer}' matched with score {current_achieved} in set {ans_set}")
                if current_achieved > best_achieved:
                    best_achieved = current_achieved
                    best_set_max = current_max
                    best_set = ans_set
                
                # Tie-breaker: If we get 0 score on all (or equal score), 
                # pick the one with the higher 'max_score' (stricter denominator)
                elif current_achieved == best_achieved:
                    if current_max > best_set_max:
                        best_set_max = current_max
                        best_set = ans_set
            
            valid_answers = best_set if best_set is not None else {}
            score = best_achieved if best_achieved > 0 else 0
            max_score = best_set_max
            
            pred_achieved_score += score
            best_possible_score += max_score

        result_data.append({
            'ID': current_id,
            'predicted_answer': candidate_answer,
            'valid_answers': valid_answers,
            'achieved_score': score,
            'max_score': max_score,
            'is_correct': score == max_score and max_score > 0,
            'semi_correct': score > 0
        })

    results_df = pd.DataFrame(result_data)

    accuracy = (pred_achieved_score / best_possible_score) if best_possible_score > 0 else 0

    print("\n" + "="*40)
    print("SAQ EVALUATION RESULTS")
    print("="*40)
    print(f"Total Questions Evaluated:  {len(results_df)}")
    print(f"Exact Matches (Full Score): {results_df['is_correct'].sum()}")
    print(f"Total Achieved Score:       {pred_achieved_score}")
    print(f"Total Max Score:            {best_possible_score}")
    print(f"Overall Accuracy:           {accuracy:.2%}")
    print("="*40)

    output_filename = "results/evaluation_report_saq.csv"
    results_df.to_csv(output_filename, index=False)
    print(f"Detailed report saved to '{output_filename}'")

def create_zip_for_submission(saq, mcq, additional_files=None):
    """
    Create submission files in correct format and include optional extra files.
    
    Args:
        saq: DataFrame for SAQ predictions
        mcq: DataFrame for MCQ predictions
        additional_files (list): List of paths to other files to include in the zip
    """
    saq.to_csv("saq_prediction.tsv", sep='\t', index=False)
    mcq.to_csv("mcq_prediction.tsv", sep='\t', index=False)
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    zip_name = f"submission_{timestamp}.zip"
    output_dir = "results/submissions"
    
    os.makedirs(output_dir, exist_ok=True)
    zip_path = os.path.join(output_dir, zip_name)

    print(f"Creating {zip_name}...")

    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write("saq_prediction.tsv")
        z.write("mcq_prediction.tsv")
        
        if additional_files:
            for file_path in additional_files:
                if os.path.exists(file_path):
                    # arcname=basename ensures files are flattened into the zip root
                    # e.g., 'sub/dir/image.png' becomes just 'image.png' inside the zip
                    z.write(file_path, arcname=os.path.basename(file_path))
                    print(f"Added: {file_path}")
                else:
                    print(f"Warning: File not found - {file_path}")
    
    print(f"Submission successfully created: {zip_path}")
    
    # 4. Cleanup temporary files
    os.remove("saq_prediction.tsv")
    os.remove("mcq_prediction.tsv")



def start_inference_process_training(tokenizer, model, n_samples: int, task:int = -1, debug: bool = False):
    """Start evaluation for training data
    task: -1 (both), 0 (SAQ), 1 (MCQ)
    """
    print("Generating predictions on training data...\n")
    os.makedirs("results", exist_ok=True)
    if task == -1 or task == 0:
        saq = _answer_n_saq(tokenizer, model, n_samples, path=SAQ_TRAINING_PATH, debug=debug)
        saq.to_csv("results/saq_train.tsv", sep='\t', index=False)
    if task == -1 or task == 1:
        mcq = _answer_n_mcq(tokenizer, model, n_samples, path=MCQ_TRAINING_PATH, debug=debug)
        mcq.to_csv("results/mcq_train.tsv", sep='\t', index=False)

    print("Training predictions saved to submissions/")

def start_inference_process_testing(tokenizer, model, task:int = -1, debug: bool = False):
    """Start evaluation for testing data
    task: -1 (both), 0 (SAQ), 1 (MCQ)
    """
    os.makedirs("results/submissions", exist_ok=True)
    if task == -1 or task == 0:
        print(f"Generating predictions on testing data for SAQ...\n")
        answers = _answer_n_saq(tokenizer, model, -1, path=SAQ_TESTING_PATH, debug=debug)
    if task == -1 or task == 1:
        print(f"Generating predictions on testing data for MCQ...\n")
        answers = _answer_n_mcq(tokenizer, model, -1, path=MCQ_TESTING_PATH, debug=debug)

    return answers

def evaluate_results():
    """Evaluate results for both SAQ and MCQ"""
    _evaluate_saq_predictions("results/saq_train.tsv")
    _evaluate_mcq_predictions("results/mcq_train.tsv")


if __name__ == "__main__":
    evaluate_results()