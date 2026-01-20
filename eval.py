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
import rag # Import the RAG module

MCQ_TRAINING_PATH = "datasets/train_dataset_mcq.csv"
SAQ_TRAINING_PATH = "datasets/train_dataset_saq.csv"
MCQ_TESTING_PATH = "datasets/test_dataset_mcq.csv"
SAQ_TESTING_PATH = "datasets/test_dataset_saq.csv"

def _format_rag_query(query: str, retrieved_docs: list, max_context_length: int = 400) -> str:
    """Helper to format query with RAG context"""
    context_parts = []
    for i, doc in enumerate(retrieved_docs, 1):
        passage = doc['passage'][:max_context_length]
        if len(doc['passage']) > max_context_length:
            passage += "..."
        context_parts.append(f"[Doc {i}] {doc['title']}: {passage}")
    
    context_str = "\n\n".join(context_parts)
    
    return (
        f"Context Information:\n{context_str}\n\n"
        f"Question: {query}"
    )

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
        
        # Fallback: Look for isolated A-D after [/INST] or header
        # Split on [/INST] to ignore the prompt part (if present)
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

def _answer_n_mcq(tokenizer, model, n: int, path, debug: bool, rag_system=None):
    """
    Answer n multiple choice questions
    n=-1 <-> Answer ALL questions
    """
    mcq = pd.read_csv(path)
    # Extract sample or all
    if n != -1:
        # Safety check: don't sample more than available
        n = min(n, len(mcq))
        mcq = mcq.sample(n=n, random_state=42)
        
    # Keep necessary columns
    mcq = mcq[["MCQID", "prompt"]]
    
    # Get answers
    preds = []
    debug_i = 0
    
    for q in tqdm(mcq["prompt"], desc="Processing MCQ"):
        if debug_i >= 10:
            debug = False
        debug_i += 1
        
        query_text = q
        
        # --- RAG PROCESSING ---
        if rag_system:
            try:
                # Extract clean question for search (remove options)
                # Heuristic: Split at first '?'
                search_query = q.split('?')[0] + '?'
                
                # Retrieve
                retrieved = rag_system.retrieve(
                    search_query, 
                    k=3, 
                    use_mmr=True, 
                    mmr_diversity=0.3
                )
                
                if retrieved:
                    # Enhance query with context
                    query_text = _format_rag_query(q, retrieved)
            except Exception as e:
                print(f"RAG Error on MCQ: {e}")
        # ----------------------
        
        answer = _mcq_func(query_text, tokenizer, model, debug)
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
    """Evaluates MCQ predictions against a ground truth dataset."""
    answer_column = "answer_idx"
    try:
        preds_df = pd.read_csv(prediction_file, sep='\t')
    except Exception as e:
        print(f"Error loading prediction file: {e}")
        return
    try:
        gt_df = pd.read_csv(MCQ_TRAINING_PATH)
    except Exception as e:
        print(f"Error loading ground truth file: {e}")
        return

    def get_predicted_choice(row):
        choices = ['A', 'B', 'C', 'D']
        selected = [c for c in choices if row.get(c) == True]
        if len(selected) == 1:
            return selected[0]
        elif len(selected) > 1:
            return "Ambiguous"
        else:
            return "None"

    required_cols = ['MCQID', 'A', 'B', 'C', 'D']
    if not all(col in preds_df.columns for col in required_cols):
        print(f"[ERROR] Prediction file is missing columns: {required_cols}")
        return

    preds_df['predicted_answer'] = preds_df.apply(get_predicted_choice, axis=1)
    merged_df = pd.merge(preds_df[['MCQID', 'predicted_answer']], 
                         gt_df[['MCQID', answer_column]], 
                         on='MCQID', how='inner')
    
    if merged_df.empty:
        print("[ERROR] No matching MCQIDs found.")
        return

    merged_df['predicted_answer'] = merged_df['predicted_answer'].astype(str).str.strip().str.upper()
    merged_df[answer_column] = merged_df[answer_column].astype(str).str.strip().str.upper()
    merged_df['is_correct'] = merged_df['predicted_answer'] == merged_df[answer_column]

    total = len(merged_df)
    correct = merged_df['is_correct'].sum()
    accuracy = correct / total if total > 0 else 0

    print("\n" + "="*40)
    print("MCQ EVALUATION RESULTS")
    print("="*40)
    print(f"Total:    {total}")
    print(f"Correct:  {correct}")
    print(f"Accuracy: {accuracy:.2%}")
    print("="*40)

    output_filename = "results/evaluation_report_mcq.csv"
    merged_df.to_csv(output_filename, index=False)
    print(f"Detailed report saved to '{output_filename}'")


def _saq_func(query: str, tokenizer, model, debug: bool):
    """
    SAQ (Short Answer Questions) using chat template
    """
    def _extract_answer_from_text(full_text: str) -> str:
        """Robust answer extraction"""
        answer_text = full_text.split("\n")[0].strip()
        answer_text = answer_text.rstrip(".,!?").strip()
        return answer_text
    
    # Add instruction (Provide ONLY...)
    # Note: query might already contain RAG context at this point
    final_query = datapipe.create_saq_prompt(query) 
    
    messages = [
        {"role": "user", "content": final_query}
    ]
    
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

def _answer_n_saq(tokenizer, model, n: int, path, debug: bool, rag_system=None):
    """
    Answer n short answer questions
    n=-1 <-> Answer ALL questions
    """
    saq = pd.read_csv(path)

    if n != -1:
        n = min(n, len(saq))
        saq = saq.sample(n=n, random_state=42)

    saq = saq[["ID", "en_question"]]

    preds = []
    debug_i = 0
    for q in tqdm(saq["en_question"], desc="Processing SAQ"):
        if debug_i >= 10:
            debug = False
        debug_i += 1
        
        query_text = q
        
        # --- RAG PROCESSING ---
        if rag_system:
            try:
                # Retrieve using raw question
                retrieved = rag_system.retrieve(
                    q, 
                    k=3, 
                    use_mmr=True, 
                    mmr_diversity=0.3
                )
                
                if retrieved:
                    # Enhance query with context
                    # Note: We do this BEFORE create_saq_prompt adds the instruction
                    query_text = _format_rag_query(q, retrieved)
            except Exception as e:
                print(f"RAG Error on SAQ: {e}")
        # ----------------------
        
        answer = _saq_func(query_text, tokenizer, model, debug)
        preds.append(answer)

    saq["answer"] = preds
    return saq[["ID", "answer"]]

def _evaluate_saq_predictions(prediction_file):
    """Evaluates SAQ predictions."""
    try:
        preds_df = pd.read_csv(prediction_file, sep='\t')
        gt_df = pd.read_csv(SAQ_TRAINING_PATH)
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    results_by_id = defaultdict(list)
    for row in gt_df.itertuples():
        annotations = ast.literal_eval(row.annotations)
        idks = ast.literal_eval(row.idks)
        merged = {}
        for item in annotations:
            if item.get('en_answers'):
                merged[item['en_answers'][0]] = item['count']
        if isinstance(idks, dict):
            merged.update(idks)
        results_by_id[row.ID].append(merged)

    best_possible_score = 0
    pred_achieved_score = 0
    result_data = []

    for row in preds_df.itertuples():
        candidate = row.answer
        
        valid = {}
        score = 0
        max_s = 0

        if row.ID in results_by_id:
            possible_sets = results_by_id[row.ID]
            best_achieved = -1
            best_set_max = 0
            best_set = None

            for ans_set in possible_sets:
                curr_achieved = ans_set.get(candidate, 0)
                curr_max = max(ans_set.values()) if ans_set else 0
                
                if curr_achieved > best_achieved:
                    best_achieved = curr_achieved
                    best_set_max = curr_max
                    best_set = ans_set
                elif curr_achieved == best_achieved:
                    if curr_max > best_set_max:
                        best_set_max = curr_max
                        best_set = ans_set
            
            valid = best_set if best_set else {}
            score = best_achieved if best_achieved > 0 else 0
            max_s = best_set_max
            
            pred_achieved_score += score
            best_possible_score += max_s

        result_data.append({
            'ID': row.ID,
            'predicted': candidate,
            'score': score,
            'max_score': max_s
        })

    accuracy = (pred_achieved_score / best_possible_score) if best_possible_score > 0 else 0

    print("\n" + "="*40)
    print("SAQ EVALUATION RESULTS")
    print("="*40)
    print(f"Total Score: {pred_achieved_score} / {best_possible_score}")
    print(f"Accuracy:    {accuracy:.2%}")
    print("="*40)
    
    pd.DataFrame(result_data).to_csv("results/evaluation_report_saq.csv", index=False)

def create_zip_for_submission(saq, mcq, additional_files=None):
    """Create submission zip"""
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
            for f in additional_files:
                if os.path.exists(f):
                    z.write(f, arcname=os.path.basename(f))
    
    print(f"Submission created: {zip_path}")
    os.remove("saq_prediction.tsv")
    os.remove("mcq_prediction.tsv")

def start_inference_process_training(tokenizer, model, n_samples: int, task:int = -1, debug: bool = False, use_rag=False):
    """Start evaluation for training data with optional RAG"""
    print("Generating predictions on training data...\n")
    os.makedirs("results", exist_ok=True)
    rag_system = None
    if use_rag:
        print("\n[Eval] Initializing RAG system for inference...")
        rag_system = rag.setup_rag_system(
            rag_method = "hybrid"
        )
    if task == -1 or task == 0:
        saq = _answer_n_saq(tokenizer, model, n_samples, path=SAQ_TRAINING_PATH, debug=debug, rag_system=rag_system)
        saq.to_csv("results/saq_train.tsv", sep='\t', index=False)
    if task == -1 or task == 1:
        mcq = _answer_n_mcq(tokenizer, model, n_samples, path=MCQ_TRAINING_PATH, debug=debug, rag_system=rag_system)
        mcq.to_csv("results/mcq_train.tsv", sep='\t', index=False)

    print("Training predictions saved to results/")

def start_inference_process_testing(tokenizer, model, task:int = -1, debug: bool = False, use_rag=False):
    """Start evaluation for testing data with optional RAG"""
    os.makedirs("results/submissions", exist_ok=True)
    answers_saq = None
    answers_mcq = None
    rag_system = None
    if use_rag:
        print("\n[Eval] Initializing RAG system for inference...")
        rag_system = rag.setup_rag_system(
            rag_method = "hybrid"
        )
    if task == -1 or task == 0:
        print(f"Generating predictions on testing data for SAQ...\n")
        answers_saq = _answer_n_saq(tokenizer, model, -1, path=SAQ_TESTING_PATH, debug=debug, rag_system=rag_system)
    
    if task == -1 or task == 1:
        print(f"Generating predictions on testing data for MCQ...\n")
        answers_mcq = _answer_n_mcq(tokenizer, model, -1, path=MCQ_TESTING_PATH, debug=debug, rag_system=rag_system)

    return answers_saq if task == 0 else answers_mcq if task == 1 else (answers_saq, answers_mcq)

def evaluate_results():
    """Evaluate results for both SAQ and MCQ"""
    _evaluate_saq_predictions("results/saq_train.tsv")
    _evaluate_mcq_predictions("results/mcq_train.tsv")

if __name__ == "__main__":
    evaluate_results()