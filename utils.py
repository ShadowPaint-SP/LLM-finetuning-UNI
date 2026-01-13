import argparse
import os
import torch # type: ignore
from transformers import ( # type: ignore
    AutoTokenizer,
    AutoModelForCausalLM,
)


# I am working on that so we can use arguments when starting some scripts from terminal
# could be useful for not always having to change the code to test different parameters

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "LoRA fine-tune Mistral-7B/LLama3.0-8B "
        )
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.2",
        #default="meta-llama/Meta-Llama-3-8B",
        help="Base model to fine-tune.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Cap training samples to keep runtime under 1 hour. Set None to use full dataset.",
    )
    parser.add_argument(
        "--gen_test_preds",
        type=bool,
        default=False,
        help="Whether to generate predictions on the test set. And make a submission .zip file.",
		)
    parser.add_argument(
        "--gen_train_preds",
        type=bool,
        default=True,
        help="Whether to generate predictions on the train set.",
		)
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Number of training epochs (use fractional values to cap runtime).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,
        help="Per-GPU batch size. Lower if you hit OOM; increase if you have headroom.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="LoRA learning rate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Reproducibility seed.",
    )
    parser.add_argument(
        "--eval_train_samples",
        type=int,
        default=100,
        help="Number of training samples to use for evaluation during training. -1 for all.",
		)
    parser.add_argument(
        "--debug",
        type=bool,
				default=True,
				help="Whether to run in debug mode with more prints.",
		)
    return parser.parse_args()


#functions taken from coreFunctions.py

def ask_model(prompt:str, tokenizer, model):
    """
    Main Interaction Point with the LLM
    
    :param prompt: Full Question asked to the model
    :type prompt: str
    :param tokenizer: Tokenizer Object
    :param model: Model Object

    :return(str): Everything the model added after the input (the answer).
    """
    
    # tokenize the input into pytorch tensors
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # generate answer
    with torch.no_grad():                        # deactivate backpropagation
        outputs = model.generate(                # generate answer
            **inputs,
            max_new_tokens=20,                   # answer length <= 100 tokens
            do_sample=False,                     # ensure deterministic answer 
            pad_token_id=tokenizer.eos_token_id, # use end-of-sequence token
        )

    # deencode answer (tokens -> text)
    # output[0] contains array(input_tokens)+array(output tokens)
    # only extract the output tokens
    generated = tokenizer.decode(                  
        outputs[0][inputs["input_ids"].shape[-1]:], 
        skip_special_tokens=True
    )
    
    return generated