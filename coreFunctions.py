import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def load_tokenizer_and_model(device:str, model_name:str, model_dir:str,
                             cache_dir:str):
    """
    Standard Function that returns both the tokenizer and model.
    """
    
    # Load and save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        cache_dir=cache_dir
        )
    print(f"Tokenizer loaded from {model_name}")
    
    os.makedirs(model_dir, exist_ok=True)
    tokenizer.save_pretrained(model_dir)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
    ).to(device)
    print(f"Model loaded from {model_name}")
    
    return tokenizer, model

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