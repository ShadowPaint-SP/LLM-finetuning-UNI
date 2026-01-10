"""
Optimized Mistral-7B finetuning for cultural Q&A with LoRA
Improvements: LoRA finetuning, better prompts, multi-word answers, country context
"""

import os
import datapipe
import torch  # type: ignore
from eval import *
import pandas as pd  # type: ignore
from utils import parse_args
from datasets import Dataset
from peft import LoraConfig, get_peft_model, PeftModel  # type: ignore
from transformers import ( # type: ignore
    Trainer,
    AutoTokenizer, 
    TrainingArguments,
    AutoModelForCausalLM, 
    DataCollatorForLanguageModeling
)

# Fix torch.utils.checkpoint warning by setting use_reentrant=False
torch.utils.checkpoint.use_reentrant = False


def load_tokenizer_and_model(device, model_name, model_dir, cache_dir):
    """Load model and tokenizer with optional quantization for memory efficiency"""

    # Quantization config for memory efficiency
    #bnb_config = BitsAndBytesConfig(
    #    load_in_4bit=True,
    #    bnb_4bit_use_double_quant=True,
    #    bnb_4bit_quant_type="nf4",
    #    bnb_4bit_compute_dtype=torch.bfloat16
    #)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir
    )
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Tokenizer loaded from {model_name}")

    os.makedirs(model_dir, exist_ok=True)
    tokenizer.save_pretrained(model_dir)

    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        #quantization_config=bnb_config,
        cache_dir=cache_dir
    ).to(device)
    print(f"Model loaded from {model_name}")
    return tokenizer, model

def load_finetuned_model(tokenizer, adapter_path):
    """Load base model with LoRA adapters from finetuned checkpoint
    
    If adapter_path is a directory, will find the latest checkpoint.
    If adapter_path is a specific checkpoint, will use that.
    """

    # Convert relative path to absolute
    adapter_path = os.path.abspath(adapter_path)
    
    # If path doesn't exist and looks like a checkpoint dir, find the latest checkpoint
    if not os.path.exists(adapter_path):
        base_dir = os.path.dirname(adapter_path)
        if os.path.isdir(base_dir):
            # Find all checkpoint directories
            checkpoints = [d for d in os.listdir(base_dir) if d.startswith("checkpoint-")]
            if checkpoints:
                # Sort by checkpoint number and get the latest
                checkpoints.sort(key=lambda x: int(x.split("-")[1]))
                latest_checkpoint = checkpoints[-1]
                adapter_path = os.path.join(base_dir, latest_checkpoint)
                print(f"Checkpoint not found at specified path. Using latest: {latest_checkpoint}")
            else:
                raise ValueError(f"No checkpoints found in directory: {base_dir}")
        else:
            raise ValueError(f"Adapter path does not exist and base directory not found: {adapter_path}")
    
    if not os.path.exists(adapter_path):
        raise ValueError(f"Adapter path does not exist: {adapter_path}")

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2",
        torch_dtype=torch.bfloat16,
        #quantization_config=bnb_config,
        cache_dir="./Mistral-7B"
    ).to("cuda")
    print(f"Base model loaded")

    # Load and merge LoRA adapters from local path
    model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    print(f"LoRA adapters loaded from {adapter_path}")

    return model

def setup_lora(model):
    """Setup LoRA configuration for efficient finetuning"""
    
    # Disable use_cache when using gradient checkpointing
    model.config.use_cache = False
    
    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    
    # For non-quantized models, manually set requires_grad for all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # More comprehensive target modules for better learning
    lora_config = LoraConfig(
        r=32,  # Increased from 16 for more expressiveness
        lora_alpha=64,  # Increased from 32 for stronger adaptation
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Added k_proj and o_proj
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    
    # Set model to training mode AFTER applying LoRA
    model.train()
    
    # Print statistics
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")
    print(f"Trainable ratio: {trainable_params / total_params * 100:.2f}%")
    
    return model

def finetune_model(model, tokenizer, train_dataset, output_dir="./mistral_finetuned"):
    """Finetune model with LoRA and optimized hyperparameters"""

    # Ensure model is in training mode
    model.train()
    
    # Verify that gradients are enabled
    has_trainable = any(p.requires_grad for p in model.parameters())
    if not has_trainable:
        print("[WARNING] No trainable parameters found! LoRA may not be properly configured.")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,  # Increased from 3 to better learn the data
        per_device_train_batch_size=2,  # Reduced to allow better gradient updates
        gradient_accumulation_steps=8,  # Effective batch size = 2 * 8 = 16
        warmup_steps=200,  # Better warmup for learning rate schedule
        weight_decay=0.01,
        logging_steps=20,  # More frequent logging
        save_steps=200,  # Save more checkpoints
        save_total_limit=3,  # Keep more checkpoints
        learning_rate=1e-4,  # Lower learning rate for more stable finetuning
        bf16=True,  # Use bfloat16 precision
        logging_dir="./logs",
        eval_strategy="no",  # We'll do custom evaluation if needed
        save_strategy="steps",
        gradient_checkpointing=False,  # Already enabled on model itself
        max_grad_norm=1.0,  # Gradient clipping to prevent exploding gradients
        dataloader_pin_memory=True,  # Pin memory for faster data loading
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    
    trainer.train()
    print(f"Model finetuned and saved to {output_dir}")
    return model


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    # Configuration
    finetune = True  # Set to True to finetune on training data
    load_from_checkpoint = args.load_from_checkpoint  # Set to True to load from finetuned checkpoint (skip finetuning)
    n_samples = -1    # -1 for all, or specify a number
    generate_test_predictions = args.gen_test_preds     # Create submission files
    generate_train_predictions = args.gen_train_preds  # Generate predictions for training data

    print("\nInitializing...\n")
    
    # Load model - either from checkpoint or fresh
    if load_from_checkpoint:
        print("Loading finetuned model from checkpoint...\n")
        tokenizer = AutoTokenizer.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.2",
            cache_dir="./Mistral-7B"
        )
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Tokenizer loaded")
        model = load_finetuned_model(tokenizer, "./mistral_finetuned_saq/checkpoint-234")
        print("Model and adapters loaded from checkpoint\n")
    else:
        # Load model and tokenizer
        tokenizer, model = load_tokenizer_and_model(
            device="cuda",
            model_name="mistralai/Mistral-7B-Instruct-v0.2",
            model_dir="./cache",
            cache_dir="./Mistral-7B"
        )
        print("\nModel and Tokenizer loaded\n")

    # Optional: Finetune on training data
    if finetune:
        print("Preparing training data for finetuning...\n")

        # Setup LoRA
        model = setup_lora(model)

        # Prepare MCQ training data
        print("Preparing MCQ training data...")
        mcq_train = datapipe.create_training_data_tokenized(task_type='mcq', tokenizer=tokenizer)
        print(f"MCQ training samples: {len(mcq_train)}\n")

        # Finetune on MCQ
        print("Starting MCQ finetuning...")
        model = finetune_model(model, tokenizer, mcq_train, output_dir="./mistral_finetuned_mcq")

        # Prepare SAQ training data
        print("\nPreparing SAQ training data...")
        saq_train = datapipe.create_training_data_tokenized(task_type='saq', tokenizer=tokenizer)
        print(f"SAQ training samples: {len(saq_train)}\n")

        # Finetune on SAQ
        print("Starting SAQ finetuning...")
        model = finetune_model(model, tokenizer, saq_train, output_dir="./mistral_finetuned_saq")

        print("\nFinetuning complete!\n")

    if generate_train_predictions:
        start_inference_process_training(tokenizer, model, n_samples)

    if generate_test_predictions:
        start_inference_process_testing(tokenizer, model, n_samples)
    
    evaluate_results()

if __name__ == "__main__":
    main()
