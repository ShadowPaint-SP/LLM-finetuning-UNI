"""
Mistral-7B Fine-tuning Pipeline for Cultural Q&A

This script implements a complete fine-tuning pipeline using:
- LoRA (Low-Rank Adaptation) for efficient parameter updates
- Full fine-tuning option for complete model training
- Mistral-7B-Instruct-v0.2 as the base model
- Custom datapipe for MCQ and SAQ training data
- Evaluation utilities for model assessment
"""

import os
import torch # type: ignore
import logging
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from transformers import ( # type: ignore
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training # type: ignore
from datasets import Dataset, concatenate_datasets

import datapipe
import eval
from utils import parse_args

# Suppress warnings
logging.basicConfig(level=logging.INFO)
torch.utils.checkpoint.use_reentrant = False

# Configuration constants
BASE_DIR = Path(__file__).resolve().parent
DATASETS_DIR = BASE_DIR / "datasets"
CACHE_DIR = BASE_DIR / "Model-Cache"
MODELS_DIR = BASE_DIR / "finetuned_models"
LOGS_DIR = BASE_DIR / "logs"


@dataclass
class FineTuningConfig:
    """Configuration for fine-tuning pipeline"""
    debug_mode: bool = True
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    cache_dir: str = str(CACHE_DIR)
    output_base_dir: str = str(MODELS_DIR)
    seperate_models: bool = False
    
    # Fine-tuning mode
    use_lora: bool = True  # Set to False for full fine-tuning
    use_quantization: bool = False  # Set to True to enable 4-bit quantization (only for LoRA)
    quantization_type: str = "nf4"  # "nf4" or "fp4" - nf4 is generally better
    
    # LoRA Configuration (only used if use_lora=True)
    lora_r: int = 8 # Defines the precision of the output Matrix (higher rank = more parameters are trained) - REDUCE to 4 for less overfitting
    lora_alpha: int = 32 # multiplyer applied to the weight changes when added to the original weights (scale= alpha/r)
    lora_dropout: float = 0.2 # is the percentage that randomly leaves out some weight changes each time to deter overfitting - INCREASE to 0.3-0.5 for less overfitting
    lora_layers = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"]  # REDUCE layers to target only q_proj, v_proj for less overfitting
    
    # Training Configuration
    num_epochs: int = 4  # REDUCE to 2-3 if overfitting
    batch_size: int = 4 # sets how many examples are processed on each GPU/device per forward pass (can use 8-16 for 80GB GPU)
    gradient_accumulation_steps: int = 4 # simulate larger batches by accumulating gradients across multiple steps before updating weights
    learning_rate: float = 2e-4 # How large should each weight update be (use 2e-5 for full fine-tuning) - REDUCE to 1e-4 or 5e-5 if overfitting
    warmup_steps: int = 100 # gradually increases the learning rate from zero over the first N steps (stabilizes early training)
    weight_decay: float = 0.01 # adds L2 regularization to prevent overfitting. INCREASE to 0.05-0.1 if overfitting
    max_grad_norm: float = 0.3 # clips gradients to prevent extreme updates that could destabilize training
    safe_steps: int = 100
    neftune_noise_alpha: int = 5
    val_set_size: float = None # None to disable testing set out of training data - SET to 0.1-0.2 to monitor overfitting
    
    # Data Configuration
    max_train_samples: Optional[int] = None
    seed: int = 42
    use_all_answers: bool = False
    weight_sampling: bool = False

    # Eval Configuration
    gen_train_preds: bool = True
    eval_train_samples: int = -1
    gen_test_preds: bool = True
    

class FineTuningPipeline:
    """Complete fine-tuning pipeline for LoRA or full fine-tuning"""
    
    def __init__(self, config: FineTuningConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.device = config.device
        
        # Validate quantization settings
        if config.use_quantization and not config.use_lora:
            raise ValueError("Quantization is only supported with LoRA fine-tuning. Set use_lora=True.")
        
        # Create output directories
        os.makedirs(self.config.output_base_dir, exist_ok=True)
        os.makedirs(LOGS_DIR, exist_ok=True)
        
        print(f"Device: {self.device}")
        print(f"Fine-tuning Mode: {'LoRA' if self.config.use_lora else 'Full Fine-tuning'}")
        if self.config.use_lora and self.config.use_quantization:
            print(f"Quantization: Enabled ({self.config.quantization_type.upper()})")
        print(f"Config: {config}\n")
    
    def load_tokenizer(self) -> AutoTokenizer:
        """Load tokenizer from pretrained model"""
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            cache_dir=self.config.cache_dir,
            trust_remote_code=True
        )
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
        print(f"✓ Tokenizer loaded from {self.config.model_name}")
        return self.tokenizer
    
    def load_model(self) -> AutoModelForCausalLM:
        """Load base model for fine-tuning"""
        print("Loading base model...")
        
        # Setup quantization config if enabled
        quantization_config = None
        if self.config.use_lora and self.config.use_quantization:
            print(f"Setting up {self.config.quantization_type.upper()} quantization...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.config.quantization_type,  # "nf4" or "fp4"
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,  # Nested quantization for more memory savings
            )
        
        # Load model with or without quantization
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=quantization_config,
            dtype=torch.bfloat16 if not self.config.use_quantization else None,
            cache_dir=self.config.cache_dir,
            trust_remote_code=True,
            device_map="auto" if self.config.use_quantization else None,
        )
        
        if not self.config.use_quantization:
            self.model = self.model.to(self.device)
        
        print(f"✓ Model loaded from {self.config.model_name}")
        if self.config.use_quantization:
            print(f"✓ Model quantized to 4-bit ({self.config.quantization_type.upper()})")
        return self.model
    
    def setup_lora(self) -> AutoModelForCausalLM:
        """Configure and apply LoRA to the model"""
        print("\nSetting up LoRA configuration...")
        
        # Prepare model for k-bit training if using quantization
        if self.config.use_quantization:
            print("Preparing model for k-bit training...")
            self.model = prepare_model_for_kbit_training(self.model)
        
        # Disable cache
        self.model.config.use_cache = False
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_layers,
            task_type="CAUSAL_LM"
        )
        
        # Apply LoRA to model
        self.model = get_peft_model(self.model, lora_config)
        self.model.train()
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        print(f"\n✓ LoRA Configuration:")
        print(f"  - Rank (r): {self.config.lora_r}")
        print(f"  - Alpha: {self.config.lora_alpha}")
        print(f"  - Dropout: {self.config.lora_dropout}")
        if self.config.use_quantization:
            print(f"  - Quantization: {self.config.quantization_type.upper()} (4-bit)")
        print(f"\nTrainable Parameters: {trainable_params:,} / {total_params:,}")
        print(f"Trainable Ratio: {trainable_params / total_params * 100:.2f}%\n")
        
        return self.model
    
    def setup_full_finetuning(self) -> AutoModelForCausalLM:
        """Configure model for full fine-tuning"""
        print("\nSetting up full fine-tuning...")
        
        # Disable cache for training
        self.model.config.use_cache = False
        
        # Enable gradient checkpointing to save memory
        self.model.gradient_checkpointing_enable()
        
        # Unfreeze all parameters
        for param in self.model.parameters():
            param.requires_grad = True
        
        self.model.train()
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        print(f"\n✓ Full Fine-tuning Configuration:")
        print(f"Trainable Parameters: {trainable_params:,} / {total_params:,}")
        print(f"Trainable Ratio: {trainable_params / total_params * 100:.2f}%")
        print(f"NOTE: All model parameters will be updated during training\n")
        
        return self.model
    
    def finetune(self, train_dataset: Dataset, task_name: str = "mcq") -> Trainer:
        """Fine-tune model on training dataset"""
        mode_suffix = "lora" if self.config.use_lora else "full"
        print(f"\n{'='*60}")
        print(f"Starting {mode_suffix.upper()} Fine-tuning for {task_name.upper()}")
        print(f"{'='*60}\n")

        eval_strategy = "no"
        val_data = None
        load_best_at_end = False

        if self.config.val_set_size is not None:
            dataset_split = train_dataset.train_test_split(test_size=self.config.val_set_size, seed=self.config.seed)
            train_dataset = dataset_split["train"]
            val_data = dataset_split["test"]
            eval_strategy = "steps"
            load_best_at_end = True
            print(f"Dataset split: {len(train_dataset)} training samples | {len(val_data)} validation samples")

        
        output_dir = os.path.join(self.config.output_base_dir, f"{mode_suffix}_{task_name}")

        # Adjust learning rate for full fine-tuning if needed
        learning_rate = self.config.learning_rate
        if not self.config.use_lora and learning_rate > 5e-5:
            print(f"WARNING: Learning rate {learning_rate} may be too high for full fine-tuning.")
            print(f"Consider using a lower learning rate (e.g., 2e-5 to 5e-5)\n")

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            learning_rate=learning_rate,
            neftune_noise_alpha=self.config.neftune_noise_alpha,
            bf16=True,
            logging_dir=LOGS_DIR,
            logging_steps=10,
            eval_strategy=eval_strategy,
            eval_steps=self.config.safe_steps if val_data else None,
            save_strategy="steps",
            save_steps=self.config.safe_steps,
            save_total_limit=3,
            load_best_model_at_end=load_best_at_end,
            gradient_checkpointing=not self.config.use_lora,  # Enable for full fine-tuning to save memory
            max_grad_norm=self.config.max_grad_norm,
            dataloader_pin_memory=True,
            seed=self.config.seed,
        )
        
        # Use proper data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True,
            return_tensors="pt"
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_data,
            data_collator=data_collator,
        )
        
        # Train
        trainer.train()
        
        # Save model
        if self.config.use_lora:
            # For LoRA, save adapter weights
            self.model.save_pretrained(output_dir)
        else:
            # For full fine-tuning, save the entire model
            self.model.save_pretrained(output_dir)
        
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"\n✓ Fine-tuning complete!")
        print(f"  - Model saved to: {output_dir}\n")
        
        return trainer
    
    
    def finetune_pipeline(self, tasks: list) -> None:
        """Run complete fine-tuning pipeline for specified tasks"""
        
        print("\n" + "="*60)
        print("MISTRAL-7B FINE-TUNING PIPELINE")
        print("="*60 + "\n")
        
        # Load tokenizer and model
        self.load_tokenizer()
        self.load_model()
        
        # Setup model based on fine-tuning mode
        if self.config.use_lora:
            self.setup_lora()
        else:
            self.setup_full_finetuning()
        
        saq_preds = None
        mcq_preds = None
        
        # Fine-tune for each task
        for task in tasks:
            try:
                if task.lower() == 'mcq':
                    print("\n" + "-"*60)
                    print("MCQ FINE-TUNING")
                    print("-"*60)
                    data = datapipe.create_training_data_tokenized(
                        task_type='mcq', 
                        tokenizer=self.tokenizer,
                        use_all_answers=self.config.use_all_answers, 
                        debug=self.config.debug_mode
                    )
                    
                    self.finetune(data, task_name='mcq')
                    if self.config.gen_train_preds:
                        eval.start_inference_process_training(
                            self.tokenizer, 
                            self.model, 
                            n_samples=self.config.eval_train_samples, 
                            task=1,
                            debug=self.config.debug_mode
                        )
                        eval._evaluate_mcq_predictions("results/mcq_train.tsv")
                    if self.config.gen_test_preds:
                        mcq_preds = eval.start_inference_process_testing(
                            self.tokenizer, 
                            self.model, 
                            task=1,
                            debug=self.config.debug_mode
                        )
                elif task.lower() == 'saq':
                    print("\n" + "-"*60)
                    print("SAQ FINE-TUNING")
                    print("-"*60)
                    data = datapipe.create_training_data_tokenized(
                        task_type='saq', 
                        tokenizer=self.tokenizer, 
                        seed=self.config.seed,
                        use_all_answers=self.config.use_all_answers, 
                        weight_sampling=self.config.weight_sampling,
                        debug=self.config.debug_mode
                    )
                    
                    self.finetune(data, task_name='saq')
                    if self.config.gen_train_preds:
                        eval.start_inference_process_training(
                            self.tokenizer, 
                            self.model, 
                            n_samples=self.config.eval_train_samples, 
                            task=0,
                            debug=self.config.debug_mode
                        )
                        eval._evaluate_saq_predictions("results/saq_train.tsv")
                    if self.config.gen_test_preds:
                        saq_preds = eval.start_inference_process_testing(
                            self.tokenizer, 
                            self.model, 
                            task=0,
                            debug=self.config.debug_mode
                        )
                elif task.lower() == 'both':
                    data1 = datapipe.create_training_data_tokenized(
                        task_type='mcq', 
                        tokenizer=self.tokenizer,
                        use_all_answers=self.config.use_all_answers, 
                        debug=self.config.debug_mode
                    )
                    data2 = datapipe.create_training_data_tokenized(
                        task_type='saq', 
                        tokenizer=self.tokenizer, 
                        seed=self.config.seed, 
                        use_all_answers=self.config.use_all_answers, 
                        weight_sampling=self.config.weight_sampling, 
                        debug=self.config.debug_mode
                    )
                    data = concatenate_datasets([data1,data2]).shuffle(seed=self.config.seed)
                    self.finetune(data, task_name='both')
                    if self.config.gen_train_preds:
                        eval.start_inference_process_training(
                            self.tokenizer, 
                            self.model, 
                            n_samples=self.config.eval_train_samples, 
                            task=0,
                            debug=self.config.debug_mode
                        )
                    if self.config.gen_test_preds:
                        saq_preds = eval.start_inference_process_testing(
                            self.tokenizer, 
                            self.model, 
                            task=0,
                            debug=self.config.debug_mode
                        )
                    if self.config.gen_train_preds:
                        eval.start_inference_process_training(
                            self.tokenizer, 
                            self.model, 
                            n_samples=self.config.eval_train_samples, 
                            task=1,
                            debug=self.config.debug_mode
                        )
                    if self.config.gen_test_preds:
                        mcq_preds = eval.start_inference_process_testing(
                            self.tokenizer, 
                            self.model, 
                            task=1,
                            debug=self.config.debug_mode
                        )
                    
            except Exception as e:
                print(f"Error during {task.upper()} fine-tuning: {str(e)}")
                raise
        
        if self.config.gen_test_preds:
            eval.create_zip_for_submission(saq_preds, mcq_preds)
        print("\n" + "="*60)
        print("✓ FINE-TUNING PIPELINE COMPLETE")
        print("="*60 + "\n")


def main():
    """Main entry point"""
    # Parse arguments
    args = parse_args()
    
    # Create configuration
    config = FineTuningConfig(
        #model_name=args.model_name,
        #num_epochs=args.num_train_epochs,
        #max_train_samples=args.max_train_samples,
        #gen_test_preds=args.gen_test_preds,
        #gen_train_preds=args.gen_train_preds,
        #eval_train_samples=args.eval_train_samples,
        #debug_mode=args.debug
        #use_lora=args.use_lora,  # Add this to your argument parser
    )
    
    # Create pipeline
    pipeline = FineTuningPipeline(config)
    
    if config.seperate_models:
        pipeline.finetune_pipeline(tasks=['mcq', 'saq'])
    else:
        pipeline.finetune_pipeline(tasks=['both'])
    eval.evaluate_results()

if __name__ == "__main__":
    main()