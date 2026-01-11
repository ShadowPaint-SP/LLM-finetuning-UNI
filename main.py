"""
Mistral-7B LoRA Fine-tuning Pipeline for Cultural Q&A

This script implements a complete fine-tuning pipeline using:
- LoRA (Low-Rank Adaptation) for efficient parameter updates
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
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model # type: ignore
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
    debug_mode: bool = True  # Turn off for actual training
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    cache_dir: str = str(CACHE_DIR)
    output_base_dir: str = str(MODELS_DIR)
    
    # LoRA Configuration - Optimized for LLaMA 3
    lora_r: int = 16  # 16-32 works well for 8B models
    lora_alpha: int = 32  # Typically 2x the rank
    lora_dropout: float = 0.05  # Lower dropout for smaller models
    lora_target_modules: list = None  # Will be set in setup_lora
    
    # Training Configuration
    num_epochs: int = 3  # Start with fewer epochs
    batch_size: int = 4
    gradient_accumulation_steps: int = 4  # Increase if memory allows
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0  # Increase slightly for stability
    save_steps: int = 100
    neftune_noise_alpha: int = 0  # Try without noise first
    val_set_size: float = 0.1  # Use 10% for validation
    
    # Data Configuration
    max_train_samples: Optional[int] = None
    seed: int = 42
    use_all_answers: bool = True
    weight_sampling: bool = False

    # Eval Configuration
    gen_train_preds: bool = True
    eval_train_samples: int = 400
    gen_test_preds: bool = True
    

class FineTuningPipeline:
    """Complete fine-tuning pipeline for LoRA"""
    
    def __init__(self, config: FineTuningConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.device = config.device
        
        # Create output directories
        os.makedirs(self.config.output_base_dir, exist_ok=True)
        os.makedirs(LOGS_DIR, exist_ok=True)
        
        print(f"Device: {self.device}")
        print(f"Config: {config}\n")
    
    def load_tokenizer(self) -> AutoTokenizer:
        """Load tokenizer from pretrained model"""
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            cache_dir=self.config.cache_dir,
            trust_remote_code=True
        )
        
        # For LLaMA 3, set pad token to eos_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # LLaMA 3 works better with left padding for generation
        self.tokenizer.padding_side = "right"  # Keep right for training
        
        print(f"✓ Tokenizer loaded from {self.config.model_name}")
        print(f"  - Vocab size: {len(self.tokenizer)}")
        print(f"  - PAD token: {self.tokenizer.pad_token} (ID: {self.tokenizer.pad_token_id})")
        print(f"  - EOS token: {self.tokenizer.eos_token} (ID: {self.tokenizer.eos_token_id})")
        return self.tokenizer
    
    def load_model(self) -> AutoModelForCausalLM:
        """Load base model for fine-tuning"""
        print("Loading base model...")
        
        # For LLaMA 3, you might want to use flash attention if available
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                dtype=torch.bfloat16,
                cache_dir=self.config.cache_dir,
                trust_remote_code=True,
                device_map="auto"
            )
        except Exception as e:
            print(f"Flash attention not available, using default: {e}")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.bfloat16,
                cache_dir=self.config.cache_dir,
                trust_remote_code=True,
                device_map="auto"
            )
        
        print(f"✓ Model loaded from {self.config.model_name}")
        return self.model
    
    def setup_lora(self) -> AutoModelForCausalLM:
        """Configure and apply LoRA to the model"""
        print("\nSetting up LoRA configuration...")
        
        # Disable cache and enable gradient checkpointing
        self.model.config.use_cache = False
        self.model.gradient_checkpointing_enable()
        
        # For LLaMA 3, target these modules for best results
        target_modules = [
            "q_proj",
            "k_proj", 
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ]
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=target_modules,
            task_type="CAUSAL_LM",
            bias="none"  # Don't train biases
        )
        
        # Apply LoRA to model
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()  # Nice built-in method
        
        return self.model
    
    def finetune(self, train_dataset: Dataset, task_name: str = "mcq") -> Trainer:
        """Fine-tune model on training dataset"""
        print(f"\n{'='*60}")
        print(f"Starting LoRA Fine-tuning for {task_name.upper()}")
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

        
        output_dir = os.path.join(self.config.output_base_dir, f"lora_{task_name}")

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            learning_rate=self.config.learning_rate,
            neftune_noise_alpha=self.config.neftune_noise_alpha,
            bf16=True,
            logging_dir=LOGS_DIR,
            logging_steps=10,
            eval_strategy=eval_strategy,
            eval_steps=self.config.save_steps if val_data else None,
            save_strategy="steps",
            save_steps=self.config.save_steps,
            save_total_limit=3,
            load_best_model_at_end=load_best_at_end,
            gradient_checkpointing=False, # we use LoRA so dont needed
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
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"\n✓ Fine-tuning complete!")
        print(f"  - Model saved to: {output_dir}\n")
        
        return trainer
    
    
    def finetune_pipeline(self, tasks: list = None) -> None:
        """Run complete fine-tuning pipeline for specified tasks"""
        if tasks is None:
            tasks = ['mcq', 'saq']
        
        print("\n" + "="*60)
        print("MISTRAL-7B LORA FINE-TUNING PIPELINE")
        print("="*60 + "\n")
        
        # Load tokenizer and model
        self.load_tokenizer()
        self.load_model()
        self.setup_lora()
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
    )
    
    # Create pipeline
    pipeline = FineTuningPipeline(config)
    
    # Run fine-tuning
    #pipeline.finetune_pipeline(tasks=['mcq', 'saq'])
    pipeline.finetune_pipeline(tasks=['both'])
    eval.evaluate_results()

if __name__ == "__main__":
    main()