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
    debug_mode: bool = False
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    cache_dir: str = str(CACHE_DIR)
    output_base_dir: str = str(MODELS_DIR)
    
    # LoRA Configuration
    lora_r: int = 32 # Defines the precision of the output Matrix (higher rank = more parameters are trained)
    lora_alpha: int = 64 # multiplyer applied to the weight changes when added to the original weights (scale= alpha/r)
    lora_dropout: float = 0.1 # is the percentage that randomly leaves out some weight changes each time to deter overfitting
    
    # Training Configuration
    num_epochs: int = 3
    batch_size: int = 4 # sets how many examples are processed on each GPU/device per forward pass
    gradient_accumulation_steps: int = 2 # simulate larger batches by accumulating gradients across multiple steps before updating weights
    learning_rate: float = 1e-4 # How large should each eight update be
    warmup_steps: int = 100 # gradually increases the learning rate from zero over the first N steps (stabilizes early training)
    weight_decay: float = 0.01 # adds L2 regularization to prevent overfitting.
    max_grad_norm: float = 0.3 # clips gradients to prevent extreme updates that could destabilize training
    safe_steps: int = 100
    # Data Configuration
    max_train_samples: Optional[int] = None
    seed: int = 42
    use_all_answers: bool = True
    weight_sampling: bool = False

    # Eval Configuration
    gen_train_preds: bool = True
    eval_train_samples: int = -1
    gen_test_preds: bool = True
    


class FineTuningPipeline:
    """Complete fine-tuning pipeline for Mistral-7B with LoRA"""
    
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
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"✓ Tokenizer loaded from {self.config.model_name}")
        return self.tokenizer
    
    def load_model(self) -> AutoModelForCausalLM:
        """Load base model for fine-tuning"""
        print("Loading base model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            dtype=torch.bfloat16,
            cache_dir=self.config.cache_dir,
            trust_remote_code=True
        ).to(self.device)
        print(f"✓ Model loaded from {self.config.model_name}")
        return self.model
    
    def setup_lora(self) -> AutoModelForCausalLM:
        """Configure and apply LoRA to the model"""
        print("\nSetting up LoRA configuration...")
        
        # Disable cache and enable gradient checkpointing
        self.model.config.use_cache = False
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
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
        print(f"\nTrainable Parameters: {trainable_params:,} / {total_params:,}")
        print(f"Trainable Ratio: {trainable_params / total_params * 100:.2f}%\n")
        
        return self.model
    
    def finetune(self, train_dataset: Dataset, task_name: str = "mcq") -> Trainer:
        """Fine-tune model on training dataset"""
        print(f"\n{'='*60}")
        print(f"Starting LoRA Fine-tuning for {task_name.upper()}")
        print(f"{'='*60}\n")
        
        output_dir = os.path.join(self.config.output_base_dir, f"lora_{task_name}")

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            learning_rate=self.config.learning_rate,
            bf16=True, # uses bfloat16 (16-bit) precision instead of float32, reducing memory usage and speeding up computation with minimal accuracy loss
            logging_dir=LOGS_DIR,
            logging_steps=10,
            save_steps=self.config.safe_steps,
            save_total_limit=3,
            eval_strategy="no",
            save_strategy="steps",
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
                        print("\n*** Running MCQ Evaluation ***")
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
                        print("\n*** Running SAQ Evaluation ***")
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
                    print("Fine-tuning both MCQ and SAQ is not yet implemented.")
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
                        print("\n*** Running SAQ Evaluation ***")
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
                    if self.config.gen_train_preds:
                        print("\n*** Running MCQ Evaluation ***")
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
    pipeline.finetune_pipeline(tasks=['mcq', 'saq'])
    #pipeline.finetune_pipeline(tasks=['saq'])
    eval.evaluate_results()

if __name__ == "__main__":
    main()