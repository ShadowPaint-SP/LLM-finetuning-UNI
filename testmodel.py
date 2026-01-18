"""
Load Fine-tuned Mistral-7B LoRA Model for Evaluation Testing

This script loads a previously fine-tuned model with LoRA adapters
and runs evaluation tasks for testing purposes.
"""

import os
import torch # type: ignore
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from transformers import AutoTokenizer, AutoModelForCausalLM # type: ignore
from peft import PeftModel # type: ignore

import eval

# Suppress warnings
logging.basicConfig(level=logging.INFO)

# Configuration constants
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "finetuned_models"
CACHE_DIR = BASE_DIR / "Model-Cache"


@dataclass
class EvalConfig:
    """Configuration for loading and evaluating fine-tuned models"""
    #base_model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"
    base_model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    cache_dir: str = str(CACHE_DIR)
    
    # Model paths (will be constructed based on task)
    mcq_model_path: str = str(MODELS_DIR / "lora_mcq")
    saq_model_path: str = str(MODELS_DIR / "lora_saq")
    both_model_path: str = str(MODELS_DIR / "lora_both")
    
    # Checkpoint selection
    # Options: 'best', 'latest', or specific checkpoint like 'checkpoint-500'
    task_to_evaluate: str = 'saq'
    checkpoint: Optional[str] = 'checkpoint-18500'
    
    # Evaluation settings
    debug_mode: bool = True
    eval_train_samples: int = 400
    
    # Which evaluations to run
    eval_mcq_train: bool = False
    eval_mcq_test: bool = True
    eval_saq_train: bool = False
    eval_saq_test: bool = True


class ModelLoader:
    """Load and manage fine-tuned LoRA models"""
    
    def __init__(self, config: EvalConfig):
        self.config = config
        self.device = config.device
        print(f"Device: {self.device}\n")
    
    def load_finetuned_model(self, model_path: str, task_name: str):
        """
        Load a fine-tuned LoRA model
        
        Args:
            model_path: Path to the fine-tuned model directory
            task_name: Name of the task (for logging)
        
        Returns:
            tuple: (tokenizer, model)
        """
        print(f"\n{'='*60}")
        print(f"Loading Fine-tuned Model: {task_name.upper()}")
        print(f"{'='*60}\n")
        
        # Check if model path exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}\n"
                f"Please run fine-tuning first or check the path."
            )
        
        # Determine which checkpoint to load
        final_model_path = self._get_checkpoint_path(model_path)
        print(f"Model path: {final_model_path}")
        
        # Load tokenizer
        print("\nLoading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            final_model_path,
            trust_remote_code=True
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        print("✓ Tokenizer loaded")
        
        # Load base model
        print("\nLoading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model_name,
            dtype=torch.bfloat16,
            cache_dir=self.config.cache_dir,
            trust_remote_code=True
        )
        print("✓ Base model loaded")
        
        # Load LoRA adapters
        print("\nLoading LoRA adapters...")
        model = PeftModel.from_pretrained(
            base_model,
            final_model_path,
            torch_dtype=torch.bfloat16
        ).to(self.device)
        
        # Merge adapters for faster inference (optional)
        # Uncomment the next line if you want to merge LoRA weights into base model
        # model = model.merge_and_unload()
        
        model.eval()
        print("✓ LoRA adapters loaded")
        
        print(f"\n✓ Model successfully loaded from {final_model_path}\n")
        
        return tokenizer, model
    
    def _get_checkpoint_path(self, base_path: str) -> str:
        """
        Determine which checkpoint to load based on config
        
        Args:
            base_path: Base model directory path
        
        Returns:
            str: Path to the checkpoint to load
        """
        checkpoint_option = self.config.checkpoint
        
        # If checkpoint is None or base path has no checkpoints, use base path
        checkpoint_dirs = [d for d in os.listdir(base_path) 
                          if d.startswith('checkpoint-') and 
                          os.path.isdir(os.path.join(base_path, d))]
        
        if not checkpoint_dirs:
            print(f"No checkpoints found in {base_path}, using base model directory")
            return base_path
        
        if checkpoint_option is None or checkpoint_option == 'base':
            print(f"Using base model directory (no checkpoint)")
            return base_path
        
        elif checkpoint_option == 'best':
            # Since you use load_best_model_at_end, the best checkpoint 
            # will be saved to the base directory at the end of training
            print(f"Loading best checkpoint from base directory")
            return base_path
        
        elif checkpoint_option == 'latest':
            # Find the latest checkpoint by number
            checkpoint_nums = [int(d.split('-')[1]) for d in checkpoint_dirs]
            latest_num = max(checkpoint_nums)
            latest_checkpoint = f"checkpoint-{latest_num}"
            checkpoint_path = os.path.join(base_path, latest_checkpoint)
            print(f"Loading latest checkpoint: {latest_checkpoint}")
            return checkpoint_path
        
        elif checkpoint_option.startswith('checkpoint-'):
            # Load specific checkpoint
            checkpoint_path = os.path.join(base_path, checkpoint_option)
            if not os.path.exists(checkpoint_path):
                available = ', '.join(sorted(checkpoint_dirs))
                raise FileNotFoundError(
                    f"Checkpoint '{checkpoint_option}' not found.\n"
                    f"Available checkpoints: {available}"
                )
            print(f"Loading specific checkpoint: {checkpoint_option}")
            return checkpoint_path
        
        else:
            # Try to interpret as a step number
            try:
                step_num = int(checkpoint_option)
                checkpoint_name = f"checkpoint-{step_num}"
                checkpoint_path = os.path.join(base_path, checkpoint_name)
                if not os.path.exists(checkpoint_path):
                    available = ', '.join(sorted(checkpoint_dirs))
                    raise FileNotFoundError(
                        f"Checkpoint at step {step_num} not found.\n"
                        f"Available checkpoints: {available}"
                    )
                print(f"Loading checkpoint at step {step_num}")
                return checkpoint_path
            except ValueError:
                raise ValueError(
                    f"Invalid checkpoint option: {checkpoint_option}\n"
                    f"Valid options: 'best', 'latest', 'checkpoint-XXX', or step number"
                )
    
    def list_available_checkpoints(self, model_path: str):
        """List all available checkpoints for a model"""
        if not os.path.exists(model_path):
            print(f"Model path does not exist: {model_path}")
            return
        
        checkpoint_dirs = [d for d in os.listdir(model_path) 
                          if d.startswith('checkpoint-') and 
                          os.path.isdir(os.path.join(model_path, d))]
        
        if checkpoint_dirs:
            print(f"\nAvailable checkpoints in {model_path}:")
            for cp in sorted(checkpoint_dirs):
                print(f"  - {cp}")
        else:
            print(f"\nNo checkpoints found in {model_path}")
        
        print(f"  - base (model directory itself)")
        print()
    
    def evaluate_model(self, tokenizer, model, task_name: str):
        """
        Run evaluation on a loaded model
        
        Args:
            tokenizer: Loaded tokenizer
            model: Loaded model
            task_name: 'mcq', 'saq', or 'both'
        """
        print(f"\n{'='*60}")
        print(f"Running Evaluation: {task_name.upper()}")
        print(f"{'='*60}\n")
        
        saq_preds = None
        mcq_preds = None
        
        if task_name in ['mcq', 'both']:
            print("\n" + "-"*60)
            print("MCQ EVALUATION")
            print("-"*60)
            
            if self.config.eval_mcq_train:
                print("\n→ Generating MCQ training predictions...")
                eval.start_inference_process_training(
                    tokenizer,
                    model,
                    n_samples=self.config.eval_train_samples,
                    task=1,  # MCQ task
                    debug=self.config.debug_mode
                )
                print("\n→ Evaluating MCQ training predictions...")
                eval._evaluate_mcq_predictions("results/mcq_train.tsv")
            
            if self.config.eval_mcq_test:
                print("\n→ Generating MCQ test predictions...")
                mcq_preds = eval.start_inference_process_testing(
                    tokenizer,
                    model,
                    task=1,  # MCQ task
                    debug=self.config.debug_mode
                )
                mcq_preds.to_csv("mcq_prediction.tsv", sep='\t', index=False)
        
        if task_name in ['saq', 'both']:
            print("\n" + "-"*60)
            print("SAQ EVALUATION")
            print("-"*60)
            
            if self.config.eval_saq_train:
                print("\n→ Generating SAQ training predictions...")
                eval.start_inference_process_training(
                    tokenizer,
                    model,
                    n_samples=self.config.eval_train_samples,
                    task=0,  # SAQ task
                    debug=self.config.debug_mode
                )
                print("\n→ Evaluating SAQ training predictions...")
                eval._evaluate_saq_predictions("results/saq_train.tsv")
            
            if self.config.eval_saq_test:
                print("\n→ Generating SAQ test predictions...")
                saq_preds = eval.start_inference_process_testing(
                    tokenizer,
                    model,
                    task=0,  # SAQ task
                    debug=self.config.debug_mode
                )
                saq_preds.to_csv("saq_prediction.tsv", sep='\t', index=False)
        
        print(f"\n✓ Evaluation complete for {task_name.upper()}")


def main():
    """Main entry point for loading and evaluating fine-tuned models"""
    
    # Parse arguments (if you have a parse_args function)
    # args = parse_args()
    
    # Create configuration
    config = EvalConfig(
    )
    
    # Create loader
    loader = ModelLoader(config)
    
    # Choose which model to load and evaluate
    # Options: 'mcq', 'saq', or 'both'
    
    if config.task_to_evaluate == 'mcq':
        model_path = config.mcq_model_path
    elif config.task_to_evaluate == 'saq':
        model_path = config.saq_model_path
    elif config.task_to_evaluate == 'both':
        model_path = config.both_model_path
    else:
        raise ValueError(f"Invalid task: {config.task_to_evaluate}")
    
    # Load the fine-tuned model
    tokenizer, model = loader.load_finetuned_model(
        model_path=model_path,
        task_name=config.task_to_evaluate
    )
    
    # Run evaluation
    loader.evaluate_model(tokenizer, model, task_name=config.task_to_evaluate)
    
    # Optionally run final evaluation summary
    print("\n" + "="*60)
    print("FINAL EVALUATION SUMMARY")
    print("="*60 + "\n")
    eval.evaluate_results()
    
    print("\n" + "="*60)
    print("✓ EVALUATION COMPLETE")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()