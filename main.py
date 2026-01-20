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
import matplotlib.pyplot as plt # type: ignore
import logging
from pathlib import Path
from dataclasses import asdict, dataclass

from transformers import ( # type: ignore
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training # type: ignore
from datasets import concatenate_datasets, DatasetDict

import datapipe
import eval

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
    zip_files = []
    #model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    cache_dir: str = str(CACHE_DIR)
    output_base_dir: str = str(MODELS_DIR)
    combined_model: bool = True
		
    # LoRA Configuration
    lora_r: int = 8 # Defines the precision of the output Matrix (higher rank = more parameters are trained)
    lora_alpha: int = 8 # multiplyer applied to the weight changes when added to the original weights (scale= alpha/r)
    lora_dropout: float = 0.1 # is the percentage that randomly leaves out some weight changes each time to deter overfitting

    # Training Configuration
    num_epochs: int = 8
    batch_size: int = 4 # sets how many examples are processed on each GPU/device per forward pass
    gradient_accumulation_steps: int = 1 # simulate larger batches by accumulating gradients across multiple steps before updating weights
    learning_rate: float = 1e-4 # How large should each eight update be
    warmup_steps: int = 100 # gradually increases the learning rate from zero over the first N steps (stabilizes early training)
    weight_decay: float = 0.05 # adds L2 regularization to prevent overfitting.
    max_grad_norm: float = 0.3 # clips gradients to prevent extreme updates that could destabilize training
    safe_steps: int = 400
    val_set_size: float = 0.1 # None to disable testing set out of training data

    # Data Configuration
    seed: int = 42
    use_all_answers: bool = True # gives around 2% of score
    data_augmentation: bool = False
    augmentation_factor: float = 1.5

    # RAG Configuration (NEW!)
    use_rag: bool = True  # Enable RAG augmentation
    rag_k: int = 3  # Number of passages to retrieve per question
    rag_method: str = "hybrid"  # "sparse", "dense", or "hybrid"
    rag_cache_dir: str = "rag_cache"  # Where to cache RAG indices
    wikivoyage_path: str = DATASETS_DIR / "wikivoyage.xml"  # Path to Wikivoyage corpus

    # Eval Configuration
    gen_train_preds: bool = True
    eval_train_samples: int = -1
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
            trust_remote_code=True,
            local_files_only=True
        )
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        #self.tokenizer.padding_side = "right"
        
        print(f"✓ Tokenizer loaded from {self.config.model_name}")
        return self.tokenizer
    
    def load_model(self) -> AutoModelForCausalLM:
        """Load base model for fine-tuning"""
        print("Loading base model...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,              # <--- ACTIVATE 4-BIT
            bnb_4bit_quant_type="nf4",      # <--- Use NF4 (Normal Float 4)
            bnb_4bit_compute_dtype=torch.float16, # <--- Compute in 16-bit for stability
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            dtype=torch.bfloat16,
            cache_dir=self.config.cache_dir,
            quantization_config=bnb_config,
            trust_remote_code=True,
            local_files_only=True
        ).to(self.device)
        self.model = prepare_model_for_kbit_training(self.model,use_gradient_checkpointing=False)
        print(f"✓ Model loaded from {self.config.model_name}")
        return self.model
    
    def setup_lora(self) -> AutoModelForCausalLM:
        """Configure and apply LoRA to the model"""
        print("\nSetting up LoRA configuration...")
        
        self.model.config.use_cache = False
        
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.train()
        
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        print(f"\n✓ LoRA Configuration:")
        print(f"  - Rank (r): {self.config.lora_r}")
        print(f"  - Alpha: {self.config.lora_alpha}")
        print(f"  - Dropout: {self.config.lora_dropout}")
        print(f"\nTrainable Parameters: {trainable_params:,} / {total_params:,}")
        print(f"Trainable Ratio: {trainable_params / total_params * 100:.2f}%\n")
        
        return self.model
    
    def prepare_dataset(self, task_type: str):
        """Prepare dataset with optional augmentation and RAG"""
        print(f"Preparing {task_type.upper()} dataset...")
        
        dataset_dict = datapipe.create_training_data_tokenized(
            task_type=task_type,
            tokenizer=self.tokenizer,
            seed=self.config.seed,
            use_all_answers=self.config.use_all_answers,
            augment_data=self.config.data_augmentation,
            augmentation_factor=self.config.augmentation_factor,
            use_rag=self.config.use_rag,
            rag_k=self.config.rag_k,
            rag_method=self.config.rag_method,
            rag_cache_dir=self.config.rag_cache_dir,
            debug=self.config.debug_mode,
            val_set_size=self.config.val_set_size
        )
        
        print(f"✓ {task_type.upper()} dataset ready.")
        return dataset_dict
    

    def finetune(self, dataset_container, task_name: str = "mcq") -> Trainer:
        """Fine-tune model"""
        print(f"\n{'='*60}")
        print(f"Fine-tuning for {task_name.upper()}")
        print(f"{'='*60}\n")
        
        if isinstance(dataset_container, dict) or hasattr(dataset_container, 'keys'):
             train_dataset = dataset_container["train"]
             val_data = dataset_container.get("test") # Might be None
        else:
             # Fallback for old behavior (though we replaced it)
             train_dataset = dataset_container
             val_data = None
        
        eval_strategy = "steps" if val_data else "no"
        load_best_at_end = True if val_data else False
        
        if val_data:
            print(f"Training Samples: {len(train_dataset)} | Validation Samples: {len(val_data)}")
        
        output_dir = os.path.join(self.config.output_base_dir, f"lora_{task_name}")
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            learning_rate=self.config.learning_rate,
            logging_dir=LOGS_DIR,
            logging_steps=200,
            eval_strategy=eval_strategy,
            eval_steps=self.config.safe_steps if val_data else None,
            save_strategy="steps",
            save_steps=self.config.safe_steps,
            save_total_limit=3,
            load_best_model_at_end=load_best_at_end,
            gradient_checkpointing=False, # we use LoRA so dont need it
            max_grad_norm=self.config.max_grad_norm,
            dataloader_pin_memory=True,
            seed=self.config.seed,
        )
        
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
            data_collator=data_collator
        )
        
        trainer.train()
        
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        print("\nGenerating loss curve...")
        
        # Extract logs from trainer state
        history = trainer.state.log_history
        
        # Separate training and validation logs
        # Hugging Face logs are a list of dicts. We filter by keys.
        train_steps = []
        train_loss = []
        eval_steps = []
        eval_loss = []

        for log in history:
            if "loss" in log and "step" in log:
                train_steps.append(log["step"])
                train_loss.append(log["loss"])
            elif "eval_loss" in log and "step" in log:
                eval_steps.append(log["step"])
                eval_loss.append(log["eval_loss"])

        # Create the plot
        plt.figure(figsize=(10, 6))
        
        # Plot training loss
        if train_steps:
            plt.plot(train_steps, train_loss, label="Training Loss", alpha=0.8)
            
        # Plot validation loss (if available)
        if eval_steps:
            plt.plot(eval_steps, eval_loss, label="Validation Loss", marker='o', linestyle='--')

        plt.yscale('log')
        plt.title(f"Training Loss Curve - {task_name.upper()}")
        plt.xlabel("Global Steps")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        
        # Save plot to the same output directory
        plot_path = os.path.join(output_dir, f"loss_curve_{task_name}.png")
        self.config.zip_files.append(plot_path)
        plt.savefig(plot_path, dpi=300)
        plt.close() # Close to free memory
        
        print(f"✓ Loss curve saved to: {plot_path}")
        if task_name != "saq":
            config_path = os.path.join(output_dir, "config.txt")
            with open(config_path, "w", encoding="utf-8") as f:
                f.write("FineTuningConfig Settings:\n")
                f.write("=" * 30 + "\n")
                
                # Convert dataclass to dict and iterate
                for key, value in asdict(self.config).items():
                    f.write(f"{key}: {value}\n")
                
                # Note: 'zip_files' in your dataclass lacks a type hint, 
                # so asdict() might miss it. We manually check for it here just in case:
                if hasattr(self.config, 'zip_files'):
                    f.write(f"zip_files: {self.config.zip_files}\n")
            self.config.zip_files.append(config_path)

        print(f"\n✓ Fine-tuning complete! Saved to: {output_dir}\n")
        return trainer
    
    def finetune_pipeline(self, tasks: list = None) -> None:
        """Run fine-tuning pipeline"""
        print("\n" + "="*60)
        print("LORA FINE-TUNING PIPELINE")
        print("="*60 + "\n")
        
        self.load_tokenizer()
        self.load_model()
        self.setup_lora()
        
        saq_preds = None
        mcq_preds = None
        
        # Train on individual tasks with augmented data
        for task in tasks:
            try:
                if task.lower() == 'mcq':
                    print("\n" + "-"*60)
                    print("MCQ FINE-TUNING")
                    print("-"*60)
                    
                    data = self.prepare_dataset('mcq')
                    self.finetune(data, task_name='mcq')
                    
                    if self.config.gen_train_preds:
                        eval.start_inference_process_training(
                            self.tokenizer, self.model, 
                            n_samples=self.config.eval_train_samples, 
                            task=1, debug=self.config.debug_mode,
                            use_rag=self.config.use_rag
                        )
                        eval._evaluate_mcq_predictions("results/mcq_train.tsv")
                    
                    if self.config.gen_test_preds:
                        mcq_preds = eval.start_inference_process_testing(
                            self.tokenizer, self.model, task=1,
                            debug=self.config.debug_mode,
                            use_rag=self.config.use_rag
                        )
                
                elif task.lower() == 'saq':
                    print("\n" + "-"*60)
                    print("SAQ FINE-TUNING")
                    print("-"*60)
                    
                    data = self.prepare_dataset('saq')
                    self.finetune(data, task_name='saq')
                    
                    if self.config.gen_train_preds:
                        eval.start_inference_process_training(
                            self.tokenizer, self.model,
                            n_samples=self.config.eval_train_samples,
                            task=0, debug=self.config.debug_mode,
                            use_rag=self.config.use_rag
                        )
                        eval._evaluate_saq_predictions("results/saq_train.tsv")
                    
                    if self.config.gen_test_preds:
                        saq_preds = eval.start_inference_process_testing(
                            self.tokenizer, self.model, 
                            task=0, debug=self.config.debug_mode,
                            use_rag=self.config.use_rag
                        )
                elif task.lower() == 'both':
                    print("\n" + "-"*60)
                    print("SAQ FINE-TUNING")
                    print("-"*60)
                    mcq_data = self.prepare_dataset('mcq')
                    saq_data = self.prepare_dataset('saq')

                    data = DatasetDict({"train": concatenate_datasets([mcq_data["train"],saq_data["train"]]).shuffle(seed=self.config.seed), 
                                        "test": concatenate_datasets([mcq_data["test"],saq_data["test"]]).shuffle(seed=self.config.seed)})
                    self.finetune(data, task_name='both')
                    if self.config.gen_train_preds:
                        eval.start_inference_process_training(
                            self.tokenizer, 
                            self.model, 
                            n_samples=self.config.eval_train_samples, 
                            task=0,
                            debug=self.config.debug_mode,
                            use_rag=self.config.use_rag
                        )
                    if self.config.gen_test_preds:
                        saq_preds = eval.start_inference_process_testing(
                            self.tokenizer, 
                            self.model, 
                            task=0,
                            debug=self.config.debug_mode,
                            use_rag=self.config.use_rag
                        )
                    if self.config.gen_train_preds:
                        eval.start_inference_process_training(
                            self.tokenizer, 
                            self.model, 
                            n_samples=self.config.eval_train_samples, 
                            task=1,
                            debug=self.config.debug_mode,
                            use_rag=self.config.use_rag
                        )
                    if self.config.gen_test_preds:
                        mcq_preds = eval.start_inference_process_testing(
                            self.tokenizer, 
                            self.model, 
                            task=1,
                            debug=self.config.debug_mode,
                            use_rag=self.config.use_rag
                        )
            
            except Exception as e:
                print(f"Error during {task.upper()} fine-tuning: {str(e)}")
                raise
        
        if self.config.gen_test_preds:
            eval.create_zip_for_submission(saq_preds, mcq_preds, self.config.zip_files)
        
        print("\n" + "="*60)
        print("✓ FINE-TUNING PIPELINE COMPLETE")
        print("="*60 + "\n")


def main():
    """Main entry point"""
    
    # Create configuration
    config = FineTuningConfig()
    
    # Create pipeline
    pipeline = FineTuningPipeline(config)
    
    # Run fine-tuning
    if config.combined_model:
        pipeline.finetune_pipeline(tasks=['both'])
    else:
        pipeline.finetune_pipeline(tasks=['mcq', 'saq'])
    if config.gen_train_preds:
        eval.evaluate_results()


if __name__ == "__main__":
    main()