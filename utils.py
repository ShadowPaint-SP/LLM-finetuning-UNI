import argparse
import json


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

