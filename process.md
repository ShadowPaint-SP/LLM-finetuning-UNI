alpha: 
base model

beta: 
our own first changes. Those include:
- reduce answer length from 100 to 20 tokens (SAQ and MCQ)
- change 


# base results on plain Mistral 7B

These results are from running our eval just with our system prompt over the trainingdata to see how the model performs on it by default.

========================================
SAQ EVALUATION RESULTS
========================================
Total Questions Evaluated:  1333
Exact Matches (Full Score): 105
Total Achieved Score:       467
Total Max Score:            5270
Overall Accuracy:           8.86%
========================================

========================================
MCQ EVALUATION RESULTS
========================================
Total Questions Evaluated: 836
Correct Predictions:       569
Accuracy:                  68.06%
========================================



# Fine Tuning with LoRA


## independent Models
here we are fine tuning the model for each task independently so in the end each model is only trained on the corresponding data and should thus achieve a higher score. Including the other data would decrease the accuracy as the model gets trained on multiple tasks and is less precise i think.

========================================
SAQ EVALUATION RESULTS
========================================
Total Questions Evaluated:  1333
Exact Matches (Full Score): 1253
Total Achieved Score:       4304
Total Max Score:            4526
Overall Accuracy:           95.10%
========================================

========================================
MCQ EVALUATION RESULTS
========================================
Total Questions Evaluated: 836
Correct Predictions:       812
Accuracy:                  97.13%
========================================

testing data results: MCQ: 74% SAQ: 57%
This indicates overfitting

### Current Config

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

```python
# LoRA Configuration
lora_r: int = 32
lora_alpha: int = 64
lora_dropout: float = 0.05
lora_target_modules: list = ["q_proj", "v_proj", "k_proj", "o_proj"]

# Training Configuration
num_epochs: int = 4
batch_size: int = 4
gradient_accumulation_steps: int = 4
learning_rate: float = 2e-4
warmup_steps: int = 100
weight_decay: float = 0.01
max_grad_norm: float = 1.0

# Data Configuration
max_train_samples: Optional[int] = None
seed: int = 42
use_all_answers: bool = False
weight_sampling: bool = False

# Eval Configuration
gen_train_preds: bool = True
eval_train_samples: int = 100
gen_test_preds: bool = False
```

## One General model
There are no losses with training one model on all the data and for both tasks at the same time. as overfitting on the training data is expected the model actually performs better than the individual models. Also tried to increase the dataset for SAQ by including also the not perfect answers but got practically no improvement(SAQ +1%) as it can also be random noise. but the trainingdata accurracy went down testing this (SAQ 68,15% on training data).

========================================
SAQ EVALUATION RESULTS
========================================
Total Questions Evaluated:  100
Exact Matches (Full Score): 85 -> 53
Total Achieved Score:       302 -> 229
Total Max Score:            333 -> 336
Overall Accuracy:           90.69% -> 68.15%
========================================

========================================
MCQ EVALUATION RESULTS
========================================
Total Questions Evaluated: 100
Correct Predictions:       95
Accuracy:                  95.00%
========================================

testing data results: MCQ: 76% SAQ: 60%

Current overfitting:
Small SAQ:
	MCQ - 25%
	SAQ - 30%
Large SAQ:
	MCQ - 25%
	SAQ - 7%

this shows that the larger dataset for SAQ gives some kind of improvement as the performance is the same but it didnt overfit so much as before

### Current Config
```python
# LoRA Configuration
lora_r: int = 64
lora_alpha: int = 128
lora_dropout: float = 0.1

# Training Configuration
num_epochs: int = 3
batch_size: int = 4
gradient_accumulation_steps: int = 4
learning_rate: float = 2e-4
warmup_steps: int = 100
weight_decay: float = 0.01
max_grad_norm: float = 0.3

# Data Configuration
max_train_samples: Optional[int] = None
seed: int = 42
use_all_answers: bool = False
weight_sampling: bool = False

# Eval Configuration
gen_train_preds: bool = True
eval_train_samples: int = 100
gen_test_preds: bool = False
```

## Improving architecture
first we just cloned the input_ids into the labels for the tokenized dataset. but then added functionality to mask the user input so during training the loss function ignores the user input. so we help the model not recognize or predict the question asked and let it focus on the answer. also added validation split during training. also added artificial mcq data by shuffeling the answer options.
also this model is a general model answering both types of questions.
MCQ: 3344 Questions
SAQ: 6111 Questions

========================================
SAQ EVALUATION RESULTS
========================================
Total Questions Evaluated:  400
Exact Matches (Full Score): 297
Total Achieved Score:       1132
Total Max Score:            1338
Overall Accuracy:           84.60%
========================================

========================================
MCQ EVALUATION RESULTS
========================================
Total Questions Evaluated: 400
Correct Predictions:       400
Accuracy:                  100.00%
========================================
testing data scores:
79%->MCQ 63%->SAQ

### Current Config
```python
# LoRA Configuration
lora_r: int = 32 # Defines the precision of the output Matrix (higher rank = more parameters are trained)
lora_alpha: int = 64 # multiplyer applied to the weight changes when added to the original weights (scale= alpha/r)
lora_dropout: float = 0.1 # is the percentage that randomly leaves out some weight changes each time to deter overfitting

# Training Configuration
num_epochs: int = 3
batch_size: int = 4 # sets how many examples are processed on each GPU/device per forward pass
gradient_accumulation_steps: int = 2 # simulate larger batches by accumulating gradients across multiple steps before updating weights
learning_rate: float = 2e-4 # How large should each eight update be
warmup_steps: int = 100 # gradually increases the learning rate from zero over the first N steps (stabilizes early training)
weight_decay: float = 0.01 # adds L2 regularization to prevent overfitting.
max_grad_norm: float = 0.3 # clips gradients to prevent extreme updates that could destabilize training
safe_steps: int = 100
neftune_noise_alpha: int = 5
val_set_size: float = 0.1 # None to disable testing set out of training data
# Data Configuration
max_train_samples: Optional[int] = None
seed: int = 42
use_all_answers: bool = True
weight_sampling: bool = False

# Eval Configuration
gen_train_preds: bool = True
eval_train_samples: int = 400
gen_test_preds: bool = True
```


# Llama vs Mistral
wasnt able to identify a difference between the two models while testing

saw imporvements when combining the two models to one for both tasks most likely because the datasets arnt that large.