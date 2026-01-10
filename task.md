
**Project goal:** Develop an LLM-based approach to cultural question answering

**You can do:** model finetuning, prompt tuning, implement agentic frameworks, self-consistency, retrieval-augmentation, self-RAG, …

**Common sense fairness:**

- Do not game the evaluation with unreasonable amouts of configurations submitted
- Do not train on the test data

The benchmark (https://www.codabench.org/competitions/11605/) is designed to assess the accuracy of the model you build and to compare your results with those of your peers.  

The training data contains English-language questions originating from four different cultural contexts.

There are two tasks included in the benchmark:

1. **Short Answer Questions (SAQ)**
2. **Multiple Choice Questions (MCQ)**

We begin with the simpler **MCQ task**. You are presented with a question and four possible answers, labeled **A** through **D**. The model must interpret cultural and linguistic cues to select the most appropriate option.

**Example (MCQ):**  

_What is the most popular traditional musical instrument in the UK? Choose only one option (A–D)._  

- A. angklung  
- B. derbouka  
- C. erhu  
- D. guitar    
    

**Correct answer:** D

Next is the **SAQ task**, which extends the MCQ task. Instead of selecting from predefined options, your model must generate the answer directly.

**Example (SAQ):**  

_On which holiday do all family members tend to reunite in the US?_  

Acceptable answers:  

- thanksgiving  
- christmas  

## Evaluation

Model performance for the **MCQ task** is evaluated using **accuracy**, computed through a one-to-one comparison between your predicted answer and the answer key. Since each question has exactly one correct option, accuracy reflects the proportion of correct predictions.

For the **SAQ task**, accuracy is calculated based on human-annotated answers. A model-generated response is considered correct if it matches any of the provided annotations. Note that this does **not** account for synonyms or paraphrasing, so keep this limitation in mind.

Most importantly, **have fun exploring and experimenting with your model!**

# How to Submit:

## Submission

You may submit solutions for both tasks or only one. **Make sure to name the submission files correctly**, as the scoring program depends on the filenames. If either solution file (MCQ or SAQ) is missing, the resulting displayed accuracy will be **0.0**. In the **Submission** tab, only the MCQ task score is shown. To view scores for both tasks, click the detailed submission icon (eye symbol).

### MCQ Task

Create a `.tsv` file named **mcq_prediction.tsv** containing the predictions of your model for **MCQ task**.  

The file should follow this format:

```tsv

MCQID A B C D

Kik-in-31_149    False    False    False    True

New-am-54_914    False    True     False    False

Kik-in-10_4988   True     False    False    False

Na-ko-45_549     False    True     False    False

New-en-41_1802   False    False    True     False

```

### SAQ Task

Create a `.tsv` file named **saq_prediction.tsv** containing the predictions of your model for **SAQ task**.

The file should follow this format:

```tsv

ID answer

Kik-in-31_149   Guitar

New-am-54_914   Chips

Kik-in-10_4988   Christmas

```