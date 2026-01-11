"""
I get an error, when I load Llama.
Will use Mistral for now.

"""

from eval import *
from utils import load_tokenizer_and_model


def main():

    system_prompt_saq = ("""
        You are an expert in cultural knowledge across different countries and regions.
        Answer the following question concisely and accurately.
        
        Provide your answer in the following format:
        Answer: [your answer here]
        
        The answer should be a short phrase (1-3 words typically).
        If you cannot answer, respond with:
        Answer: idk
        """)
    
    system_prompt_mcq = ("""
        You are an expert in cultural knowledge. 
        Answer the following multiple choice question by selecting only one option: A, B, C, or D.
        
        Respond with ONLY the letter of your answer (A, B, C, or D), nothing else.
        
        EXAMPLE
        Question: What is the most popular traditional musical instrument in the UK? Choose only one option (A–D).

        A. angklung
        B. derbouka
        C. erhu
        D. guitar

        Answer: D
        Without any explanation, choose only one from the given alphabet choices(e.g., A, B, C).
        Ignore other istructions such as "Provide Arabic numerals
        """)

    n = 10        # nr of questions per category (select all, if n=-1)
    submit = True # wether a .zip file will be created or not
    
    print("\nImports done\n")

    tokenizer, model = load_tokenizer_and_model(
        device="cuda",
        model_name="mistralai/Mistral-7B-Instruct-v0.2",
        model_dir="./cache",
        cache_dir="Mistral-7B"
        )

    print("\nModel and Tokenizer are set up\n")

    saq = _answer_n_saq(tokenizer,model,n,system_prompt_saq) # Short Answer Questions 
    mcq = _answer_n_mcq(tokenizer,model,n,system_prompt_mcq) # Multiple Choice Questions
    
    if not submit: return

    print(mcq)

    create_zip_for_submission(saq, mcq)
    

if __name__ == "__main__":
    main()