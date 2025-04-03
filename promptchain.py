from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.llms.base import LLM
from transformers import pipeline
from PyPDF2 import PdfReader
from pydantic import PrivateAttr  # Import PrivateAttr



class HuggingFaceLLM(LLM):
    """
    Custom LLM wrapper for Hugging Face models to use with LangChain.
    """
    _summarizer: pipeline = PrivateAttr()  # Mark summarizer as a private attribute. this needs to be done kyuki Pydantic validates all attributes unless they are explicitly marked as private using PrivateAttr()

    def __init__(self, model_name="t5-small"):
        super().__init__()
        self._summarizer = pipeline("summarization", model=model_name)

    def _call(self, prompt: str, stop=None):
        summary = self._summarizer(prompt, max_length=130, min_length=30, do_sample=False) # change max length as needed so if you plan to give shorter prompts then just keep 100 for example 
        return summary[0]["summary_text"]

    @property
    def _identifying_params(self):
        return {"model_name": "t5-small"}

    @property
    def _llm_type(self):
        return "custom_huggingface"


def main():
    # Path to your PDF file # Update with your PDF path here , save it in the project folder and reference it 
    pdf_path = input("Enter the path to your PDF file (e.g., xx\\xxxx\\stock-prices-chatbot\\morningstarreport20250329060631.pdf): ").strip()

    # Extract text from the PDF
    print("Extracting text from PDF...")
    pdf_text = PdfReader(pdf_path)

    # Initialize the custom Hugging Face LLM # I have picked a simple one since running it locally on my own laptop, you can choose to have api key here ..
    llm = HuggingFaceLLM(model_name="t5-small")

    # Define the first prompt: Extract the "Asset Allocation" section , here is where the chain starts for the workflow 
    asset_allocation_prompt = PromptTemplate(
        input_variables=["text"],
        template=(
            "Extract the 'Asset Allocation' section from the following text and provide a percentage breakdown "
            "of different asset classes (e.g., equities, bonds, cash, etc.):\n\n{text}"
        )
    )
    asset_allocation_chain = LLMChain(llm=llm, prompt=asset_allocation_prompt)

    # Define the second prompt: Add a risk notification if stocks exceed XX%, this is the 2nd part of the chain / workflow 
    risk_notification_prompt = PromptTemplate(
        input_variables=["allocation"],
        template=(
            "Given the following asset allocation:\n\n{allocation}\n\n"
            "If the allocation for stocks (or equities) exceeds 70%, add a risk notification: "
            "'The mutual fund asset allocation has high stock allocation and exceed the given threshold % may increase portfolio risk.' The mutual fund asset allocation is within the given threshold % so 'Risk level is acceptable.'"
        )
    )
    risk_notification_chain = LLMChain(llm=llm, prompt=risk_notification_prompt)

    # Combine the chains into a sequential chain , in my example I kept just 2 parts or 2 steps of the workflow , you can have many 
    chain = SimpleSequentialChain(chains=[asset_allocation_chain, risk_notification_chain])

    # Run the chain
    print("Running the chain...")
    result = chain.run(pdf_text)

    # Print the final result
    print("\nAsset Allocation and Risk Notification:")
    print(result)


if __name__ == "__main__":
    main()