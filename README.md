
# PDF Analysis with Prompt Chaining

This project demonstrates how to build a chatbot capable of reading, analyzing, and summarizing PDF files. The chatbot leverages the Hugging Face model and LangChain libraries to process and extract meaningful insights from the content of a PDF. It uses **prompt chaining** to perform multiple tasks, such as summarizing specific sections, extracting key information, and providing risk notifications based on the content.

## Features

- Reads and processes PDF files.
- Summarizes specific sections of a PDF (e.g., "Investment Objective," "Asset Allocation").
- Provides percentage breakdowns for asset allocation and generates risk notifications.
- Utilizes Hugging Face models for natural language processing.
- Employs LangChain for chaining prompts and managing conversational workflows.

## Requirements

- Python 3.8 or higher
- Required libraries:
    - `transformers` (Hugging Face)
    - `langchain`
    - `PyPDF2` (for PDF processing)
    - `torch` (for running Hugging Face models)

Install the dependencies using:

```bash
pip install transformers langchain PyPDF2 torch
```

## How It Works

1. **PDF Input**: The user provides the path to the PDF file. For example:
     ```python
     pdf_path = input("Enter the path to your PDF file (e.g., xxxx\\xxx\\stock-prices-chatbot\\morningstarreport20250329060631.pdf): ").strip()
     ```
     This makes the chatbot flexible and capable of processing any PDF file provided by the user.

2. **PDF Processing**: The script reads the content of the specified PDF file using PyPDF2.

3. **Prompt Chaining**: LangChain is used to create a sequence of prompts. Each prompt performs a specific task, such as summarizing a section or analyzing asset allocation.

4. **Hugging Face Model**: The extracted text is passed through a Hugging Face model (e.g., `t5-small`) for summarization and natural language understanding.

5. **Risk Notification**: If certain conditions are met (e.g., stock allocation exceeds 80%), the chatbot generates a risk notification.

## Usage

1. Place the PDF file you want to analyze in the project directory or provide its full path.
2. Run the `promptchain.py` script:
     ```bash
     python promptchain.py
     ```
3. When prompted, enter the path to your PDF file. For example:
     ```
     Enter the path to your PDF file: C:\Users\goure\stock-prices-chatbot\sample.pdf
     ```
4. The chatbot will process the PDF and provide insights based on its content.

## Example Workflow

### Input:
A PDF file containing the following sections:
- **Investment Objective**: "The fund aims to provide long-term capital growth by investing in global equities."
- **Asset Allocation**:
    - Equities: 85%
    - Bonds: 10%
    - Cash: 5%

### Output:
The chatbot performs the following tasks:
1. Summarizes the "Investment Objective" section:
     ```
     Summary: The fund aims to provide long-term capital growth by investing in global equities.
     ```
2. Analyzes the "Asset Allocation" section and provides a risk notification:
     ```
     Asset Allocation:
     - Equities: 85%
     - Bonds: 10%
     - Cash: 5%
     Risk Notification: High risk due to equities allocation exceeding 80%.
     ```

## File Structure

```
/stock-prices-chatbot
│
├── README.md          # Project documentation
├── promptchain.py     # Main script for PDF analysis
├── requirements.txt   # List of dependencies
└── sample.pdf         # Example PDF file for testing
```

## Customization

You can customize the chatbot to handle different types of PDFs or perform additional tasks:

- **Add New Prompts**: Modify the `promptchain.py` script to include new prompts for extracting specific sections or answering custom questions.
- **Change Models**: Replace the Hugging Face model (e.g., `t5-small`) with a different model for better performance or specific use cases.
- **Handle Large PDFs**: Split large PDFs into smaller chunks and process them sequentially.

## Example Commands

To analyze a PDF file named `sample.pdf`:

```bash
python promptchain.py
```
