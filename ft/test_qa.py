import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

# Load the news data
with open('altibase_news_corpus.jsonl', 'r') as f:
    news_data = [line for line in f]

context = " ".join(news_data)

# Load the tokenizer and model
model_name = "distilbert-base-cased-distilled-squad"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Create a question-answering pipeline
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

# Ask a question
question = "What is the latest version of ALTIBASE?"
result = qa_pipeline(question=question, context=context)

# Print the answer
print(f"Question: {question}")
print(f"Answer: {result['answer']}")