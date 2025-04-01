from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import os

# Set up your OpenAI API key
os.environ["OPENAI_API_KEY"] = "your-api-key-here"

# Example documents to serve as your knowledge base
documents = [
    "Document text 1 with useful information.",
    "Document text 2 that provides more context.",
    # Add more documents as needed
]

# Create embeddings using OpenAI's embeddings
embeddings = OpenAIEmbeddings()

# Build a vector store (FAISS) from your documents
vectorstore = FAISS.from_texts(documents, embeddings)

# Create a retriever from the vector store
retriever = vectorstore.as_retriever()

# Initialize the OpenAI LLM (e.g., GPT-3.5 or GPT-4)
llm = OpenAI(temperature=0)

# Create the RetrievalQA chain, which combines the retriever with the LLM
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

# Define a function to handle chat queries
def chat(query):
    response = qa_chain.run(query)
    return response

# Example interaction
if __name__ == "__main__":
    user_query = "Can you tell me more about the information in Document text 1?"
    print("User:", user_query)
    print("Bot:", chat(user_query))
