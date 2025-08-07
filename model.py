import os
import fitz  
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Pinecone as PineconeStore
from langchain.chains import RetrievalQA
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import Pinecone as PineconeStore
from langchain.prompts import PromptTemplate
from typing import List

# API Keys and Configuration
PINECONE_API_KEY = "pcsk_ACBQT_DLycsuJC4V2HXEBuLkwFsrjwNXB9GDNgSsv9Fe58FN1mgG1x8Go4gP3kVK4LTDb"
GOOGLE_API_KEY = "AIzaSyB53RARZM4Y9BpHSOpYgY_SvZs5CxLfPdw"
PINECONE_INDEX_NAME = "quickstart-py"
PINECONE_REGION = "us-east-1"
PINECONE_CLOUD = "aws"
EMBEDDING_DIM = 768

# Initialize environment variables
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

def chunk_text(text, chunk_size=800, chunk_overlap=100):
    """Split text into overlapping chunks"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    return splitter.create_documents([text])

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if it doesn't exist
if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=EMBEDDING_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION)
    )

# Get the index
index = pc.Index(PINECONE_INDEX_NAME)

# Initialize embedding model
embedder = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY
)

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.3)

# Initialize vector store
vector_store = PineconeStore.from_existing_index(
    index_name=PINECONE_INDEX_NAME,
    embedding=embedder
)

# Define prompt template
prompt_template = """You are a helpful medical insurance assistant. Use the following pieces of context to answer the question. If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}

Question: {question}

Answer in this exact format without any asterisks or brackets:
Answer: Write your detailed answer here based on the context
Policy Reference: Add relevant policy sections or page numbers here
Additional Notes: Add any important disclaimers or clarifications here

Answer: """

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# Initialize QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
    return_source_documents=False,
    chain_type_kwargs={
        "prompt": PROMPT,
        "verbose": False
    }
)

def clean_output(text: str) -> str:
    """
    Clean the output text by removing asterisks, brackets, and normalizing newlines.
    
    Args:
        text (str): The text to clean
        
    Returns:
        str: Cleaned text
    """
    # Remove asterisks and brackets
    text = text.replace('*', '').replace('[', '').replace(']', '')
    
    # Normalize newlines (replace multiple newlines with a single newline)
    text = ' '.join(line.strip() for line in text.split('\n') if line.strip())
    
    return text

def get_answers(context: str, questions: List[str]) -> List[str]:
    """
    Process multiple questions against the given context and return answers.
    
    Args:
        context (str): The text content from the PDF
        questions (List[str]): List of questions to answer
        
    Returns:
        List[str]: List of answers in the specified format
    """
    try:
        # Process the context into chunks
        chunks = chunk_text(context)
        texts = [chunk.page_content for chunk in chunks]
        
        # Embed the chunks and update the vector store
        vectors = embedder.embed_documents(texts)
        index.upsert([
            {"id": f"doc-{i}", "values": vector, "metadata": {"text": texts[i]}}
            for i, vector in enumerate(vectors)
        ])
        
        # Process each question and collect answers
        answers = []
        for question in questions:
            try:
                result = qa_chain.invoke({"query": question})
                cleaned_answer = clean_output(result["result"])
                answers.append(cleaned_answer)
            except Exception as e:
                answers.append(f"Error processing question: {str(e)}")
        
        return answers
    except Exception as e:
        raise Exception(f"Error processing questions: {str(e)}")

if __name__ == "__main__":
    # Test the model with sample questions
    test_context = """
    Sample policy text for testing. Replace with actual policy text.
    """
    test_questions = [
        "What is the extent of coverage for AYUSH treatments?",
        "Are there any sub-limits on room rent and ICU charges for Plan A?"
    ]
    
    answers = get_answers(test_context, test_questions)
    for q, a in zip(test_questions, answers):
        print("\n" + "="*80)
        print("\nQuestion:", q)
        print("\nAnswer:", a)
