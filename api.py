from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl, validator
from typing import List, Optional
import requests
import fitz
import os
import time
from dotenv import load_dotenv
from io import BytesIO
from model import get_answers

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Document QA API",
    description="API for processing PDF documents and answering questions",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

@app.get("/")
async def root():
    return JSONResponse(
        status_code=200,
        content={
            "status": "success",
            "message": "Welcome to the Document QA API",
            "documentation": "/docs",
            "endpoints": {
                "question_answering": "/hackrx/run"
            }
        }
    )

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()
API_KEY = "Ahdrk@865"

# Request and Response Models
class QuestionRequest(BaseModel):
    documents: str  # Changed from HttpUrl to str to accept file paths
    questions: List[str]

    @validator("questions")
    def validate_questions(cls, questions):
        if not questions:
            raise ValueError("At least one question is required")
        if len(questions) > 10:  # Limit number of questions per request
            raise ValueError("Maximum 10 questions allowed per request")
        return questions
    
    @validator("documents")
    def validate_document_source(cls, v):
        # Check if it's a local file
        if os.path.exists(v):
            return v
        # Check if it's a URL
        if v.startswith(("http://", "https://")):
            return v
        raise ValueError("Document must be a valid file path or URL")

    class Config:
        json_schema_extra = {
            "example": {
                "documents": "https://example.com/document.pdf",
                "questions": [
                    "What is the coverage limit?",
                    "What are the exclusions?"
                ]
            }
        }

def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify the API key from the Authorization header"""
    if not credentials:
        raise HTTPException(
            status_code=401,
            detail="Authorization header is missing"
        )
    if credentials.credentials != API_KEY:
        raise HTTPException(
            status_code=401,
            detail={
                "status": "error",
                "message": "Invalid API key",
                "received": credentials.credentials,
                "scheme": credentials.scheme
            }
        )
    return credentials

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Middleware to add processing time to response headers"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

def process_document(document_path: str) -> str:
    """Process PDF from either URL or local file path"""
    doc = None
    try:
        # Handle local file
        if os.path.exists(document_path):
            try:
                doc = fitz.open(document_path)
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Error opening local PDF file: {str(e)}"
                )
        # Handle URL
        elif document_path.startswith(("http://", "https://")):
            try:
                response = requests.get(document_path, timeout=120, 
                                    headers={'User-Agent': 'Document-QA-API/1.0'})
                response.raise_for_status()
                
                # Check content type for URLs
                content_type = response.headers.get('content-type', '')
                if 'application/pdf' not in content_type.lower():
                    raise HTTPException(
                        status_code=400,
                        detail=f"URL does not point to a PDF file. Content-Type: {content_type}"
                    )
                
                pdf_file = BytesIO(response.content)
                doc = fitz.open(stream=pdf_file, filetype="pdf")
            except requests.RequestException as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Error downloading PDF: {str(e)}"
                )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid document source: {document_path}"
            )
        
        if doc.page_count == 0:
            raise HTTPException(
                status_code=400,
                detail="The PDF file is empty"
            )
        
        # Extract text with proper handling of each page
        text = ""
        for page in doc:
            text += page.get_text()
        
        if not text.strip():
            raise HTTPException(
                status_code=400,
                detail="No text content found in the PDF"
            )
        
        return text
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing PDF: {str(e)}"
        )
    finally:
        if doc:
            doc.close()

@app.post("v1/hackrx/run")
async def process_questions(
    request: QuestionRequest,
    credentials: HTTPAuthorizationCredentials = Depends(verify_api_key)
):
    """
    Process questions against a PDF document
    
    Parameters:
    - documents: File path or URL to the PDF file
    - questions: List of questions to answer
    
    Returns:
    - JSON object with answers array
    """
    try:
        # Process the PDF document
        context = process_document(str(request.documents))
        
        # Get answers using the model
        answers = get_answers(context, request.questions)
        
        # Return formatted response
        return JSONResponse(
            status_code=200,
            content={
                "answers": answers
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": "Internal server error",
                "error": str(e)
            }
        )

