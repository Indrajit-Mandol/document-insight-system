# src/api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train_classifier import DocumentClassifier
from src.insight_extractor import InsightExtractor
import pickle

app = FastAPI(
    title="Document Classification & Insight Extraction API",
    description="API for classifying documents and extracting insights",
    version="1.0.0"
)

# Load models at startup
try:
    classifier = DocumentClassifier.load("models/classifier.pkl")
    extractor = InsightExtractor()
    print("Models loaded successfully")
except Exception as e:
    print(f"Error loading models: {e}")
    classifier = None
    extractor = None

class DocumentRequest(BaseModel):
    text: str
    document_type: Optional[str] = None  # Optional: if known, skip classification

class DocumentResponse(BaseModel):
    document_type: str
    confidence: Dict[str, float]
    insights: Dict[str, Any]
    processing_time: float

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Document Classification & Insight Extraction API",
        "endpoints": {
            "/classify": "POST - Classify document and extract insights",
            "/health": "GET - API health check",
            "/docs": "API documentation"
        },
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "classifier_loaded": classifier is not None,
        "extractor_loaded": extractor is not None
    }

@app.post("/classify", response_model=DocumentResponse)
async def classify_document(request: DocumentRequest):
    """
    Classify a document and extract insights
    
    - **text**: The document text to classify
    - **document_type**: (Optional) If document type is already known
    """
    import time
    start_time = time.time()
    
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Document text cannot be empty")
    
    try:
        # Step 1: Classify document if type not provided
        if request.document_type:
            doc_type = request.document_type
            confidence = {"user_provided": 1.0}
        else:
            if classifier is None:
                raise HTTPException(status_code=503, detail="Classifier not available")
            
            doc_type, confidence = classifier.predict(request.text)
        
        # Step 2: Extract insights based on document type
        if extractor is None:
            raise HTTPException(status_code=503, detail="Insight extractor not available")
        
        insights = extractor.extract_insights(request.text, doc_type)
        
        # Step 3: Calculate processing time
        processing_time = time.time() - start_time
        
        # Step 4: Prepare response
        response = {
            "document_type": doc_type,
            "confidence": confidence,
            "insights": insights,
            "processing_time": round(processing_time, 4)
        }
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/batch-classify")
async def batch_classify(documents: list[DocumentRequest]):
    """
    Classify multiple documents in batch
    
    Returns classification results for all documents
    """
    results = []
    
    for i, doc_request in enumerate(documents):
        try:
            response = await classify_document(doc_request)
            results.append(response)
        except Exception as e:
            results.append({
                "error": str(e),
                "index": i,
                "document_preview": doc_request.text[:100] + "..." if len(doc_request.text) > 100 else doc_request.text
            })
    
    return {
        "total_documents": len(documents),
        "successful": len([r for r in results if "error" not in r]),
        "failed": len([r for r in results if "error" in r]),
        "results": results
    }

# CLI endpoint for testing
@app.post("/cli-test")
async def cli_test(text: str):
    """Simple endpoint for CLI testing"""
    request = DocumentRequest(text=text)
    return await classify_document(request)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)