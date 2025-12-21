Intelligent Document Classification & Insight Extraction System
📋 Project Overview
A lightweight AI-powered system that classifies text documents into four categories (Invoice, Resume, Legal Document, News Article) and extracts meaningful insights based on document type. The system combines traditional machine learning with rule-based NLP techniques to provide accurate classification and relevant information extraction.

Key Features
Multi-class Document Classification: Automatically identifies document type with confidence scores

Type-specific Insight Extraction: Extracts relevant information tailored to each document category

Multiple Interface Options: REST API, CLI tool, and batch processing capabilities

Robust Fallback Mechanisms: Keyword-based matching when ML model confidence is low

🚀 Quick Start
Prerequisites
Python 3.8+

pip package manager

Installation

Clone and setup repository  

bash
git clone https://github.com/Indrajit-Mandol/document-insight-system.git
cd document-classifier
Create virtual environment (recommended)

bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies

bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
Prepare dataset and train model

bash
python data/create_sample_dataset.py  # Creates sample dataset
python src/train_classifier.py       # Trains classification model
🛠️ Usage
Option 1: REST API Server
bash
python src/api/main.py
Server runs at http://localhost:8000

API Endpoints:

GET / - API information

GET /health - Health check

POST /classify - Classify single document

POST /batch-classify - Classify multiple documents

Example API Request:

bash
curl -X POST "http://localhost:8000/classify" \
  -H "Content-Type: application/json" \
  -d '{"text": "INVOICE #123\\nDate: 2024-01-15\\nTotal: $500.00"}'
Option 2: Command Line Interface
bash
# Process text directly
python cli.py --text "Invoice #123..."

# Process from file
python cli.py --file document.txt

# Interactive mode
python cli.py --interactive

# Specify document type (skip classification)
python cli.py --file resume.txt --type "Resume"
Option 3: Python Module
python
from src.train_classifier import DocumentClassifier
from src.insight_extractor import InsightExtractor

classifier = DocumentClassifier.load("models/classifier.pkl")
extractor = InsightExtractor()

# Classify document
doc_type, confidence = classifier.predict(text)

# Extract insights
insights = extractor.extract_insights(text, doc_type)
📊 System Architecture
High-Level Design
text
┌─────────────────────────────────────────┐
│           Input Layer                    │
│  (Text, File, API Request, CLI)         │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│        Document Classifier               │
│  ┌─────────────────────────────────┐    │
│  │ TF-IDF + Structural Features    │    │
│  │ LinearSVC / Random Forest       │    │
│  │ Keyword Fallback                │    │
│  └─────────────────────────────────┘    │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│       Insight Extractor                  │
│  ┌─────────────────────────────────┐    │
│  │ Type-specific Rules             │    │
│  │ spaCy NER                       │    │
│  │ Regex Patterns                  │    │
│  └─────────────────────────────────┘    │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│          Output Formatter                │
│  ┌─────────────────────────────────┐    │
│  │ JSON Response                   │    │
│  │ Pretty CLI Output               │    │
│  │ Structured Insights             │    │
│  └─────────────────────────────────┘    │
└─────────────────────────────────────────┘#

