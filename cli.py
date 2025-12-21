# cli.py
import argparse
import json
import sys
import os
from src.train_classifier import DocumentClassifier
from src.insight_extractor import InsightExtractor

def load_models():
    """Load classification and extraction models"""
    try:
        classifier = DocumentClassifier.load("models/classifier.pkl")
        extractor = InsightExtractor()
        print("✓ Models loaded successfully")
        return classifier, extractor
    except Exception as e:
        print(f"✗ Error loading models: {e}")
        print("Please train the classifier first: python src/train_classifier.py")
        sys.exit(1)

def process_single_document(text, classifier, extractor, doc_type=None):
    """Process a single document"""
    # Classify if type not provided
    if doc_type:
        doc_type_result = doc_type
        confidence = {"user_provided": 1.0}
    else:
        doc_type_result, confidence = classifier.predict(text)
    
    # Extract insights
    insights = extractor.extract_insights(text, doc_type_result)
    
    # Prepare result
    result = {
        "document_type": doc_type_result,
        "confidence": confidence,
        "insights": insights
    }
    
    return result

def process_file(filepath, classifier, extractor, doc_type=None):
    """Process a file containing document text"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        
        return process_single_document(text, classifier, extractor, doc_type)
    except Exception as e:
        return {"error": f"Failed to process file: {str(e)}"}

def main():
    parser = argparse.ArgumentParser(
        description="Document Classification & Insight Extraction CLI"
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("-t", "--text", help="Document text to process")
    input_group.add_argument("-f", "--file", help="Path to document file")
    input_group.add_argument("-i", "--interactive", action="store_true", 
                           help="Interactive mode")
    
    # Optional arguments
    parser.add_argument("-o", "--output", choices=["json", "pretty"], 
                       default="pretty", help="Output format")
    parser.add_argument("-d", "--type", help="Specify document type (skip classification)")
    parser.add_argument("--train", action="store_true", 
                       help="Train classifier before processing")
    
    args = parser.parse_args()
    
    # Train classifier if requested
    if args.train:
        print("Training classifier...")
        os.system("python src/train_classifier.py")
    
    # Load models
    classifier, extractor = load_models()
    
    # Process based on input method
    if args.interactive:
        print("\n=== Interactive Document Processing ===")
        print("Enter document text (Ctrl+D on empty line to finish):")
        
        lines = []
        try:
            while True:
                line = input()
                if line.strip() == ":q":
                    break
                lines.append(line)
        except EOFError:
            pass
        
        text = "\n".join(lines)
        result = process_single_document(text, classifier, extractor, args.type)
    
    elif args.text:
        result = process_single_document(args.text, classifier, extractor, args.type)
    
    elif args.file:
        result = process_file(args.file, classifier, extractor, args.type)
    
    # Output result
    if args.output == "json":
        print(json.dumps(result, indent=2))
    else:
        print("\n" + "="*50)
        print("DOCUMENT ANALYSIS RESULT")
        print("="*50)
        
        if "error" in result:
            print(f"Error: {result['error']}")
            return
        
        print(f"\n📄 Document Type: {result['document_type']}")
        
        print("\n📊 Confidence Scores:")
        for doc_type, score in result['confidence'].items():
            print(f"  {doc_type}: {score:.3f}")
        
        print("\n🔍 Extracted Insights:")
        insights = result['insights']
        
        if not insights:
            print("  No insights extracted")
        else:
            for key, value in insights.items():
                if isinstance(value, list):
                    print(f"  {key}:")
                    for item in value[:3]:  # Show first 3 items
                        print(f"    • {item}")
                elif isinstance(value, dict):
                    print(f"  {key}:")
                    for k, v in value.items():
                        print(f"    {k}: {v}")
                else:
                    print(f"  {key}: {value}")
        
        print("\n" + "="*50)

if __name__ == "__main__":
    main()