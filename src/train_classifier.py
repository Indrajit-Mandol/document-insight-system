# src/train_classifier.py

import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings('ignore')

from collections import Counter

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class DocumentClassifier:
    def __init__(self, model_type='svm'):
        # Use both TF-IDF and CountVectorizer for better features
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=2000,
            stop_words='english',
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.9
        )
        
        self.count_vectorizer = CountVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        if model_type == 'svm':
            self.model = LinearSVC(
                C=1.0,
                class_weight='balanced',
                random_state=42,
                max_iter=2000
            )
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=300,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced_subsample'
            )
        else:
            self.model = LinearSVC(class_weight='balanced', random_state=42)
        
        # Enhanced keyword patterns with weights
        self.keyword_patterns = {
            'Invoice': [
                (r'(?i)invoice\s*(?:no\.?|#?)\s*[A-Z0-9\-]+', 2.0),
                (r'(?i)total\s*[\$₹€]?\s*[\d,]+\.?\d*', 1.5),
                (r'(?i)amount\s*due', 1.5),
                (r'(?i)date:\s*\d{4}[-/]\d{1,2}[-/]\d{1,2}', 1.0),
                (r'(?i)bill\s*to', 1.0),
                (r'(?i)vendor|supplier', 1.0)
            ],
            'Resume': [
                (r'(?i)resume|cv|curriculum\s*vitae', 2.0),
                (r'(?i)skills?:', 1.5),
                (r'(?i)experience:', 1.5),
                (r'(?i)education:', 1.5),
                (r'(?i)\d+\s*years?\s*experience', 1.0),
                (r'(?i)phone|email|contact', 1.0),
                (r'(?i)summary|objective', 1.0)
            ],
            'Legal Document': [
                (r'(?i)agreement|contract', 2.0),
                (r'(?i)party\s+[A-Z]', 1.5),
                (r'(?i)shall|hereby', 1.5),
                (r'(?i)terms?\s*and\s*conditions', 1.0),
                (r'(?i)effective\s*date', 1.0),
                (r'(?i)witnesseth', 2.0),
                (r'(?i)jurisdiction', 1.0)
            ],
            'News Article': [
                (r'(?i)breaking\s*news', 2.0),
                (r'(?i)report(?:ed|ing|s)?', 1.5),
                (r'(?i)said|according\s*to', 1.0),
                (r'(?i)headline', 1.5),
                (r'(?i)article|journalist', 1.0),
                (r'(?i)exclusive', 1.0)
            ]
        }
    
    def extract_structural_features(self, text):
        """Extract structural features from text"""
        features = []
        
        # Text length features
        features.append(len(text))
        features.append(len(text.split()))
        features.append(len(text.split('\n')))
        
        # Capitalization features
        upper_ratio = sum(1 for c in text if c.isupper()) / max(1, len(text))
        features.append(upper_ratio)
        
        # Special character features
        features.append(text.count('$') + text.count('₹') + text.count('€'))
        features.append(text.count('#'))
        features.append(text.count(':'))
        
        # Date patterns
        date_patterns = [
            r'\d{4}[-/]\d{1,2}[-/]\d{1,2}',
            r'[A-Z][a-z]+ \d{1,2}, \d{4}',
            r'\d{1,2}/?\d{1,2}/?\d{4}'
        ]
        date_count = sum(len(re.findall(pattern, text)) for pattern in date_patterns)
        features.append(date_count)
        
        # Number patterns
        number_count = len(re.findall(r'\d+', text))
        features.append(number_count)
        
        return features
    
    def preprocess_text(self, text):
        """Clean and preprocess text"""
        # Keep important punctuation and numbers
        text = text.lower()
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def train(self, X_train, y_train):
        print("Preprocessing training data...")
        X_train_processed = [self.preprocess_text(text) for text in X_train]
        
        # Fit vectorizers
        print("Fitting vectorizers...")
        X_train_tfidf = self.tfidf_vectorizer.fit_transform(X_train_processed)
        X_train_count = self.count_vectorizer.fit_transform(X_train_processed)
        
        # Extract structural features
        print("Extracting structural features...")
        struct_features = np.array([self.extract_structural_features(text) for text in X_train])
        
        # Combine features
        from scipy.sparse import hstack
        X_train_combined = hstack([X_train_tfidf, X_train_count, struct_features])
        
        # Check class distribution
        print(f"Original class distribution: {Counter(y_train)}")
        
        # Apply SMOTE if needed
        try:
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train_combined, y_train)
            print(f"After SMOTE: {Counter(y_train_resampled)}")
            X_train_final, y_train_final = X_train_resampled, y_train_resampled
        except:
            print("SMOTE not available, using original data")
            X_train_final, y_train_final = X_train_combined, y_train
        
        # Train model
        print("Training model...")
        self.model.fit(X_train_final, y_train_final)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_final, y_train_final, cv=5)
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean CV accuracy: {cv_scores.mean():.4f}")
        
        return self
    
    def predict_with_keywords(self, text):
        """Predict using keyword patterns as fallback"""
        scores = {}
        
        for doc_type, patterns in self.keyword_patterns.items():
            total_score = 0
            for pattern, weight in patterns:
                matches = re.findall(pattern, text)
                total_score += len(matches) * weight
            
            # Normalize by text length
            word_count = len(text.split())
            if word_count > 0:
                total_score = total_score / (word_count ** 0.5)
            
            scores[doc_type] = total_score
        
        if scores:
            best_type = max(scores.items(), key=lambda x: x[1])[0]
            if scores[best_type] > 0.5:  # Threshold
                return best_type, {best_type: 1.0}
        
        return None, {}
    
    def predict(self, text, threshold=0.4):
        """Predict document type with improved logic"""
        
        # First try keyword-based prediction for very short texts
        if len(text.split()) < 10:
            keyword_pred, keyword_conf = self.predict_with_keywords(text)
            if keyword_pred:
                return keyword_pred, keyword_conf
        
        processed_text = self.preprocess_text(text)
        
        # Transform text
        text_tfidf = self.tfidf_vectorizer.transform([processed_text])
        text_count = self.count_vectorizer.transform([processed_text])
        struct_features = np.array([self.extract_structural_features(text)])
        
        # Combine features
        from scipy.sparse import hstack
        text_combined = hstack([text_tfidf, text_count, struct_features])
        
        # Get prediction
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(text_combined)[0]
            prediction = self.model.predict(text_combined)[0]
            class_labels = self.model.classes_
            
            confidence_scores = {
                class_labels[i]: float(probabilities[i]) 
                for i in range(len(class_labels))
            }
        else:
            # For LinearSVC, use decision function
            decision_scores = self.model.decision_function(text_combined)[0]
            prediction = self.model.predict(text_combined)[0]
            class_labels = self.model.classes_
            
            # Convert decision scores to probabilities
            from scipy.special import softmax
            probabilities = softmax(decision_scores)
            confidence_scores = {
                class_labels[i]: float(probabilities[i]) 
                for i in range(len(class_labels))
            }
        
        max_confidence = max(confidence_scores.values())
        
        # If confidence is low, use keyword fallback
        if max_confidence < threshold:
            keyword_pred, keyword_conf = self.predict_with_keywords(text)
            if keyword_pred:
                return keyword_pred, keyword_conf
        
        return prediction, confidence_scores
    
    def evaluate(self, X_test, y_test):
        predictions = []
        confidences = []
        
        for text in X_test:
            pred, conf = self.predict(text)
            predictions.append(pred)
            confidences.append(conf)
        
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions)
        
        return {
            'accuracy': accuracy,
            'report': report,
            'predictions': predictions,
            'confidences': confidences
        }
    
    def save(self, filepath):
        """Save model to disk"""
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'tfidf_vectorizer': self.tfidf_vectorizer,
                'count_vectorizer': self.count_vectorizer,
                'model': self.model,
                'keyword_patterns': self.keyword_patterns
            }, f)
    
    @classmethod
    def load(cls, filepath):
        """Load model from disk"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        classifier = cls()
        classifier.tfidf_vectorizer = data['tfidf_vectorizer']
        classifier.count_vectorizer = data['count_vectorizer']
        classifier.model = data['model']
        classifier.keyword_patterns = data.get('keyword_patterns', {})
        
        return classifier

def main():
    print("Document Classifier Training")
    print("="*50)
    
    # Check if dataset exists
    dataset_path = 'data/dataset.csv'
    
    if os.path.exists(dataset_path):
        print(f"Loading dataset from {dataset_path}")
        df = pd.read_csv(dataset_path)
    else:
        print("Dataset not found. Please create a dataset first.")
        return
    
    print(f"\nDataset loaded: {len(df)} documents")
    print("Class distribution:")
    print(df['label'].value_counts())
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'], 
        test_size=0.2, 
        random_state=42, 
        stratify=df['label']
    )
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Train classifier
    print("\nTraining Document Classifier with SVM...")
    classifier = DocumentClassifier(model_type='svm')
    classifier.train(X_train, y_train)
    
    # Evaluate
    results = classifier.evaluate(X_test, y_test)
    print(f"\nAccuracy: {results['accuracy']:.4f}")
    print("\nClassification Report:")
    print(results['report'])
    
    # Save model
    os.makedirs('models', exist_ok=True)
    classifier.save('models/classifier.pkl')
    print("\nModel saved to 'models/classifier.pkl'")
    
    # Test with sample documents
    test_docs = [
        "Invoice #1234\nDate: 2024-01-15\nTotal: $500.00",
        "John Smith\nSkills: Python, ML\nExperience: 5 years",
        "This Agreement is made between Party A and Party B",
        "Breaking news: Stock market reaches record high today",
        "Hello world",
        "Invoice from XYZ Corp for services rendered",
        "RESUME\nJohn Doe\nSkills: Java, Spring Boot\nExperience: 3 years",
        "News update: Company announces new product launch"
    ]
    
    print("\n" + "="*50)
    print("Sample Predictions:")
    print("="*50)
    
    for i, doc in enumerate(test_docs, 1):
        pred, conf = classifier.predict(doc)
        print(f"\nTest {i}:")
        print(f"Document: {doc[:60]}...")
        print(f"Prediction: {pred}")
        print(f"Confidence: {conf}")
        print("-" * 40)

if __name__ == "__main__":
    main()