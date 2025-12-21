# src/insight_extractor.py

import re
import spacy
from datetime import datetime
import json
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    print("Downloading spaCy model...")
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

class InsightExtractor:
    def __init__(self):
        # Improved invoice patterns
        self.invoice_patterns = [
            # Invoice number patterns
            (r'(?i)(?:invoice|bill|inv)[\s#:]*([A-Z0-9][A-Z0-9\-]+[A-Z0-9])', 'invoice_number'),
            (r'(?i)no\.?\s*([A-Z0-9\-]+)', 'invoice_number'),
            
            # Date patterns
            (r'(?i)date[\s:]*(\d{4}[-/]\d{1,2}[-/]\d{1,2}|\w+ \d{1,2}, \d{4})', 'date'),
            (r'(?i)invoice date[\s:]*([^\n]+)', 'invoice_date'),
            (r'(?i)due date[\s:]*([^\n]+)', 'due_date'),
            
            # Amount patterns
            (r'(?i)total[\s:]*[\$₹€]?\s*([\d,]+\.?\d+)', 'total_amount'),
            (r'(?i)amount due[\s:]*[\$₹€]?\s*([\d,]+\.?\d+)', 'total_amount'),
            (r'(?i)balance due[\s:]*[\$₹€]?\s*([\d,]+\.?\d+)', 'total_amount'),
            
            # Vendor/Customer patterns
            (r'(?i)from[\s:]*([^\n]+(?=\n|$))', 'vendor'),
            (r'(?i)vendor[\s:]*([^\n]+(?=\n|$))', 'vendor'),
            (r'(?i)supplier[\s:]*([^\n]+(?=\n|$))', 'vendor'),
            (r'(?i)to[\s:]*([^\n]+(?=\n|$))', 'customer'),
            (r'(?i)bill to[\s:]*([^\n]+(?=\n|$))', 'customer'),
            (r'(?i)customer[\s:]*([^\n]+(?=\n|$))', 'customer'),
            
            # Currency
            (r'[\$₹€](?=\s*\d)', 'currency')
        ]
        
        # Improved resume patterns
        self.resume_patterns = [
            # Name (first non-empty line that looks like a name)
            (r'^[A-Z][a-z]+\s+[A-Z][a-z]+$', 'name', re.MULTILINE),
            
            # Contact info
            (r'[\w\.-]+@[\w\.-]+\.\w+', 'email'),
            (r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]', 'phone'),
            (r'(?i)linkedin\.com/[^\s]+', 'linkedin'),
            (r'(?i)github\.com/[^\s]+', 'github'),
            
            # Sections
            (r'(?i)skills?[\s:]+([^\n]+(?:\n[^\n]+){0,2})', 'skills'),
            (r'(?i)experience[\s:]+([^\n]+(?:\n[^\n]+){0,5})', 'experience'),
            (r'(?i)education[\s:]+([^\n]+(?:\n[^\n]+){0,5})', 'education'),
            (r'(?i)summary[\s:]+([^\n]+(?:\n[^\n]+){0,3})', 'summary'),
            (r'(?i)objective[\s:]+([^\n]+(?:\n[^\n]+){0,3})', 'summary'),
            
            # Experience years
            (r'(\d+)\+?\s*(?:years?|yrs)\s*(?:of\s+)?experience', 'years_experience')
        ]
        
        self.skill_keywords = [
            'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'go', 'rust', 'php', 'ruby',
            'react', 'angular', 'vue', 'node.js', 'express', 'django', 'flask', 'spring', 'laravel',
            'sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'oracle', 'sql server',
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform', 'ansible', 'jenkins',
            'machine learning', 'deep learning', 'ai', 'data science', 'tensorflow', 'pytorch',
            'tableau', 'power bi', 'excel', 'spark', 'hadoop', 'kafka',
            'git', 'github', 'gitlab', 'jira', 'agile', 'scrum'
        ]
    
    def extract_with_patterns(self, text: str, patterns: List[Tuple]) -> Dict[str, Any]:
        """Extract information using multiple patterns"""
        insights = {}
        
        for pattern_info in patterns:
            if len(pattern_info) == 2:
                pattern, key = pattern_info
                flags = 0
            else:
                pattern, key, flags = pattern_info
            
            try:
                matches = re.findall(pattern, text, flags)
                if matches:
                    if key in insights:
                        # Keep the best match (longest for most keys)
                        if key in ['skills', 'experience', 'education']:
                            current = insights[key]
                            if isinstance(current, str) and len(matches[0]) > len(current):
                                insights[key] = matches[0]
                            elif isinstance(current, list):
                                insights[key].extend(matches)
                        elif isinstance(matches[0], str) and len(matches[0]) > 2:
                            # For single values, take the first good match
                            if key not in insights or len(matches[0]) > len(str(insights[key])):
                                insights[key] = matches[0]
                    else:
                        if key in ['skills', 'experience', 'education']:
                            insights[key] = matches
                        else:
                            insights[key] = matches[0] if matches else None
            except Exception as e:
                continue
        
        return insights
    
    def extract_from_invoice(self, text: str) -> Dict[str, Any]:
        """Extract insights from invoice documents"""
        insights = {}
        
        # Extract using patterns
        pattern_insights = self.extract_with_patterns(text, self.invoice_patterns)
        insights.update(pattern_insights)
        
        # Clean up extracted values
        if 'customer' in insights and isinstance(insights['customer'], str):
            # Remove trailing punctuation and numbers
            customer = insights['customer'].strip()
            customer = re.sub(r'[\d\.:]+\s*$', '', customer)
            insights['customer'] = customer
        
        if 'vendor' in insights and isinstance(insights['vendor'], str):
            vendor = insights['vendor'].strip()
            vendor = re.sub(r'[\d\.:]+\s*$', '', vendor)
            insights['vendor'] = vendor
        
        # Use spaCy for additional extraction
        doc = nlp(text)
        
        # Extract monetary values
        money_values = []
        for ent in doc.ents:
            if ent.label_ == 'MONEY':
                money_values.append(ent.text)
        
        if money_values:
            insights['money_entities'] = list(set(money_values))[:5]
        
        # Extract dates
        dates = []
        for ent in doc.ents:
            if ent.label_ == 'DATE':
                dates.append(ent.text)
        
        if dates:
            insights['all_dates'] = dates[:5]
        
        # Extract organizations
        organizations = []
        for ent in doc.ents:
            if ent.label_ == 'ORG':
                organizations.append(ent.text)
        
        if organizations:
            insights['organizations'] = list(set(organizations))[:3]
        
        # Extract line items (simple approach)
        line_pattern = r'([A-Za-z\s\-]+)\s+[\$₹€]?\s*([\d,]+\.?\d{2})'
        line_matches = re.findall(line_pattern, text)
        if line_matches:
            line_items = []
            for desc, amount in line_matches:
                if len(desc.strip()) > 2 and float(amount.replace(',', '')) > 0:
                    line_items.append(f"{desc.strip()}: ${amount}")
            
            if line_items:
                insights['line_items'] = line_items[:5]
        
        # Extract tax information
        tax_pattern = r'(?i)(?:tax|gst|vat)[\s:]*[\$₹€]?\s*([\d,]+\.?\d{2})'
        tax_matches = re.findall(tax_pattern, text)
        if tax_matches:
            insights['tax_amounts'] = tax_matches[:3]
        
        return insights
    
    def extract_from_resume(self, text: str) -> Dict[str, Any]:
        """Extract insights from resume documents"""
        insights = {}
        
        # Extract using patterns
        pattern_insights = self.extract_with_patterns(text, self.resume_patterns)
        insights.update(pattern_insights)
        
        # Special handling for name (first line that looks like a name)
        if 'name' not in insights:
            lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
            for line in lines[:3]:  # Check first 3 lines
                if re.match(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+$', line):
                    insights['name'] = line
                    break
        
        # Process skills
        if 'skills' in insights:
            if isinstance(insights['skills'], str):
                # Split by commas, semicolons, or newlines
                skills_text = insights['skills']
                skills_list = re.split(r'[,;\n]', skills_text)
                skills_list = [s.strip() for s in skills_list if s.strip()]
                insights['skills'] = skills_list[:15]
        
        # Use spaCy for additional extraction
        doc = nlp(text)
        
        # Extract technical skills from keywords
        text_lower = text.lower()
        found_skills = []
        for skill in self.skill_keywords:
            if skill in text_lower:
                found_skills.append(skill.title())
        
        if found_skills:
            insights['technical_skills'] = list(set(found_skills))[:10]
        
        # Extract education institutions and degrees
        education_info = []
        for sent in doc.sents:
            sent_text = sent.text.lower()
            education_keywords = ['university', 'college', 'institute', 'bachelor', 'master', 'phd', 'degree']
            if any(keyword in sent_text for keyword in education_keywords):
                education_info.append(sent.text.strip())
        
        if education_info:
            insights['education_details'] = education_info[:3]
        
        # Extract job titles/companies
        job_patterns = [
            r'(?i)(?:senior|junior|lead)?\s*(?:software|data|devops|frontend|backend)\s*(?:engineer|developer|analyst|architect)',
            r'(?i)(?:product|project|technical)\s*manager',
            r'(?i)cto|ceo|director|head of'
        ]
        
        possible_positions = []
        for pattern in job_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            possible_positions.extend(matches)
        
        if possible_positions:
            insights['possible_positions'] = list(set(possible_positions))[:5]
        
        # Extract company names (simple approach)
        company_keywords = ['inc', 'llc', 'ltd', 'corp', 'company', 'technologies']
        companies = []
        for token in doc:
            if token.text.isupper() and len(token.text) > 2:
                # Check if it's likely a company name
                companies.append(token.text)
        
        if companies:
            insights['companies_mentioned'] = list(set(companies))[:3]
        
        return insights
    
    def extract_from_legal(self, text: str) -> Dict[str, Any]:
        """Extract insights from legal documents"""
        insights = {}
        
        # Document number
        doc_num_patterns = [
            r'(?i)(?:agreement|contract|document)\s*(?:no\.?|#)\s*([A-Z0-9\-]+)',
            r'(?i)ref(?:erence)?\s*(?:no\.?|#)\s*([A-Z0-9\-]+)'
        ]
        
        for pattern in doc_num_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                insights['document_number'] = match.group(1)
                break
        
        # Parties
        party_pattern = r'(?i)(?:between|parties)[\s:]*([^\n]+(?:\n[^\n]+){0,3})'
        party_match = re.search(party_pattern, text, re.IGNORECASE)
        if party_match:
            party_text = party_match.group(1)
            # Clean and split parties
            parties = re.split(r'\s+and\s+|\n|,', party_text)
            parties = [p.strip() for p in parties if len(p.strip()) > 3]
            if parties:
                insights['parties'] = parties[:4]
        
        # Dates
        date_pattern = r'(?i)(?:effective|date|signed|executed)[\s:]*([A-Za-z0-9\-\/,\.\s]+)(?=\n|$)'
        dates = re.findall(date_pattern, text, re.IGNORECASE)
        if dates:
            insights['key_dates'] = [d.strip() for d in dates[:3]]
        
        # Use spaCy for clause extraction
        doc = nlp(text)
        
        # Extract key clauses
        clause_keywords = ['shall', 'must', 'will', 'agree to', 'obligated', 'responsible', 'warrant']
        key_clauses = []
        
        for sent in doc.sents:
            sent_text = sent.text.lower()
            if any(keyword in sent_text for keyword in clause_keywords):
                if 10 < len(sent.text.split()) < 50:  # Reasonable length
                    key_clauses.append(sent.text.strip())
        
        if key_clauses:
            insights['key_clauses'] = key_clauses[:5]
        
        # Extract monetary terms
        money_values = []
        for ent in doc.ents:
            if ent.label_ == 'MONEY':
                money_values.append(ent.text)
        
        if money_values:
            insights['monetary_terms'] = list(set(money_values))[:3]
        
        # Extract duration/term
        term_patterns = [
            r'(?i)term[\s:]*(\d+\s*(?:months?|years?|days?))',
            r'(?i)duration[\s:]*(\d+\s*(?:months?|years?|days?))',
            r'(?i)for\s+a\s+period\s+of\s+(\d+\s*(?:months?|years?|days?))'
        ]
        
        for pattern in term_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                insights['term'] = match.group(1)
                break
        
        # Extract jurisdiction
        jurisdiction_pattern = r'(?i)governed\s+by.*laws?\s+of\s+([A-Za-z\s,]+)(?=\n|\.|$)'
        jurisdiction_match = re.search(jurisdiction_pattern, text, re.IGNORECASE)
        if jurisdiction_match:
            insights['jurisdiction'] = jurisdiction_match.group(1).strip()
        
        return insights
    
    def extract_from_news(self, text: str) -> Dict[str, Any]:
        """Extract insights from news articles"""
        insights = {}
        
        # Extract headline
        lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
        if lines:
            insights['headline'] = lines[0][:200]
        
        # Extract date
        date_patterns = [
            r'([A-Z][a-z]+ \d{1,2}, \d{4})',
            r'(\d{4}[-/]\d{1,2}[-/]\d{1,2})',
            r'(\d{1,2}/\d{1,2}/\d{4})'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text)
            if match:
                insights['date'] = match.group(1)
                break
        
        # Use spaCy for extraction
        doc = nlp(text)
        
        # Extract named entities
        entity_types = {}
        for ent in doc.ents:
            if ent.label_ not in entity_types:
                entity_types[ent.label_] = set()
            entity_types[ent.label_].add(ent.text)
        
        # Keep important entity types
        important_entities = ['PERSON', 'ORG', 'GPE', 'LOC', 'DATE', 'EVENT']
        for label in important_entities:
            if label in entity_types:
                insights[f'{label.lower()}_entities'] = list(entity_types[label])[:5]
        
        # Generate summary (first 2-3 sentences)
        sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 10]
        if len(sentences) > 2:
            insights['summary'] = ' '.join(sentences[:3])
        elif sentences:
            insights['summary'] = ' '.join(sentences)
        
        # Simple sentiment analysis
        positive_words = [
            'good', 'great', 'excellent', 'positive', 'success', 'growth',
            'profit', 'gain', 'improve', 'strong', 'win', 'achievement',
            'progress', 'breakthrough', 'innovation', 'record'
        ]
        
        negative_words = [
            'bad', 'poor', 'negative', 'failure', 'decline', 'crisis',
            'loss', 'drop', 'problem', 'issue', 'concern', 'worry',
            'controversy', 'scandal', 'allegation', 'investigation'
        ]
        
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count > neg_count:
            insights['sentiment'] = 'positive'
        elif neg_count > pos_count:
            insights['sentiment'] = 'negative'
        else:
            insights['sentiment'] = 'neutral'
        
        insights['sentiment_score'] = {
            'positive_words': pos_count,
            'negative_words': neg_count,
            'total_words': len(text_lower.split())
        }
        
        # Extract topics
        topic_categories = {
            'technology': ['tech', 'software', 'digital', 'ai', 'computer', 'internet', 'app', 'startup'],
            'business': ['business', 'market', 'company', 'economic', 'financial', 'stock', 'profit', 'revenue'],
            'sports': ['sports', 'game', 'team', 'player', 'championship', 'tournament', 'score', 'win'],
            'politics': ['government', 'political', 'election', 'policy', 'minister', 'president', 'law', 'bill'],
            'health': ['health', 'medical', 'doctor', 'hospital', 'disease', 'treatment', 'medicine', 'vaccine'],
            'entertainment': ['movie', 'film', 'celebrity', 'music', 'actor', 'award', 'show', 'entertainment']
        }
        
        detected_topics = []
        for topic, keywords in topic_categories.items():
            keyword_count = sum(1 for keyword in keywords if keyword in text_lower)
            if keyword_count >= 2:
                detected_topics.append(topic)
        
        if detected_topics:
            insights['topics'] = detected_topics
        
        # Extract key phrases (noun phrases)
        key_phrases = []
        for chunk in doc.noun_chunks:
            phrase = chunk.text.strip()
            if 2 <= len(phrase.split()) <= 5 and phrase.lower() not in key_phrases:
                key_phrases.append(phrase)
        
        if key_phrases:
            insights['key_phrases'] = key_phrases[:8]
        
        return insights
    
    def extract_insights(self, text: str, doc_type: str) -> Dict[str, Any]:
        """Main method to extract insights based on document type"""
        
        extraction_methods = {
            'Invoice': self.extract_from_invoice,
            'Resume': self.extract_from_resume,
            'Legal Document': self.extract_from_legal,
            'News Article': self.extract_from_news
        }
        
        if doc_type in extraction_methods:
            try:
                return extraction_methods[doc_type](text)
            except Exception as e:
                print(f"Error extracting insights for {doc_type}: {e}")
                # Fallback to basic extraction
        
        # Basic fallback extraction
        try:
            doc = nlp(text[:5000])  # Limit text length for speed
            
            entities = {}
            for ent in doc.ents:
                if ent.label_ not in entities:
                    entities[ent.label_] = []
                if ent.text not in entities[ent.label_]:
                    entities[ent.label_].append(ent.text)
            
            return {
                'entities': {k: v[:3] for k, v in entities.items() if v},
                'text_length': len(text),
                'word_count': len(text.split()),
                'sentence_count': len(list(doc.sents))
            }
        except:
            return {
                'error': 'Failed to extract insights',
                'text_preview': text[:100] + '...' if len(text) > 100 else text
            }

def main():
    """Test the insight extractor"""
    extractor = InsightExtractor()
    
    # Test documents
    test_invoice = """INVOICE #INV-2024-001
Date: 2024-01-15
Due Date: 2024-02-15

From: Tech Solutions Inc.
To: ABC Corporation

Description: Software License
Amount: $1,500.00

Total Due: $1,500.00"""
    
    test_resume = """John Smith
Email: john.smith@email.com
Phone: +1-555-123-4567

Summary: Software engineer with 5 years of experience in Python and machine learning.

Skills: Python, Machine Learning, SQL, AWS, Docker

Education: MS in Computer Science, Stanford University"""
    
    # Test extraction
    print("Testing Invoice Extraction:")
    invoice_insights = extractor.extract_from_invoice(test_invoice)
    print(json.dumps(invoice_insights, indent=2))
    
    print("\nTesting Resume Extraction:")
    resume_insights = extractor.extract_from_resume(test_resume)
    print(json.dumps(resume_insights, indent=2))

if __name__ == "__main__":
    main()