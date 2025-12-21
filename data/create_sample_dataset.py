# data/create_sample_dataset.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def create_sample_documents(num_samples=200):
    """Create a balanced dataset of sample documents for all categories"""
    
    documents = []
    labels = []
    
    # 1. Invoice samples
    invoice_templates = [
        "Invoice #{number}\nDate: {date}\nVendor: {vendor}\nTo: {client}\n\nDescription\tQuantity\tUnit Price\tAmount\n{items}\n\nSubtotal: ${subtotal}\nTax: ${tax}\nTotal: ${total}",
        "INVOICE\nInvoice No: INV-{number}\nInvoice Date: {date}\nDue Date: {due_date}\n\nFrom: {vendor}\nTo: {client}\n\nItems:\n{items}\n\nTotal Amount Due: ${total}",
        "TAX INVOICE\nNumber: {number}\nDate: {date}\n\nBill To: {client}\n\n{items}\n\nSubtotal: ${subtotal}\nGST: ${tax}\nTotal: ${total}"
    ]
    
    # 2. Resume samples
    resume_templates = [
        "RESUME\nName: {name}\nEmail: {email}\nPhone: {phone}\n\nSUMMARY\n{summary}\n\nEXPERIENCE\n{experience}\n\nEDUCATION\n{education}\n\nSKILLS\n{skills}",
        "CURRICULUM VITAE\n{name}\n\nProfessional Experience:\n{experience}\n\nEducation:\n{education}\n\nTechnical Skills: {skills}",
        "PROFESSIONAL PROFILE\n{name}\n\nWork History:\n{experience}\n\nQualifications:\n{education}\n\nCore Competencies: {skills}"
    ]
    
    # 3. Legal Document samples
    legal_templates = [
        "AGREEMENT\nThis Agreement is made on {date} between {party1} and {party2}.\n\nTERMS AND CONDITIONS\n{clauses}\n\nIN WITNESS WHEREOF, the parties have executed this Agreement as of the date first above written.",
        "CONTRACT\nContract No: {number}\nEffective Date: {date}\n\nBetween: {party1}\nAnd: {party2}\n\n{clauses}\n\nThis contract shall be governed by the laws of {jurisdiction}.",
        "NON-DISCLOSURE AGREEMENT\nThis NDA is entered into on {date}.\n\nConfidential Information: {definition}\n\nObligations: {obligations}\n\nTerm: This agreement shall remain in effect for {duration}."
    ]
    
    # 4. News Article samples
    news_templates = [
        "{headline}\n\n{location}, {date} - {lead_paragraph}\n\n{body}\n\nAccording to {source}, {conclusion}",
        "BREAKING: {headline}\n\n{body}\n\nThe development comes as {context}.\n\n{source} reports that {details}",
        "{headline}\nBy {author}\n{date}\n\n{body}\n\nIn related news, {related_info}"
    ]
    
    # Generate samples
    vendors = ["ABC Corp", "XYZ Ltd", "Global Supplies", "Tech Solutions Inc"]
    clients = ["Client A", "Client B", "Client C", "Client D"]
    names = ["John Smith", "Emma Johnson", "Michael Chen", "Sarah Williams"]
    skills_list = ["Python, Machine Learning, SQL", "Java, Spring Boot, AWS", "React, Node.js, MongoDB"]
    
    for i in range(num_samples):
        if i < num_samples//4:
            # Invoice
            doc = random.choice(invoice_templates).format(
                number=f"INV{1000+i}",
                date=(datetime.now() - timedelta(days=random.randint(1,30))).strftime("%Y-%m-%d"),
                vendor=random.choice(vendors),
                client=random.choice(clients),
                items="\n".join([f"Item {j+1}\t{random.randint(1,5)}\t${random.uniform(10,100):.2f}\t${random.uniform(10,500):.2f}" 
                                for j in range(random.randint(1,5))]),
                subtotal=f"{random.uniform(100,1000):.2f}",
                tax=f"{random.uniform(10,100):.2f}",
                total=f"{random.uniform(110,1100):.2f}",
                due_date=(datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
            )
            label = "Invoice"
            
        elif i < 2*num_samples//4:
            # Resume
            doc = random.choice(resume_templates).format(
                name=random.choice(names),
                email=f"{random.choice(names).lower().replace(' ', '.')}@email.com",
                phone=f"+1-{random.randint(200,999)}-{random.randint(100,999)}-{random.randint(1000,9999)}",
                summary=f"Experienced professional with {random.randint(2,15)} years in {random.choice(['software development', 'data science', 'product management'])}",
                experience="\n".join([f"{2020-j} - Present: Senior {random.choice(['Developer', 'Analyst', 'Engineer'])} at {random.choice(vendors)}" 
                                    for j in range(random.randint(1,3))]),
                education=f"University of {random.choice(['California', 'Michigan', 'Texas'])} - {random.choice(['BS', 'MS', 'PhD'])} in {random.choice(['Computer Science', 'Engineering', 'Mathematics'])}",
                skills=random.choice(skills_list)
            )
            label = "Resume"
            
        elif i < 3*num_samples//4:
            # Legal Document
            doc = random.choice(legal_templates).format(
                date=(datetime.now() - timedelta(days=random.randint(1,365))).strftime("%Y-%m-%d"),
                party1=random.choice(vendors),
                party2=random.choice(clients),
                clauses="\n".join([f"{j+1}. {random.choice(['Confidentiality', 'Payment Terms', 'Termination', 'Liability'])}: {random.choice(['Shall remain confidential', 'Due within 30 days', 'May be terminated with 30 days notice', 'Limited to contract value'])}" 
                                  for j in range(random.randint(3,6))]),
                number=f"CON{5000+i}",
                jurisdiction=random.choice(["California", "New York", "Texas"]),
                definition=random.choice(["All technical and business information", "Proprietary data and trade secrets"]),
                obligations=random.choice(["Recipient shall not disclose", "Recipient shall protect with reasonable care"]),
                duration=f"{random.randint(1,5)} years"
            )
            label = "Legal Document"
            
        else:
            # News Article
            topics = ["Technology", "Business", "Politics", "Sports"]
            topic = random.choice(topics)
            doc = random.choice(news_templates).format(
    headline=f"{topic} Sector Shows Strong Growth in Q{random.randint(1,4)}",
    location=random.choice(["New York", "London", "Tokyo", "San Francisco"]),
    date=(datetime.now() - timedelta(days=random.randint(1,7))).strftime("%B %d, %Y"),
    lead_paragraph=f"The {topic.lower()} industry reported significant developments this week as major companies announced new initiatives.",
    body="\n\n".join([
        f"Paragraph {j+1}: {random.choice(['Analysts are optimistic', 'Market reaction has been positive', 'Experts caution about challenges', 'This represents a major shift'])} regarding the recent changes in the {topic.lower()} landscape."
        for j in range(random.randint(2,4))
    ]),
    conclusion=random.choice([
        "experts predict continued growth",
        "there may be regulatory challenges ahead"
    ]),
    source=random.choice(["Reuters", "Bloomberg", "Associated Press"]),
    author=random.choice(names),
    related_info=random.choice([
        "similar trends were observed",
        "other sectors also showed improvement"
    ]),
    context=random.choice([
        "markets adjust to new economic data",
        "companies respond to changing consumer demand",
        "regulators review recent developments"
    ]),
    details=random.choice([
        "several firms announced earnings",
        "policy changes are under discussion",
        "investors reacted cautiously"
    ])
)

            label = "News Article"
        
        documents.append(doc)
        labels.append(label)
    
    return pd.DataFrame({
        'text': documents,
        'label': labels,
        'label_id': pd.Categorical(labels).codes
    })

# Create and save dataset
df = create_sample_documents(200)
df.to_csv('data/dataset.csv', index=False)
print(f"Created dataset with {len(df)} samples")
print(df['label'].value_counts())