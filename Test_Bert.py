from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import re

file_path = '/Users/sehwagvijay/Desktop/Projects/BERT-Based-Job-Description-Classification-and-Skill-Tagging/resume_data/Trimmed_Resume.csv'
resume_data = pd.read_csv(file_path)

#HR, Designer, Information-Technology, Teacher, Advocate, Business-Development, Healthcare, Fitness, Agriculture, BPO, Sales, Consultant, Digital-Media

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(resume_data['Category'])

model_path = '/Users/sehwagvijay/Desktop/Projects/BERT-Based-Job-Description-Classification-and-Skill-Tagging/resume_bert_model'
tokenizer_path = '/Users/sehwagvijay/Desktop/Projects/BERT-Based-Job-Description-Classification-and-Skill-Tagging/resume_bert_tokenizer'

model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

def tokenize_resume(resume_text):
    return tokenizer.encode_plus(
        resume_text,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

def predict_category(resume_text):
    tokens = tokenize_resume(resume_text)
    input_ids = tokens['input_ids']
    attention_mask = tokens['attention_mask']
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1).item()
        
    category = label_encoder.inverse_transform([preds])[0]
    return category, probs

def extract_skills_from_text(resume_text):
    # Use regex to find the text that comes after "Skills" and before any other section
    match = re.search(r'Skills\s*:\s*(.+?)(?:\n[A-Z]|$)', resume_text, re.IGNORECASE | re.DOTALL)
    if match:
        # Extract the skills section
        skills_text = match.group(1).strip()
        # Split the skills by commas (or other delimiters)
        skills_list = [skill.strip() for skill in skills_text.split(',')]
        return skills_list
    return []

new_resume = """
John Doe
123 Main Street, Anytown, USA
Email: johndoe@email.com | Phone: (555) 555-5555
LinkedIn: linkedin.com/in/johndoe | GitHub: github.com/johndoe

Objective:
Results-oriented software engineer with 5+ years of experience in full-stack development. Adept at developing scalable web applications and leading cross-functional teams in Agile environments. Seeking to leverage my technical and leadership skills in a challenging role at a forward-thinking company.

Experience:

Software Engineer | XYZ Corp
Anytown, USA | January 2018 – Present
- Developed and maintained high-performance web applications using JavaScript, React, and Node.js.
- Implemented RESTful APIs and microservices architecture, improving application scalability by 30%.
- Collaborated with UX/UI designers to enhance user experience, resulting in a 25% increase in user retention.
- Led a team of 5 developers in adopting Agile methodologies, reducing development cycle time by 20%.

Junior Web Developer | ABC Inc.
Anytown, USA | June 2016 – December 2017
- Built and optimized responsive web pages using HTML, CSS, and JavaScript.
- Assisted in the development of backend services using Python and Django.
- Conducted code reviews and provided feedback to junior developers, improving code quality by 15%.
- Worked closely with QA teams to ensure software quality and stability before deployment.

Education:

Bachelor of Science in Computer Science
Anytown University | September 2012 – May 2016

Skills:
Python, JavaScript, React, Node.js, Express, MongoDB, SQL, HTML, CSS, Git, Docker, Kubernetes, AWS, Agile Methodologies

Certifications:
Certified Kubernetes Administrator (CKA) | Linux Foundation | 2019
AWS Certified Solutions Architect | Amazon Web Services | 2018
"""


predicted_category, probabilities = predict_category(new_resume)
print(f'The predicted job category is: {predicted_category}')
print(f'Probabilities: {probabilities}')

extracted_skills = extract_skills_from_text(new_resume)
print("Extracted Skills:", extracted_skills)

