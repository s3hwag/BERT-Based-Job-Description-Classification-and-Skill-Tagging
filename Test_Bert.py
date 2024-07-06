from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder

file_path = '/Users/sehwagvijay/Desktop/resume_data/Resume/Trimmed_Resume.csv'
resume_data = pd.read_csv(file_path)

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(resume_data['Category'])

model_path = '/Users/sehwagvijay/Desktop/resume_data/resume_bert_model'
tokenizer_path = '/Users/sehwagvijay/Desktop/resume_data/resume_bert_tokenizer'

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

new_resume = """
Professional Summary:
Dynamic Human Resources Professional with over 8 years of experience in talent acquisition, employee relations, and performance management. Adept at implementing HR policies and procedures, driving employee engagement initiatives, and fostering a positive workplace culture. Proven track record in recruiting top talent, reducing employee turnover, and enhancing organizational effectiveness.

Professional Experience:
Human Resources Manager
ABC Corporation, City, State
March 2019 – Present

Lead the HR team in developing and implementing HR strategies and initiatives aligned with business goals.
Manage the recruitment and selection process, resulting in a 20% increase in quality hires.
Oversee employee relations, handling conflicts, grievances, and disciplinary actions.
Develop and implement performance management systems, including appraisals and feedback mechanisms.
Conduct training and development programs to enhance employee skills and career growth.
Ensure compliance with labor laws and regulations, maintaining a safe and inclusive workplace.
Senior HR Generalist
XYZ Inc., City, State
June 2015 – February 2019

Supported HR operations, including recruitment, onboarding, and employee engagement activities.
Conducted exit interviews and analyzed turnover data to develop retention strategies.
Assisted in the development and implementation of HR policies and procedures.
Coordinated and facilitated employee training sessions on various HR-related topics.
Collaborated with management to identify staffing needs and develop job descriptions.
Administered employee benefits programs and addressed employee queries and concerns.
HR Coordinator
DEF Company, City, State
January 2012 – May 2015

Assisted in the recruitment process, including job postings, resume screening, and interview scheduling.
Conducted new employee orientations and ensured smooth onboarding processes.
Maintained employee records and HR databases with accuracy and confidentiality.
Supported employee relations efforts by addressing employee inquiries and resolving issues.
Coordinated employee engagement activities, such as team-building events and recognition programs.
Assisted in payroll processing and benefits administration.
Education:
Master of Business Administration (MBA) in Human Resources
University of State, City, State
Graduated: May 2011

Bachelor of Science in Psychology
University of State, City, State
Graduated: May 2009

Skills:
Talent Acquisition and Recruitment
Employee Relations and Conflict Resolution
Performance Management
HR Policies and Procedures
Training and Development
Employee Engagement
Compliance and Labor Laws
HRIS and HR Analytics
Excellent Communication and Interpersonal Skills
Certifications:
Professional in Human Resources (PHR)
SHRM Certified Professional (SHRM-CP)
Talent Acquisition Specialist Certification
Certified Employee Benefits Specialist (CEBS)
"""

predicted_category, probabilities = predict_category(new_resume)
print(f'The predicted job category is: {predicted_category}')
print(f'Probabilities: {probabilities}')