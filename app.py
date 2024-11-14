import joblib
import re
import streamlit as st
from urllib.parse import urlparse

# تحميل النموذج و TF-IDF vectorizer
model = joblib.load('xgb_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# تحميل القائمة البيضاء من ملف
with open('whitelist.txt', 'r') as file:
    whitelist_domains = set(line.strip().lower() for line in file)

def preprocess_url(url):
    """تنظيف بيانات URL قبل التصنيف."""
    url = re.sub(r'http[s]?://', '', url)  # إزالة http أو https
    url = re.sub(r'www\.', '', url)        # إزالة www
    url = re.sub(r'[^A-Za-z0-9.]+', ' ', url)
    return url.lower()  # تحويل إلى أحرف صغيرة

def get_domain(url):
    """استخراج الدومين الأساسي من URL."""
    parsed_url = urlparse(url)
    domain = parsed_url.netloc or parsed_url.path  # اختيار الدومين
    if domain.startswith("www."):
        domain = domain[4:]  # إزالة www إذا كانت موجودة
    return domain.lower()

def is_safe_domain(domain):
    """التحقق مما إذا كان الدومين موجودًا في القائمة البيضاء."""
    return domain in whitelist_domains

def predict_url(url):
    domain = get_domain(url)
    
    # التحقق من القائمة البيضاء
    if is_safe_domain(domain):
        return 'safe'
    
    # تطبيق التحليل باستخدام التعلم الآلي إذا لم يكن ضمن القائمة البيضاء
    cleaned_url = preprocess_url(url)
    features = vectorizer.transform([cleaned_url])
    prediction = model.predict(features)
    return 'malicious' if prediction[0] == 1 else 'benign'

# واجهة المستخدم
st.title("SVU URL CHECKER")
url = st.text_input("Enter a url to classify:")
if st.button("Check URL"):
    if url:
        label = predict_url(url)
        st.write(f'The predicted label for the URL "{url}" is: {label}')
    else:
        st.write("Please enter a URL.")
