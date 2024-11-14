!pip install xgboost
!pip install imblearn
import pandas as pd
import numpy as np
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# تحميل البيانات
data = pd.read_csv('/content/drive/MyDrive/for web/url_data.csv')  # تأكد من مسار الملف

# معالجة النص
def preprocess_url(url):
    url = re.sub(r'http[s]?://', '', url)
    url = re.sub(r'www\.', '', url)
    url = re.sub(r'[^A-Za-z0-9.]+', ' ', url)
    return url.lower()

data['processed_url'] = data['url'].apply(preprocess_url)

# تحويل البيانات إلى ميزات باستخدام TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)  # تحديد 5000 ميزة
X = vectorizer.fit_transform(data['processed_url']).toarray()

# تحويل التصنيفات إلى أرقام
data['label'] = data['label'].map({'benign': 0, 'malicious': 1})
y = data['label']

# تقسيم البيانات
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# التعامل مع عدم توازن البيانات باستخدام SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# تدريب النموذج
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train_resampled, y_train_resampled)

# التقييم
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# حفظ النموذج والـ vectorizer
joblib.dump(model, '/content/drive/MyDrive/for web/xgb_model.pkl')
joblib.dump(vectorizer, '/content/drive/MyDrive/for web/tfidf_vectorizer.pkl')

# رسم مصفوفة الارتباك
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Benign', 'Malicious'], yticklabels=['Benign', 'Malicious'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig("confusion_matrix_final.jpg")
plt.show()