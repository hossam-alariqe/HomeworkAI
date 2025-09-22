#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle

# ---------------------------
# 1. تحميل البيانات
# ---------------------------
df = pd.read_csv(r"C:\Users\ComputerWorld\Desktop\Tweets.csv")

# فقط نحتاج النص والتصنيف
df = df[['airline_sentiment', 'text']].dropna()
df = df.rename(columns={'airline_sentiment': 'sentiment', 'text': 'text'})

# ---------------------------
# 2. تنظيف النصوص
# ---------------------------
def clean_text(s):
    s = str(s)
    s = re.sub(r"http\S+", " ", s)       # إزالة الروابط
    s = re.sub(r"@\w+", " ", s)          # إزالة المنشن
    s = re.sub(r"[^A-Za-z0-9\s]", " ", s) # إزالة الرموز الخاصة
    s = re.sub(r"\s+", " ", s).strip()
    return s.lower()

df['text_clean'] = df['text'].apply(clean_text)

# ---------------------------
# 3. تقسيم البيانات
# ---------------------------
X = df['text_clean'].values
y = df['sentiment'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------
# 4. تحويل النص إلى ميزات (TF-IDF)
# ---------------------------
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ---------------------------
# 5. تدريب Naive Bayes
# ---------------------------
clf = MultinomialNB(alpha=1.0)
clf.fit(X_train_vec, y_train)

# ---------------------------
# 6. التقييم
# ---------------------------
y_pred = clf.predict(X_test_vec)

print("✅ الدقة:", accuracy_score(y_test, y_pred))
print("\n📋 تقرير التصنيف:\n", classification_report(y_test, y_pred))
print("\n📊 مصفوفة الالتباس:\n", confusion_matrix(y_test, y_pred))

# ---------------------------
# 7. اختبار النموذج على جمل جديدة
# ---------------------------
examples = [
    "I love this airline, they were so helpful!",
    "My flight was delayed and the staff was rude.",
    "The flight was okay, nothing special."
]
examples_clean = [clean_text(t) for t in examples]
examples_vec = vectorizer.transform(examples_clean)
preds = clf.predict(examples_vec)

print("\n🔮 تنبؤات أمثلة جديدة:")
for text, p in zip(examples, preds):
    print(f" - '{text}' => {p}")



# In[ ]:




