import pandas as pd
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 1. Load data
print("Loading data...")
df = pd.read_csv("mbti_1.csv")

# 2. Clean text
def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower().strip()
    return text

print("Cleaning text...")
df['cleaned'] = df['posts'].apply(clean_text)

# 3. Create 4 labels (one per dimension)
df['IE'] = df['type'].apply(lambda x: 1 if x[0] == 'I' else 0)
df['NS'] = df['type'].apply(lambda x: 1 if x[1] == 'N' else 0)
df['TF'] = df['type'].apply(lambda x: 1 if x[2] == 'T' else 0)
df['JP'] = df['type'].apply(lambda x: 1 if x[3] == 'J' else 0)

# 4. TF-IDF
print("Extracting features...")
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words='english')
X = tfidf.fit_transform(df['cleaned'])

# 5. Train one model per dimension
models = {}
dimensions = ['IE', 'NS', 'TF', 'JP']

for dim in dimensions:
    print(f"Training model for {dim}...")
    y = df[dim]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(f"{dim} Report:\n", classification_report(y_test, preds))
    models[dim] = model

# 6. Save everything
print("Saving model and vectorizer...")
pickle.dump(models, open("models.pkl", "wb"))
pickle.dump(tfidf, open("tfidf.pkl", "wb"))

print("Done! models.pkl and tfidf.pkl saved.")