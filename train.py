import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib

file_name = "files.xlsx"

df = pd.read_excel(file_name)
print(df)

if 'A1' in df.columns and 'A2' in df.columns:
    X = df['A1']
    y = df['A2']
else:
    print(f"Warning: expected columns 'A1' and 'A2' not found. Found columns: {list(df.columns)}")
    if df.shape[1] < 2:
        raise ValueError(f"Expected at least 2 columns in {file_name}, got {df.shape[1]}")
    X = df.iloc[:, 0]
    y = df.iloc[:, 1]

model = Pipeline([('tfidf', TfidfVectorizer()), ('classifier', MultinomialNB())])
model.fit(X.astype(str), y.astype(str))
joblib.dump(model, 'model.pkl')
print("model saved")

