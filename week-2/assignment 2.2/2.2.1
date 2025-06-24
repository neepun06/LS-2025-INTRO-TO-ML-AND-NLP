import pandas as pd
import numpy as np
import string
import re
import gensim.downloader as api
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load dataset
df = pd.read_csv("spam.csv", encoding='latin-1')[["v1", "v2"]]
df.columns = ["label", "text"]

# Preprocessing
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    tokens = word_tokenize(text)
    return [word for word in tokens if word not in stop_words]

df["tokens"] = df["text"].apply(clean_text)

# Load pre-trained Word2Vec model
print("Loading Word2Vec model (Google News)...")
w2v_model = api.load("word2vec-google-news-300")
print("Model loaded.")

# Function to get average vector
def get_average_vector(tokens, model):
    valid_vectors = [model[word] for word in tokens if word in model]
    if valid_vectors:
        return np.mean(valid_vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

# Vectorize messages
df["vector"] = df["tokens"].apply(lambda x: get_average_vector(x, w2v_model))
X = np.vstack(df["vector"].values)
y = df["label"].apply(lambda x: 1 if x == "spam" else 0)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred))

# Predict function
def predict_message_class(model, w2v_model, message):
    tokens = clean_text(message)
    vector = get_average_vector(tokens, w2v_model).reshape(1, -1)
    prediction = model.predict(vector)[0]
    return "spam" if prediction == 1 else "ham"

# Test the function
test_msg = "Congratulations! You have won a free iPhone. Call now!"
print("Prediction:", predict_message_class(clf, w2v_model, test_msg))
