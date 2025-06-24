import pandas as pd
import numpy as np
import re
import nltk
import gensim.downloader as api
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Load dataset
df = pd.read_csv("Tweets.csv")[['airline_sentiment', 'text']]

# Preprocessing function
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|@\S+|#\S+", "", text)  # remove URLs, @mentions, hashtags
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # remove punctuation and digits
    text = re.sub(r"\s+", " ", text)  # remove extra spaces
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return tokens

df['tokens'] = df['text'].apply(preprocess_text)

# Load pretrained Word2Vec model
w2v_model = api.load("word2vec-google-news-300")

# Convert tweet to vector by averaging word vectors
def vectorize(tokens):
    vectors = [w2v_model[word] for word in tokens if word in w2v_model]
    return np.mean(vectors, axis=0) if vectors else np.zeros(300)

df['vector'] = df['tokens'].apply(vectorize)

# Drop rows where vectorization failed (all tokens missing from Word2Vec)
df = df[df['vector'].apply(lambda x: x.shape == (300,))]

# Prepare data for training
X = np.vstack(df['vector'].values)
le = LabelEncoder()
y = le.fit_transform(df['airline_sentiment'])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
clf = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Prediction function
def predict_tweet_sentiment(model, w2v_model, tweet):
    tokens = preprocess_text(tweet)
    vec = vectorize(tokens).reshape(1, -1)
    pred = model.predict(vec)
    return le.inverse_transform(pred)[0]

# Test the prediction function
sample = "I had a wonderful flight experience with Delta!"
print("Predicted sentiment:", predict_tweet_sentiment(clf, w2v_model, sample))
