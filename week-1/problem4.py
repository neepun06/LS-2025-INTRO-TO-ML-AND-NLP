import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Create dataset
positive_reviews = [
    "Absolutely fantastic movie!",
    "Brilliant acting and great story.",
    "I really enjoyed every moment.",
    "The cinematography was stunning.",
    "A must-watch film with heart."
] * 10

negative_reviews = [
    "Terribly boring and slow.",
    "Poor acting and no story.",
    "A complete waste of time.",
    "I didn’t enjoy it at all.",
    "Awful film with bad dialogue."
] * 10

reviews = positive_reviews + negative_reviews
sentiments = ["positive"] * 50 + ["negative"] * 50
df = pd.DataFrame({'Review': reviews, 'Sentiment': sentiments})

# Vectorize text
vectorizer = CountVectorizer(max_features=500, stop_words='english')
X = vectorizer.fit_transform(df['Review'])
y = df['Sentiment']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Prediction function
def predict_review_sentiment(model, vectorizer, review):
    vec = vectorizer.transform([review])
    return model.predict(vec)[0]

# Example
print("Example prediction:", predict_review_sentiment(model, vectorizer, "What a fantastic film!"))
