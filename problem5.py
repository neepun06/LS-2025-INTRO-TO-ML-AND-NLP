import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Create dataset
good_feedback = ["This product is excellent!"] * 50
bad_feedback = ["Terrible experience with this item."] * 50
feedbacks = good_feedback + bad_feedback
labels = ["good"] * 50 + ["bad"] * 50
df_fb = pd.DataFrame({'Feedback': feedbacks, 'Label': labels})

# Vectorize text
tfidf = TfidfVectorizer(max_features=300, stop_words='english', lowercase=True)
X_fb = tfidf.fit_transform(df_fb['Feedback'])
y_fb = df_fb['Label']

# Convert TF-IDF matrix to a DataFrame and display it
tfidf_df = pd.DataFrame(X_fb.toarray(), columns=tfidf.get_feature_names_out())
print("TF-IDF Feature Matrix:")
print(tfidf_df.head())

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_fb, y_fb, test_size=0.25, random_state=42)

# Train model
lr = LogisticRegression()
lr.fit(X_train, y_train)

# Evaluate
y_pred = lr.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Vectorize input texts
def text_preprocess_vectorize(texts, vectorizer):
    return vectorizer.transform(texts)

# Example
print("\nExample vectorized input shape:", text_preprocess_vectorize(["Great product!"], tfidf).shape)
