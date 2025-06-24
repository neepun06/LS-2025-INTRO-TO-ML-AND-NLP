import nltk
import math
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Corpus
corpus = [
    'the sun is a star',
    'the moon is a satellite',
    'the sun and moon are celestial bodies'
]

# Step 1: Tokenize and clean using NLTK
stop_words = set(stopwords.words('english'))
tokenized_corpus = []

for doc in corpus:
    tokens = word_tokenize(doc.lower())
    filtered = [word for word in tokens if word.isalpha() and word not in stop_words]
    tokenized_corpus.append(filtered)

# Step 2: Term Frequency (TF)
tf_list = []
for doc in tokenized_corpus:
    counts = Counter(doc)
    total_terms = len(doc)
    tf = {term: count / total_terms for term, count in counts.items()}
    tf_list.append(tf)

# Step 3: Inverse Document Frequency (IDF)
N = len(tokenized_corpus)
all_terms = set(term for doc in tokenized_corpus for term in doc)
idf = {}
for term in all_terms:
    containing_docs = sum(1 for doc in tokenized_corpus if term in doc)
    idf[term] = math.log((N + 1) / (containing_docs + 1)) + 1  # Smoothing

# Step 4: TF-IDF
tfidf_manual = []
for tf in tf_list:
    tfidf_doc = {}
    for term in tf:
        tfidf_doc[term] = tf[term] * idf[term]
    tfidf_manual.append(tfidf_doc)

# Step 5: Print manual TF-IDF
print("\n🔸 Manual TF-IDF using NLTK:\n")
for i, doc in enumerate(tfidf_manual):
    print(f"Document {i+1}: {doc}")

# Step 6: Compare with Scikit-learn
print("\n🔸 CountVectorizer:\n")
count_vectorizer = CountVectorizer(stop_words='english')
count_matrix = count_vectorizer.fit_transform(corpus)
print(count_vectorizer.get_feature_names_out())
print(count_matrix.toarray())

print("\n🔸 TfidfVectorizer:\n")
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
print(tfidf_vectorizer.get_feature_names_out())
print(tfidf_matrix.toarray())
