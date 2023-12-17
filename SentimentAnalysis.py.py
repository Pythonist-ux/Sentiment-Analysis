import warnings
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from collections import Counter
import numpy as np


warnings.simplefilter(action='ignore', category=FutureWarning)


data = pd.read_csv('Twitter_Data.csv')


data = data.dropna(subset=['clean_text', 'category'])

X = data['clean_text']
y = data['category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


tfidf_vectorizer = TfidfVectorizer(
    max_df=0.9,
    min_df=5,
    ngram_range=(1, 2),
    stop_words='english'
)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)


classifier = MultinomialNB(alpha=0.1)  
classifier.fit(X_train_tfidf, y_train)


y_pred = classifier.predict(X_test_tfidf)


accuracy = accuracy_score(y_test, y_pred)
print("\n\n")


class_report = classification_report(y_test, y_pred, target_names=['Negative', 'Neutral', 'Positive'])


feature_names = np.array(tfidf_vectorizer.get_feature_names_out())


for i, sentiment in enumerate(classifier.classes_):
    log_probs = classifier.feature_log_prob_[i]
    top_word_indices = log_probs.argsort()[::-1][:10]  
    top_words = feature_names[top_word_indices]

    print(f"Top words for {sentiment} sentiment:")
    for word in top_words:
        print(f"    {word}")
    print("\n")

all_words = ' '.join(data['clean_text']).split()
word_freq = Counter(all_words)

output_width = 80
separator = '-' * output_width
print(separator)

title = "Classification Report:"
padding = (output_width - len(title)) // 2
centered_title = " " * padding + title
print(centered_title)


print(class_report)


print(separator)


title = "Word Frequency Distribution (Top 20):"
padding = (output_width - len(title)) // 2
centered_title = " " * padding + title
print(centered_title)


for word, freq in word_freq.most_common(20):
    print(f"{word:<20}: {freq}")

# Separator line
print(separator)

# Centered Title
title = "Example Custom Tweet and Prediction:"
padding = (output_width - len(title)) // 2
centered_title = " " * padding + title
print(centered_title)

# Example custom tweet
custom_tweet = "eww this is so bad"

# Vectorize the custom tweet and predict sentiment
custom_tweet_tfidf = tfidf_vectorizer.transform([custom_tweet])
custom_sentiment = classifier.predict(custom_tweet_tfidf)

# Print the custom tweet and its predicted sentiment
print(f"Custom Tweet: {custom_tweet}")
print(f"Predicted Sentiment: {custom_sentiment[0]}")

# Separator line
print(separator)
