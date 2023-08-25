# Step 1: Data Preparation
# You'll need a dataset containing movie plot summaries and their corresponding genres. You can obtain such a dataset from various sources, like IMDb or Kaggle. Make sure the data is well-formatted and contains both text (plot summary) and target labels (genres).
# 
# Step 2: Data Preprocessing
# Before building the model, we need to preprocess the text data. This includes removing punctuation, converting text to lowercase, and removing stop words. We will also tokenize the text to convert it into a format suitable for machine learning algorithms.
# 
# Step 3: Feature Extraction using TF-IDF
# We will use TF-IDF (Term Frequency-Inverse Document Frequency) to convert text data into numerical features. TF-IDF represents the importance of a word in a document relative to a collection of documents. It helps in vectorizing the text data for machine learning.
# 
# Step 4: Model Building
# We'll use three classifiers: Naive Bayes, Logistic Regression, and Support Vector Machines (SVM). We'll train each model on the TF-IDF transformed data and evaluate their performance.

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Step 1: Load data from text files and create DataFrames
train_data_file = "train_data.txt"
test_data_file = "test_data.txt"

def load_data(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            entry = line.strip().split(" ::: ")
            data.append(entry)
    return data

train_data = pd.DataFrame(load_data(train_data_file), columns=["ID", "TITLE", "GENRE", "DESCRIPTION"])
test_data = pd.DataFrame(load_data(test_data_file), columns=["ID", "TITLE", "DESCRIPTION"])

# Step 3: Feature Extraction using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)  # Limiting to the top 5000 features
X_train_tfidf = vectorizer.fit_transform(train_data['DESCRIPTION'])
X_test_tfidf = vectorizer.transform(test_data['DESCRIPTION'])

# Step 4: Model Building
y_train = train_data['GENRE']

# Initialize and train a classifier
classifier = MultinomialNB()  # You can choose any other classifier like Logistic Regression or SVM here
classifier.fit(X_train_tfidf, y_train)

# Step 5: Make predictions on the test data
predictions = classifier.predict(X_test_tfidf)

# Step 6: Save the predictions to a text file
test_data['PREDICTED_GENRE'] = predictions

output_file = "predictions.txt"
with open(output_file, "w", encoding="utf-8") as file:
    for index, row in test_data.iterrows():
        file.write(f"{row['ID']} ::: {row['TITLE']} ::: {row['PREDICTED_GENRE']} ::: {row['DESCRIPTION']}\n")

print(f"Predictions saved to {output_file}")
