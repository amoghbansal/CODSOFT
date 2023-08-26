import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Step 1: Load and prepare the dataset
data = pd.read_csv('spam.csv', encoding='latin-1')
X = data['v2']
y = data['v1']

# Step 2: Data Preprocessing (Optional)
# If your dataset is already preprocessed, you can skip this step.

# Step 3: Feature Extraction using TF-IDF
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X)

# Step 4: Model Building and Evaluation
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Build the Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)
y_pred = nb_classifier.predict(X_test)

user_input = input("Enter a message: ")
user_input_tfidf = vectorizer.transform([user_input])

# Predict if the user's input is spam or not
prediction = nb_classifier.predict(user_input_tfidf)

if prediction[0] == 'spam':
    print("This message is classified as spam.")
else:
    print("This message is not spam.")
    
# Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label="spam")
recall = recall_score(y_test, y_pred, pos_label="spam")
f1 = f1_score(y_test, y_pred, pos_label="spam")
conf_matrix = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)
print("Confusion Matrix:")
print(conf_matrix)
