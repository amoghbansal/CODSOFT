import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

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

vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(train_data['DESCRIPTION'])
X_test_tfidf = vectorizer.transform(test_data['DESCRIPTION'])

y_train = train_data['GENRE']

classifier = MultinomialNB()
classifier.fit(X_train_tfidf, y_train)

predictions = classifier.predict(X_test_tfidf)

test_data['PREDICTED_GENRE'] = predictions

output_file = "predictions.txt"
with open(output_file, "w", encoding="utf-8") as file:
    for index, row in test_data.iterrows():
        file.write(f"{row['ID']} ::: {row['TITLE']} ::: {row['PREDICTED_GENRE']} ::: {row['DESCRIPTION']}\n")

print(f"Predictions saved to {output_file}")
while True:
    user_input = input("Enter a movie description (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    user_input_tfidf = vectorizer.transform([user_input])
    predicted_genre = classifier.predict(user_input_tfidf)
    print(f"Predicted genre: {predicted_genre[0]}")