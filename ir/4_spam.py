import pandas as pd          # reads the labeled SMS dataset into a table
import urllib.request, zipfile, io  # downloads & unzips the dataset on the fly
from sklearn.model_selection import train_test_split  # splits data 80/20 for honest testing
from sklearn.feature_extraction.text import CountVectorizer  # converts words to numeric counts
from sklearn.naive_bayes import MultinomialNB          # classifier that predicts spam vs ham
from sklearn.metrics import accuracy_score, classification_report  # grades the model

# Download the dataset automatically if the file doesn't exist yet
data_file = "SMSSpamCollection"
if not pd.io.common.file_exists(data_file):
    raw = urllib.request.urlopen("https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip").read()
    with zipfile.ZipFile(io.BytesIO(raw)) as z: z.extract(data_file, ".")

# Load the tab-separated file: column 1=label (spam/ham), column 2=message text
df = pd.read_csv(data_file, sep="\t", header=None, names=["label", "text"])

# Split into 80% training and 20% testing (test set stays hidden during training)
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)

# Convert text to numbers: CountVectorizer counts how often each word appears in each message
# stop_words="english" skips useless filler words like "the", "is", "and"
vectorizer = CountVectorizer(stop_words="english")
X_train_counts = vectorizer.fit_transform(X_train)  # learn vocabulary + convert training messages
X_test_counts = vectorizer.transform(X_test)         # convert test messages using the SAME vocabulary

# Train Naive Bayes: learns which words are more likely in spam vs ham messages
# MultinomialNB is specifically designed for word-count data
model = MultinomialNB()
model.fit(X_train_counts, y_train)  # trains on the 80% training set

# Predict labels for the hidden 20% test set
predictions = model.predict(X_test_counts)
print("Accuracy:", round(accuracy_score(y_test, predictions), 3))
print(classification_report(y_test, predictions))

# Test the trained model on your own messages
print(model.predict(vectorizer.transform(["Congratulations! You've won a free iPhone. Click here now"]))[0])
print(model.predict(vectorizer.transform(["Hey, are we still meeting for lunch tomorrow?"]))[0])