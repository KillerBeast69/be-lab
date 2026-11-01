import pandas as pd                          # reads the data file into a table
import urllib.request, zipfile, io           # downloads & unzips the dataset
from sklearn.model_selection import train_test_split  # splits data into train/test
from sklearn.feature_extraction.text import CountVectorizer  # turns text into numbers
from sklearn.naive_bayes import MultinomialNB          # the classifier itself
from sklearn.metrics import accuracy_score, classification_report  # grades the model

# ---- STEP 1: DOWNLOAD THE DATASET (if not already there) ----
data_file = "SMSSpamCollection"
if not pd.io.common.file_exists(data_file):
    url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    raw = urllib.request.urlopen(url).read()
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        z.extract(data_file, ".")             # unzips into your current folder

# ---- STEP 2: LOAD THE DATASET ----
df = pd.read_csv(data_file, sep="\t", header=None, names=["label", "text"])
# column 1 = label (spam or ham), column 2 = the message text

# ---- STEP 3: SPLIT INTO TRAINING & TESTING SETS ----
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42)
# trains on 80% of messages, keeps 20% hidden so testing is honest

# ---- STEP 4: CONVERT TEXT TO NUMBERS ----
vectorizer = CountVectorizer(stop_words="english")
# a computer can't read words — it needs counts
# stop_words="english" removes filler words like "the", "is", "and"
X_train_counts = vectorizer.fit_transform(X_train)  # learn vocabulary + convert
X_test_counts  = vectorizer.transform(X_test)       # convert test using SAME vocab

# ---- STEP 5: TRAIN THE CLASSIFIER (Naive Bayes) ----
model = MultinomialNB()
# Naive Bayes = "given these words, which label is more probable?"
model.fit(X_train_counts, y_train)   # learns from training examples

# ---- STEP 6: PREDICT & GRADE ----
predictions = model.predict(X_test_counts)
# feeds the unseen test messages through the trained model

print("Accuracy:", round(accuracy_score(y_test, predictions), 3))
print(classification_report(y_test, predictions))

# ---- STEP 7: TRY YOUR OWN EMAIL ----
def check(msg):
    return model.predict(vectorizer.transform([msg]))[0]

print(check("Congratulations! You've won a free iPhone. Click here now"))
print(check("Hey, are we still meeting for lunch tomorrow?"))