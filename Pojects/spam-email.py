import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, recall_score, precision_score

# Create a TfidfVectorizer object to convert text to numerical features
TfidfVectorizer_obj = TfidfVectorizer(stop_words='english', max_features=3000)

# Load the spam email dataset
data = pd.read_csv(r"C:\Users\reza\Desktop\spam.csv")

# Separate features (email text) and labels (ham/spam)
x = data["Message"]
y = data["Category"].map({"ham": 0, "spam": 1})  # Convert labels to numeric

# Split data into training and testing sets (80% train, 20% test)
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)

# Transform the text data to numerical features using TF-IDF
xtrain = TfidfVectorizer_obj.fit_transform(xtrain)  # Fit and transform training data
xtest = TfidfVectorizer_obj.transform(xtest)        # Transform test data (without fitting)

# Initialize Naive Bayes model for text classification
model = MultinomialNB()
model.fit(xtrain, ytrain)  # Train the model on training data

# Predict labels for test and training sets
ypred_test = model.predict(xtest)
ypred_train = model.predict(xtrain)

# Function to predict a single email
def pred(msg):
    message = TfidfVectorizer_obj.transform(msg)  # Convert email to numerical features
    result = model.predict(message)               # Predict using the trained model
    
    if result[0] == 1:
        print("the email is spam")
    else:
        print("the email not spam")
   
# Function to calculate and print evaluation metrics
def metric(pred, true):
    precision = precision_score(pred, true)  # Precision
    recall = recall_score(pred, true)        # Recall
    accuracy = accuracy_score(pred, true)    # Accuracy
    # Print results with 2 decimal places
    print(f"recall score : {recall * 100:.2f}\naccuracy score : {accuracy * 100:.2f}\nprecision : {precision * 100:.2f}")
