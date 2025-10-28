import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,recall_score,precision_score

TfidfVectorizer_obj = TfidfVectorizer(stop_words='english', max_features=3000)


data = pd.read_csv(r"C:\Users\reza\Desktop\spam.csv")
x = data["Message"]
y = data["Category"].map({"ham": 0, "spam": 1})
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2)

xtrain = TfidfVectorizer_obj.fit_transform(xtrain)
xtest = TfidfVectorizer_obj.transform(xtest)

model = MultinomialNB()
model.fit(xtrain,ytrain)

ypred_test = model.predict(xtest)
ypred_train = model.predict(xtrain)

def pred(msg):
    message = TfidfVectorizer_obj.transform(msg)
    result = model.predict(message)
    
    if result[0] == 1:
        print("the email is spam")
    else:
        print("the email not spam")
   

def metric(pred,true):
    precision = precision_score(pred,true)
    recall = recall_score(pred,true)
    accuracy = accuracy_score(pred,true)
    print(f"recall score : {recall * 100:.2f}\naccuracy score : {accuracy * 100:.2f}\nprecision : {precision * 100:.2f}")