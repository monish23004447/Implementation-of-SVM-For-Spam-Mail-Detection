# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load data, clean by selecting columns, and preprocess text.
2.Encode labels, vectorize text using TF-IDF.
3.Split data, train SVM classifier.
4.Predict, evaluate accuracy, and generate report 
 

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: MONISHKUMAR.V
RegisterNumber:  212223040116
*/
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from google.colab import files

uploaded = files.upload()  
df = pd.read_csv("spam.csv", encoding='latin-1')

df = df[['v1', 'v2']]  # Select relevant columns (assuming v1 is label, v2 is text)
df.columns = ['label', 'message']  # Rename columns for readability

df['label'] = df['label'].map({'ham': 0, 'spam': 1})  # Map ham to 0 and spam to 1

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['message'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svm = SVC(kernel='linear', C=1.0, random_state=42)
svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:\n", report

```

## Output:
![Screenshot 2024-11-06 033818](https://github.com/user-attachments/assets/d4f2a48e-3612-4dbd-8316-f3f479ae79a7)



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
