import pandas as pd, numpy as np, sklearn, matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize

#I used two of the homework models in order to do some stuff
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#This isletting me turn the text into numbers and then the pipeline lets me combine it with LR
#This training it to be able to tell what type of toxicity it is.
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

#If yall wanna use something more fancy than sklearn in the future feel free but I just want us to actually have something Sunday and I am at a runner's rn 
"""
You will need to install this in your venv folder.

pip install pandas
pip install scikit-learn
pip install matplotlib
pip install nltk

"""

# Input the CSV data, one of the comments and one for the labels.
# TODO: Do something with the IDs to link the comments to the labels
data = pd.read_csv('toxicity_data/test.csv')
labels = pd.read_csv('toxicity_data/test_labels.csv')
df = pd.DataFrame(data)
kf = pd.DataFrame(labels)
print(kf.columns)

# Assign the comments as the X and the different determinations of toxicity as the Y
x = df['comment_text']
print(x)
#You had it as identity hate when in the excel sheet it was identity_hate
y = kf['toxic']
print(y)
#Later on I will change the test data to be the actual test file, but I am lazy rn sorry guys
x_train = x.iloc[:1000]  
y_train = y.iloc[:1000]  

x_test = x.iloc[1000:1400]   
y_test = y.iloc[1000:1400] 

LR_model = make_pipeline(TfidfVectorizer() , LogisticRegression())


LR_model.fit(x_train , y_train)

LR_y_pred = LR_model.predict(x_test)

accuracy = accuracy_score(y_test, LR_y_pred)

print(f"\nLogistic Regression Model Accuracy: {accuracy:.2f}")
