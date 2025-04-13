import pandas as pd, numpy as np, matplotlib.pyplot as plt

#I used two of the homework models in order to do some stuff
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer



# Input the CSV data, one of the comments and one for the labels.
data = pd.read_csv('toxicity_data/train.csv')
test_labels = pd.read_csv('toxicity_data/test_labels.csv')
test_data = pd.read_csv('toxicity_data/test.csv')
df = pd.DataFrame(data)
kf = pd.DataFrame(test_labels)
tf = pd.DataFrame(test_data)

# Assign the comments as the X and the different determinations of toxicity as the Y
x = df['comment_text']
#print(x)
y = df['toxic']
z = df['severe_toxic']
k = df['obscene']
n = df['threat']
m = df['insult']
h = df['identity_hate']

test_x = tf['comment_text']
test_y = kf['toxic']
test_z = kf['severe_toxic']
test_k = kf['obscene']
test_n = kf['threat']
test_m = kf['insult']
test_h = kf['identity_hate']

x_train = x.iloc[:10000]  
y_train = y.iloc[:10000]  
z_train = z.iloc[:10000]
k_train = k.iloc[:10000]
n_train = n.iloc[:10000]
m_train = m.iloc[:10000]
h_train = h.iloc[:10000]

label_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
results = {"Model": [], "Accuracy": []}

for label in label_names:
    y_train = df[label].iloc[:10000]
    y_test = kf[label].iloc[:10000]

    model = make_pipeline(TfidfVectorizer(ngram_range=(1,2), max_df=0.9, min_df=5, stop_words='english'),
                          LogisticRegression(max_iter=100000, class_weight='balanced')
    )
    model.fit(x_train, y_train)
    preds = model.predict(test_x)
    acc = accuracy_score(test_y, preds)

    results["Model"].append(f"LR - {label}")
    results["Accuracy"].append(round(acc, 3))

# Display results
results_df = pd.DataFrame(results)
print("\nModel Accuracy Comparison:")
print(results_df)