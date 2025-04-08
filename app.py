import pandas as pd, numpy as np, sklearn, matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
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

# Assign the comments as the X and the different determinations of toxicity as the Y
x = df['comment_text']
y = kf[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identify_hate']]

print(x)