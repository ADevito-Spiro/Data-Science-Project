import pandas as pd, numpy as np, sklearn, matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
"""
pip install pandas
pip install scikit-learn
pip install matplotlib
pip install nltk

"""

data = pd.read_csv('toxicity_data/test.csv')
labels = pd.read_csv('toxicity_data/test_labels.csv')
df = pd.DataFrame(data)
kf = pd.DataFrame(labels)

x = df['comment_text']
y = kf[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identify_hate']]

print(x)