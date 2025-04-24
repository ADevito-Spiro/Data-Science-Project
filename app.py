import pandas as pd
import gradio as gr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

# Input the CSV data, one of the comments and one for the labels.
data = pd.read_csv('toxicity_data/train.csv')
test_labels = pd.read_csv('toxicity_data/test_labels_cleaned.csv')
test_data = pd.read_csv('toxicity_data/test_filtered.csv')
df = pd.DataFrame(data)

# Assign the comments as the X and the different determinations of toxicity as the Y
x = df['comment_text']
y = df['toxic']
z = df['severe_toxic']
k = df['obscene']
n = df['threat']
m = df['insult']
h = df['identity_hate']

test_x = test_data['comment_text']
test_y = test_labels['toxic']
test_z = test_labels['severe_toxic']
test_k = test_labels['obscene']
test_n = test_labels['threat']
test_m = test_labels['insult']
test_h = test_labels['identity_hate']

x_train = x.iloc[:100000]
y_train = y.iloc[:100000]
z_train = z.iloc[:100000]
k_train = k.iloc[:100000]
n_train = n.iloc[:100000]
m_train = m.iloc[:100000]
h_train = h.iloc[:100000]

label_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
results = {"Model": [], "Accuracy": []}

models = {}
for label in label_names:
    y_train = df[label].iloc[:100000]
    y_test = test_labels[label].iloc[:60000]

    model = make_pipeline(TfidfVectorizer(),
                          LogisticRegression(max_iter=100000, class_weight='balanced')
    )
    model.fit(x_train, y_train)
    preds = model.predict(test_x)
    acc = accuracy_score(test_y, preds)
    models[label] = model
    results["Model"].append(f"Logisitcal Regression - {label}")
    results["Accuracy"].append(round(acc, 3))

# Display results
results_df = pd.DataFrame(results)
print("\nAccuracy Score:")
print(results_df)

def predict_all_labels(comments):
    toxic_labels = []
    for label, model in models.items():
        if model.predict([comments])[0] == 1:
            toxic_labels.append(label)
    if toxic_labels:
        return f"This comment is toxic in: {', '.join(toxic_labels)}"
    else:
        return "This comment is not toxic in any category."

gr.Interface(fn=predict_all_labels,
              inputs=gr.Textbox(placeholder = "Type here...", label = "Enter an offensive or inoffensive comment!"),
              outputs=gr.Textbox(label = "Result")
              ).launch()
