import pandas as pd
import gradio as gr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob  # Importing TextBlob for sentiment analysis

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Define stop words
stop_words = set(stopwords.words('english'))

# Input the CSV data
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

# Ensure matching number of samples in training and testing data
train_size = 100000
test_size = len(test_labels)  # Use the full test set size

# Limit the training data to match the test data size if necessary
x_train = x.iloc[:test_size]  # Match the size of the test set
y_train = y.iloc[:test_size]  # Match the size of the test set
z_train = z.iloc[:test_size]
k_train = k.iloc[:test_size]
n_train = n.iloc[:test_size]
m_train = m.iloc[:test_size]
h_train = h.iloc[:test_size]

label_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
results = {"Model": [], "Accuracy": []}

models = {}
for label in label_names:
    # Align y_train and y_test for each label
    y_train_label = df[label].iloc[:test_size]
    y_test_label = test_labels[label].iloc[:test_size]

    model = make_pipeline(TfidfVectorizer(),
                          LogisticRegression(max_iter=100000, class_weight='balanced')
    )
    model.fit(x_train, y_train_label)
    preds = model.predict(test_x[:test_size])  # Limit test data to match training data size
    acc = accuracy_score(y_test_label, preds)
    models[label] = model
    results["Model"].append(f"Logistic Regression - {label}")
    results["Accuracy"].append(round(acc, 3))

# Display results
results_df = pd.DataFrame(results)
print("\nAccuracy Score:")
print(results_df)

# Function to remove stop words
def remove_stop_words(text):
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    return " ".join(filtered_tokens)

# Function to perform sentiment analysis using TextBlob
def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity  # Range from -1 to 1
    subjectivity = blob.sentiment.subjectivity  # Range from 0 to 1
    
    if polarity > 0:
        sentiment = 'Positive'
    elif polarity < 0:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'
    
    return sentiment, polarity, subjectivity

# Function to predict toxicity labels and sentiment
def predict_all_labels_with_sentiment(comments):
    # Remove stop words from the input comment
    cleaned_comments = remove_stop_words(comments)
    
    # Predict toxicity labels
    toxic_labels = []
    for label, model in models.items():
        if model.predict([cleaned_comments])[0] == 1:
            toxic_labels.append(label)
    
    # Perform sentiment analysis
    sentiment, polarity, subjectivity = analyze_sentiment(cleaned_comments)
    
    # Combine toxicity labels and sentiment
    if toxic_labels:
        return f"This comment is: {', '.join(toxic_labels)}. Sentiment: {sentiment} (Polarity: {polarity}, Subjectivity: {subjectivity})"
    else:
        return f"This comment is not toxic in any category. Sentiment: {sentiment} (Polarity: {polarity}, Subjectivity: {subjectivity})"

# Build Gradio interface
with gr.Blocks(css=""" 
.check-button {
    background-color: purple;
    color: white;
    border-radius: 12px;
}
.check-button:hover {
    background-color: #C5B4E3;
}
""") as demo:
    gr.Markdown("## 🧪 Multi-label Toxicity Checker with Sentiment Analysis")
    gr.Markdown("Enter a comment and check which toxicity labels it matches along with its sentiment.")

    with gr.Row():
        input_box = gr.Textbox(
            label="Comment",
            placeholder="Type a comment to analyze...",
            lines=3
        )
    with gr.Row():
        output_box = gr.Textbox(label="Toxicity Labels and Sentiment", lines=3, interactive=False)
        
    with gr.Row():
        predict_button = gr.Button("Check Toxicity and Sentiment", elem_classes=["check-button"])
        predict_button.click(fn=predict_all_labels_with_sentiment, inputs=input_box, outputs=output_box)

# Launch the app
demo.launch(share=True)
