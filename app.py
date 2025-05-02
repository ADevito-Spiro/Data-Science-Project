import pandas as pd, gradio as gr, matplotlib.pyplot as plt, seaborn as sns, os, nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm       # Cute lil progress bar when the program is loading

# Download necessary NLTK resources if they don't exist
# Tries and finds the resources in the local directory, if not found then it installs them
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt_tab')

# Use NLTK to fetch their stopwords, used for text pre-processing. 
stop_words = set(stopwords.words('english'))

# Input the CSV data
data = pd.read_csv('toxicity_data/train.csv')
test_labels = pd.read_csv('toxicity_data/test_labels_cleaned.csv')
test_data = pd.read_csv('toxicity_data/test_filtered.csv')
df = pd.DataFrame(data)
tf = pd.DataFrame(test_labels)
td = pd.DataFrame(test_data)

# This function removes the defined stopwords from the input text
def remove_stop_words(text):
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    return " ".join(filtered_tokens)

# Ensure matching number of samples in training and testing data
test_size = len(test_labels)

# Declare variables
label_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
results = []
models = {}
classification_reports = {}

# Train the LogisticRegression model using our data, required to make a pipeline with TF-IDF to ensure that LogisticRegression can process text data
for label in tqdm(label_names, desc="Training models"):
    y_train_label = df[label].iloc[:test_size]
    y_test_label = tf[label].iloc[:test_size]

    # Added hyperparameters, set the n-gram range to be a bigram 
    # --------------------------- TF-IDF -------------------------------------------------------
    # max_features = 10000, limits the maximum amount of terms to the 10000 most frequent terms
    # ngram_range = (1,2), sets the range to be a bigram, allowing to capture a lil context
    # min_df = 2, ignore terms that appear more than 2 times
    # max_df = 0.8, ignores terms that appear more than 80% of the time
    # sublinear_tf = True, applies sublinear scaling to term frequencies
    # --------------------------- LogisticRegression --------------------------------------------
    # C = 1.0, inverse the regularization strength, meaning smaller values imply stronger regularization
    # penalty = l2, L2 regularization
    # class_weight = balanced, adjusts weights, for imbalanced datasets it gives high weight to lesser frequent classes
    # max_iter = 1000, max number of iterations before convergence
    model = make_pipeline(
        TfidfVectorizer(max_features=10000, ngram_range=(1,2), min_df=2, max_df=0.8, sublinear_tf=True),
        LogisticRegression(C=1.0, penalty='l2', class_weight='balanced', max_iter=1000)
    )
    model.fit(df['comment_text'].iloc[:test_size], y_train_label)
    preds = model.predict(td['comment_text'][:test_size])
    report = classification_report(y_test_label, preds, output_dict=True, zero_division=0)
    
    # Print scores and labels to console for our Accuracy, F1, Precision, and Recall
    results.append({
        "Model": f"Logistic Regression - {label}",
        "Accuracy": round(report['accuracy'], 3),
        "F1-Score": round(report['weighted avg']['f1-score'], 3),
        "Precision": round(report['weighted avg']['precision'], 3),
        "Recall": round(report['weighted avg']['recall'], 3)
    })
    models[label] = model

results_df = pd.DataFrame(results)

# Generate all Matplotlib diagrams
def generate_diagrams(models, results_df, df, test_labels, test_x, test_size, label_names, output_dir='diagrams'):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Confusion Matrices
    for label in label_names:
        y_test_label = test_labels[label].iloc[:test_size]
        preds = models[label].predict(test_x[:test_size])
        cm = confusion_matrix(y_test_label, preds)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Non-' + label, label],
                    yticklabels=['Non-' + label, label])
        plt.title(f'Confusion Matrix - {label}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(os.path.join(output_dir, f'confusion_matrix_{label}.png'), bbox_inches='tight')
        plt.close()
    
    # 2. Bar Plot of Model Performance
    plt.figure(figsize=(10, 6))
    results_df_melted = pd.melt(results_df, id_vars='Model', value_vars=['Accuracy', 'F1-Score'], 
                                var_name='Metric', value_name='Score')
    sns.barplot(x='Model', y='Score', hue='Metric', data=results_df_melted)
    plt.title('Model Performance Across Toxicity Labels')
    plt.xlabel('Toxicity Label')
    plt.ylabel('Score')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Metric')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_bar_plot.png'))
    plt.close()
    
    # 3. Class Distribution Pie Charts
    plt.figure(figsize=(12, 8))
    for i, label in enumerate(label_names, 1):
        class_counts = df[label].iloc[:test_size].value_counts()
        plt.subplot(2, 3, i)
        plt.pie(class_counts, labels=['Non-' + label, label], autopct='%1.1f%%', colors=['#66b3ff', '#ff9999'])
        plt.title(f'Class Distribution - {label}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_distribution_pie.png'))
    plt.close()
    
    # 4. ROC Curves
    plt.figure(figsize=(8, 6))
    for label in label_names:
        model = models[label]
        y_score = model.predict_proba(test_x[:test_size])[:, 1]
        fpr, tpr, _ = roc_curve(test_labels[label][:test_size], y_score)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Toxicity Models')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'roc_curves.png'))
    plt.close()
    
    # 5. Precision-Recall Curves
    plt.figure(figsize=(8, 6))
    for label in label_names:
        model = models[label]
        y_score = model.predict_proba(test_x[:test_size])[:, 1]
        precision, recall, _ = precision_recall_curve(test_labels[label][:test_size], y_score)
        ap = average_precision_score(test_labels[label][:test_size], y_score)
        plt.plot(recall, precision, label=f'{label} (AP = {ap:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves for Toxicity Models')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'precision_recall_curves.png'))
    plt.close()

# Display results
results_df = pd.DataFrame(results)
# print("\nModel Performance:")
# print(results_df)

# Generate the diagrams
generate_diagrams(models, results_df, df, test_labels, td['comment_text'], test_size, label_names)

# Use TextBlob's built in sentiment analysis to determine the sentiment of the provided text, this checks after the stopwords have been removed
def check_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    sentiment = 'Positive' if polarity > 0 else 'Negative' if polarity < 0 else 'Neutral'
    return sentiment, polarity, subjectivity

# Process the data to remove the stop words from the inputted sentence, after processed take the new sentence and use the LogisticRegression model to predict the labels
# Then uses the processed words to check the sentiment of the sentence, then returning the output
def sentiment_prediction(comments):
    processed = remove_stop_words(comments)
    toxic_labels = [label for label, model in models.items() if model.predict([processed])[0] == 1]
    sentiment, polarity, subjectivity = check_sentiment(processed)
    if toxic_labels:
        return f"This comment is: {', '.join(toxic_labels)}. \nSentiment: {sentiment} (Polarity: {polarity}, Subjectivity: {subjectivity})"
    return f"This comment is not toxic in any category. \nSentiment: {sentiment} (Polarity: {polarity}, Subjectivity: {subjectivity})"

# Use Gradio to handle our text input and displaying our diagrams generated from matplotlib
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
    gr.Markdown("## Toxicity Detector with Sentiment Analysis")
    
    # Text input and label outputs
    gr.Markdown("Enter a comment to analyze for toxicity and sentiment.")
    with gr.Row():
        input_box = gr.Textbox(label="Comment", placeholder="Type a comment to analyze...", lines=3)
    with gr.Row():
        output_box = gr.Textbox(label="Toxicity Labels and Sentiment", lines=3, interactive=False)
    with gr.Row():
        predict_button = gr.Button("Check Toxicity and Sentiment", elem_classes=["check-button"])
        predict_button.click(fn=sentiment_prediction, inputs=input_box, outputs=output_box)
    
    # Model performance table
    gr.Markdown("## Model Performance Metrics")
    gr.DataFrame(results_df, label="Performance Summary")
    
    # Load generated images from matplotlib 
    gr.Markdown("## Generated Diagrams")
    with gr.Row():
        gr.Image(os.path.join('diagrams', 'performance_bar_plot.png'), label="Model Performance Bar Plot")
        gr.Image(os.path.join('diagrams', 'class_distribution_pie.png'), label="Class Distribution Pie Charts")
    with gr.Row():
        gr.Image(os.path.join('diagrams', 'roc_curves.png'), label="ROC Curves")
        gr.Image(os.path.join('diagrams', 'precision_recall_curves.png'), label="Precision-Recall Curves")
    gr.Markdown("### Confusion Matrices")
    for i in range(0, len(label_names), 2):
        with gr.Row():
            gr.Image(os.path.join('diagrams', f'confusion_matrix_{label_names[i]}.png'), 
                     label=f"Confusion Matrix - {label_names[i]}")
            if i + 1 < len(label_names):
                gr.Image(os.path.join('diagrams', f'confusion_matrix_{label_names[i+1]}.png'), 
                         label=f"Confusion Matrix - {label_names[i+1]}")
            else:
                gr.Image(None, visible=False)

demo.launch()