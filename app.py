import pandas as pd, gradio as gr, matplotlib.pyplot as plt, seaborn as sns, os, nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

# Download necessary NLTK resources
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt_tab')

# Define stop words
stop_words = set(stopwords.words('english'))

# Input the CSV data
data = pd.read_csv('toxicity_data/train.csv')
test_labels = pd.read_csv('toxicity_data/test_labels_cleaned.csv')
test_data = pd.read_csv('toxicity_data/test_filtered.csv')
df = pd.DataFrame(data)

# Ensure matching number of samples in training and testing data
test_size = len(test_labels)

label_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
results = []
models = {}

# Train models and collect results
for label in tqdm(label_names, desc="Training models"):
    y_train_label = df[label].iloc[:test_size]
    y_test_label = test_labels[label].iloc[:test_size]
    model = make_pipeline(
        TfidfVectorizer(max_features=10000, ngram_range=(1,2), min_df=2, max_df=0.8, sublinear_tf=True),
        LogisticRegression(C=1.0, penalty='l2', solver='lbfgs', class_weight='balanced', max_iter=1000)
    )
    model.fit(df['comment_text'].iloc[:test_size], y_train_label)
    preds = model.predict(test_data['comment_text'][:test_size])
    report = classification_report(y_test_label, preds, output_dict=True, zero_division=0)
    
    results.append({
        "Model": f"Logistic Regression - {label}",
        "Accuracy": round(report['accuracy'], 3),
        "F1-Score": round(report['weighted avg']['f1-score'], 3),
        "Precision": round(report['weighted avg']['precision'], 3),
        "Recall": round(report['weighted avg']['recall'], 3)
    })
    models[label] = model

results_df = pd.DataFrame(results)

# Function to generate all Matplotlib diagrams
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
print("\nModel Performance:")
print(results_df)

# Generate all diagrams
generate_diagrams(models, results_df, df, test_labels, test_data['comment_text'], test_size, label_names)

# Function to remove stop words
def remove_stop_words(text):
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    return " ".join(filtered_tokens)

# Function to perform sentiment analysis using TextBlob
def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    sentiment = 'Positive' if polarity > 0 else 'Negative' if polarity < 0 else 'Neutral'
    return sentiment, polarity, subjectivity

# Function to predict toxicity labels and sentiment
def predict_all_labels_with_sentiment(comments):
    cleaned_comments = remove_stop_words(comments)
    toxic_labels = [label for label, model in models.items() if model.predict([cleaned_comments])[0] == 1]
    sentiment, polarity, subjectivity = analyze_sentiment(cleaned_comments)
    if toxic_labels:
        return f"This comment is: {', '.join(toxic_labels)}. \nSentiment: {sentiment} (Polarity: {polarity}, Subjectivity: {subjectivity})"
    return f"This comment is not toxic in any category. \nSentiment: {sentiment} (Polarity: {polarity}, Subjectivity: {subjectivity})"

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
    gr.Markdown("Toxicity Detector with Sentiment Analysis")
    gr.Markdown("Enter a comment.")
    with gr.Row():
        input_box = gr.Textbox(label="Comment", placeholder="Type a comment to analyze...", lines=3)
    with gr.Row():
        output_box = gr.Textbox(label="Toxicity Labels and Sentiment", lines=3, interactive=False)
    with gr.Row():
        predict_button = gr.Button("Check Toxicity and Sentiment", elem_classes=["check-button"])
        predict_button.click(fn=predict_all_labels_with_sentiment, inputs=input_box, outputs=output_box)

demo.launch()