import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend before importing pyplot

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import mlflow
import numpy as np
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from mlflow.tracking import MlflowClient
import matplotlib.dates as mdates
import pickle
import nltk

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Define the preprocessing function
def preprocess_comment(comment):
    """Apply preprocessing transformations to a comment."""
    try:
        # Convert to lowercase
        comment = comment.lower()

        # Remove trailing and leading whitespaces
        comment = comment.strip()

        # Remove newline characters
        comment = re.sub(r'\n', ' ', comment)

        # Remove non-alphanumeric characters, except punctuation
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        # Remove stopwords but retain important ones for sentiment analysis
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])

        # Lemmatize the words
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

        return comment
    except Exception as e:
        print(f"Error in preprocessing comment: {e}")
        return comment

# Load the model and vectorizer from the model registry and local storage
def load_model_and_vectorizer(model_name, model_version, vectorizer_path):
     mlflow.set_tracking_uri("http://13.60.35.24:5000")  # Replace with your MLflow tracking URI
     client = MlflowClient()
     model_uri = f"models:/{model_name}/{model_version}"
     model = mlflow.pyfunc.load_model(model_uri)
     with open(vectorizer_path, 'rb') as file:
         vectorizer = pickle.load(file)
     return model, vectorizer

# Initialize the model and vectorizer
model, vectorizer = load_model_and_vectorizer("yt_chrome_plugin_model", "2", "./tfidf_vectorizer.pkl")

@app.route('/')
def home():
    return "Welcome to our flask api"

@app.route('/predict_with_timestamps', methods=['POST'])
def predict_with_timestamps():
    data = request.json
    comments_data = data.get('comments', [])
    
    if not comments_data:
        return jsonify({"error": "No comments provided"}), 400

    try:
        comments = [item['text'] for item in comments_data]
        preprocessed = [preprocess_comment(c) for c in comments]
        tfidf_matrix = vectorizer.transform(preprocessed)
        
        feature_names = vectorizer.get_feature_names_out()
        df_input = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
        
        raw_predictions = model.predict(df_input)

        results = []
        for i, item in enumerate(comments_data):
            results.append({
                "comment": item['text'],
                "sentiment": int(raw_predictions[i]),
                "timestamp": item['timestamp']
            })
        return jsonify(results)
    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    comments = data.get('comments')
    if not comments:
        return jsonify({"error": "No comments provided"}), 400

    try:
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        transformed_comments = vectorizer.transform(preprocessed_comments)
        feature_names = vectorizer.get_feature_names_out()
        df_input = pd.DataFrame(transformed_comments.toarray(), columns=feature_names)
        predictions = model.predict(df_input).tolist() 
        
        response = [{"comment": c, "sentiment": s} for c, s in zip(comments, predictions)]
        return jsonify(response)
    except Exception as e:
        app.logger.error(f"Prediction failed: {str(e)}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route('/generate_chart', methods=['POST'])
def generate_chart():
    try:
        data = request.get_json()
        counts = data.get('sentiment_counts', {})
        
        labels = ['Positive', 'Neutral', 'Negative']
        # Convert keys to string just in case JS sends them as numbers
        sizes = [
            int(counts.get('1', counts.get(1, 0))),
            int(counts.get('0', counts.get(0, 0))),
            int(counts.get('-1', counts.get(-1, 0)))
        ]
        
        colors = ['#a39cf4', '#3d3d5c', '#c678dd'] 
        plt.figure(figsize=(6, 6), facecolor='none')
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140, textprops={'color': '#f1f1f1'})
        
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG', transparent=True)
        img_io.seek(0)
        plt.close()
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/generate_wordcloud', methods=['POST'])
def generate_wordcloud():
    try:
        data = request.get_json()
        comments = data.get('comments', []) # Added fallback to empty list

        if not comments:
            return jsonify({"error": "No text provided for wordcloud"}), 400

        text = ' '.join(comments)

        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='#16162a', # Matches your --bg-card precisely
            colormap='magma',           # High-contrast palette for dark mode
            mode='RGB'
        ).generate(text)

        img_io = io.BytesIO()
        wordcloud.to_image().save(img_io, format='PNG')
        img_io.seek(0)

        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/generate_trend_graph', methods=['POST'])
def generate_trend_graph():
    try:
        data = request.get_json()
        sentiment_data = data.get('sentiment_data')

        df = pd.DataFrame(sentiment_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df['sentiment'] = df['sentiment'].astype(int)

        sentiment_labels = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}
        monthly_counts = df.resample('M')['sentiment'].value_counts().unstack(fill_value=0)
        monthly_totals = monthly_counts.sum(axis=1)
        monthly_percentages = (monthly_counts.T / monthly_totals).T * 100

        for val in [-1, 0, 1]:
            if val not in monthly_percentages.columns:
                monthly_percentages[val] = 0

        monthly_percentages = monthly_percentages[[-1, 0, 1]]

        # Thematic Styling
        plt.figure(figsize=(12, 6), facecolor='#0d0d1a') 
        ax = plt.gca()
        ax.set_facecolor('#16162a')
        
        colors = {-1: '#ff79c6', 0: '#6272a4', 1: '#bd93f9'}

        for val in [-1, 0, 1]:
            plt.plot(
                monthly_percentages.index,
                monthly_percentages[val],
                marker='o',
                linewidth=2,
                label=sentiment_labels[val],
                color=colors[val]
            )

        plt.title('Sentiment Trend', color='#f1f1f1', fontsize=16, pad=20)
        plt.ylabel('Percentage (%)', color='#9499c3')
        plt.grid(True, color='#3d3d5c', linestyle='--', alpha=0.5)
        
        ax.tick_params(colors='#9499c3')
        for spine in ax.spines.values():
            spine.set_color('#3d3d5c')

        plt.legend(facecolor='#16162a', edgecolor='#3d3d5c', labelcolor='#f1f1f1')
        plt.tight_layout()

        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG', facecolor='#0d0d1a')
        img_io.seek(0)
        plt.close()

        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": f"Trend graph failed: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)