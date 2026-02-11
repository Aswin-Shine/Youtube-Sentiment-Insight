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
from flask import jsonify

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



#Load the model and vectorizer from the model registry and local storage
def load_model_and_vectorizer(model_name, model_version, vectorizer_path):
         # Set MLflow tracking URI to your server
     mlflow.set_tracking_uri("http://13.60.35.24:5000")  # Replace with your MLflow tracking URI
     client = MlflowClient()
     model_uri = f"models:/{model_name}/{model_version}"
     model = mlflow.pyfunc.load_model(model_uri)
     with open(vectorizer_path, 'rb') as file:
         vectorizer = pickle.load(file)
   
     return model, vectorizer



#def load_model(model_path, vectorizer_path):
    #"""Load the trained model."""
    #try:
        #with open(model_path, 'rb') as file:
            #model = pickle.load(file)
        
        #with open(vectorizer_path, 'rb') as file:
            #vectorizer = pickle.load(file)
      
        #return model, vectorizer
    #except Exception as e:
       # raise


# Initialize the model and vectorizer
#model, vectorizer = load_model("./lgbm_model.pkl", "./tfidf_vectorizer.pkl")  

# Initialize the model and vectorizer
model, vectorizer = load_model_and_vectorizer("yt_chrome_plugin_model", "2", "./tfidf_vectorizer.pkl")  #Update paths and versions as needed

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
        # 1. Extract raw text and timestamps
        comments = [item['text'] for item in comments_data]
        
        # 2. Preprocess
        preprocessed = [preprocess_comment(c) for c in comments]
        
        # 3. Vectorize
        tfidf_matrix = vectorizer.transform(preprocessed)
        
        # 4. FIX: Convert to DataFrame with feature names to satisfy MLflow Schema
        feature_names = vectorizer.get_feature_names_out()
        df_input = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
        
        # 5. Predict
        raw_predictions = model.predict(df_input)

        # 6. Map results back to the original objects
        results = []
        for i, item in enumerate(comments_data):
            results.append({
                "comment": item['text'],        # Matches popup.js expectation
                "sentiment": int(raw_predictions[i]), # Standard Python int for JSON
                "timestamp": item['timestamp']
            })

        return jsonify(results) # Returns the list directly to the JS fetch

    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500



@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    comments = data.get('comments')
    
    # Log incoming data for debugging
    print(f"Incoming comments count: {len(comments) if comments else 0}")
    
    if not comments:
        return jsonify({"error": "No comments provided"}), 400

    try:
        # 1. Preprocess each comment before vectorizing
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        
        # 2. Transform comments using the vectorizer
        transformed_comments = vectorizer.transform(preprocessed_comments)

        # 3. FIX: Convert sparse matrix to DataFrame with feature names
        # This satisfies the MLflow Schema/Signature requirement
        feature_names = vectorizer.get_feature_names_out()
        df_input = pd.DataFrame(transformed_comments.toarray(), columns=feature_names)
        
        # 4. Make predictions using the labeled DataFrame
        predictions = model.predict(df_input).tolist() 
        
    except Exception as e:
        app.logger.error(f"Prediction failed: {str(e)}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
    
    # 5. Return the response with original comments and predicted sentiments
    response = [
        {"comment": comment, "sentiment": sentiment} 
        for comment, sentiment in zip(comments, predictions)
    ]
    return jsonify(response)



@app.route('/generate_chart', methods=['POST'])
def generate_chart():
    try:
        data = request.get_json()
        sentiment_counts = data.get('sentiment_counts')
        
        if not sentiment_counts:
            return jsonify({"error": "No sentiment counts provided"}), 400

        # Prepare data for the pie chart
        labels = ['Positive', 'Neutral', 'Negative']
        sizes = [
            int(sentiment_counts.get('1', 0)),
            int(sentiment_counts.get('0', 0)),
            int(sentiment_counts.get('-1', 0))
        ]
        if sum(sizes) == 0:
            raise ValueError("Sentiment counts sum to zero")
        
        colors = ['#36A2EB', '#C9CBCF', '#FF6384']  # Blue, Gray, Red

        # Generate the pie chart
        plt.figure(figsize=(6, 6))
        plt.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=140,
            textprops={'color': 'w'}
        )
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        # Save the chart to a BytesIO object
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG', transparent=True)
        img_io.seek(0)
        plt.close()

        # Return the image as a response
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_chart: {e}")
        return jsonify({"error": f"Chart generation failed: {str(e)}"}), 500

@app.route('/generate_wordcloud', methods=['POST'])
def generate_wordcloud():
    try:
        data = request.get_json()
        comments = data.get('comments')

        if not comments:
            return jsonify({"error": "No comments provided"}), 400

        # Preprocess comments
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]

        # Combine all comments into a single string
        text = ' '.join(preprocessed_comments)

        # Generate the word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='black',
            colormap='Blues',
            stopwords=set(stopwords.words('english')),
            collocations=False
        ).generate(text)

        # Save the word cloud to a BytesIO object
        img_io = io.BytesIO()
        wordcloud.to_image().save(img_io, format='PNG')
        img_io.seek(0)

        # Return the image as a response
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_wordcloud: {e}")
        return jsonify({"error": f"Word cloud generation failed: {str(e)}"}), 500

@app.route('/generate_trend_graph', methods=['POST'])
def generate_trend_graph():
    try:
        data = request.get_json()
        sentiment_data = data.get('sentiment_data')

        if not sentiment_data:
            return jsonify({"error": "No sentiment data provided"}), 400

        # Convert sentiment_data to DataFrame
        df = pd.DataFrame(sentiment_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Set the timestamp as the index
        df.set_index('timestamp', inplace=True)

        # Ensure the 'sentiment' column is numeric
        df['sentiment'] = df['sentiment'].astype(int)

        # Map sentiment values to labels
        sentiment_labels = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}

        # Resample the data over monthly intervals and count sentiments
        monthly_counts = df.resample('M')['sentiment'].value_counts().unstack(fill_value=0)

        # Calculate total counts per month
        monthly_totals = monthly_counts.sum(axis=1)

        # Calculate percentages
        monthly_percentages = (monthly_counts.T / monthly_totals).T * 100

        # Ensure all sentiment columns are present
        for sentiment_value in [-1, 0, 1]:
            if sentiment_value not in monthly_percentages.columns:
                monthly_percentages[sentiment_value] = 0

        # Sort columns by sentiment value
        monthly_percentages = monthly_percentages[[-1, 0, 1]]

        # Plotting
        plt.figure(figsize=(12, 6))

        colors = {
            -1: 'red',     # Negative sentiment
            0: 'gray',     # Neutral sentiment
            1: 'green'     # Positive sentiment
        }

        for sentiment_value in [-1, 0, 1]:
            plt.plot(
                monthly_percentages.index,
                monthly_percentages[sentiment_value],
                marker='o',
                linestyle='-',
                label=sentiment_labels[sentiment_value],
                color=colors[sentiment_value]
            )

        plt.title('Monthly Sentiment Percentage Over Time')
        plt.xlabel('Month')
        plt.ylabel('Percentage of Comments (%)')
        plt.grid(True)
        plt.xticks(rotation=45)

        # Format the x-axis dates
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))

        plt.legend()
        plt.tight_layout()

        # Save the trend graph to a BytesIO object
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG')
        img_io.seek(0)
        plt.close()

        # Return the image as a response
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_trend_graph: {e}")
        return jsonify({"error": f"Trend graph generation failed: {str(e)}"}), 500



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
