// popup.js - Lavender UI Compatible
document.addEventListener("DOMContentLoaded", async () => {
  const statusDiv = document.getElementById("output"); // Targeting the Status Bento
  const metricsContainer = document.getElementById("metrics-container");
  const predictionsList = document.getElementById("predictions-list");
  
  const API_KEY = 'AIzaSyAbEUm9rM6ShlZsLSUb3CUTte7gISrH_zg'; 
  const API_URL = 'http://56.228.29.222:5000';

  chrome.tabs.query({ active: true, currentWindow: true }, async (tabs) => {
    const url = tabs[0].url;
    const youtubeRegex = /^https:\/\/價(?:www\.)?youtube\.com\/watch\?v=([\w-]{11})/;
    const match = url.match(youtubeRegex);

    if (match && match[1]) {
      const videoId = match[1];
      statusDiv.innerHTML = `<span class="section-title">Status</span><p>Video ID: <b>${videoId}</b></p><p>Fetching comments...</p>`;

      const comments = await fetchComments(videoId);
      if (comments.length === 0) {
        statusDiv.innerHTML = `<span class="section-title">Status</span><p>No comments found.</p>`;
        return;
      }

      statusDiv.innerHTML = `<span class="section-title">Status</span><p>Analyzing ${comments.length} comments...</p>`;
      
      const predictions = await getSentimentPredictions(comments);
      
      // Update Status to Done
      statusDiv.innerHTML = `<span class="section-title">Status</span><p>Analysis Complete for ${comments.length} comments.</p>`;

      // 1. Calculate and Display Metrics in the new Grid
      displayMetrics(predictions);

      // 2. Render the Charts/Visuals
      await fetchAndDisplayChart(predictions);
      await fetchAndDisplayTrendGraph(predictions);
      await fetchAndDisplayWordCloud(comments.map(c => c.text));

      // 3. Render the Recent Feedback List
      renderCommentList(predictions);

    } else {
      statusDiv.innerHTML = `<p>Please open a YouTube video page.</p>`;
    }
  });

  // --- LOGIC FUNCTIONS (Keep your original logic) ---

  async function fetchComments(videoId) {
    let comments = [];
    try {
      const response = await fetch(`https://www.googleapis.com/youtube/v3/commentThreads?part=snippet&videoId=${videoId}&maxResults=100&key=${API_KEY}`);
      const data = await response.json();
      if (data.items) {
        comments = data.items.map(item => ({
          text: item.snippet.topLevelComment.snippet.textOriginal,
          timestamp: item.snippet.topLevelComment.snippet.publishedAt
        }));
      }
    } catch (error) {
      console.error("Error fetching comments:", error);
    }
    return comments;
  }

  async function getSentimentPredictions(comments) {
    const response = await fetch(`${API_URL}/predict_with_timestamps`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ comments: comments })
    });
    return await response.json();
  }

  function displayMetrics(predictions) {
    const totalComments = predictions.length;
    const sentimentScores = predictions.map(p => parseInt(p.sentiment));
    const avgSentiment = sentimentScores.reduce((a, b) => a + b, 0) / totalComments;
    
    // Lavender UI Metric Boxes
    metricsContainer.innerHTML = `
      <div class="metric">
        <h3>Total Analyzed</h3>
        <p>${totalComments}</p>
      </div>
      <div class="metric">
        <h3>Avg Sentiment</h3>
        <p>${avgSentiment.toFixed(2)}</p>
      </div>
    `;
  }

  function renderCommentList(predictions) {
    predictionsList.innerHTML = `
      <ul class="comment-list">
        ${predictions.slice(0, 10).map((item, index) => `
          <li class="comment-item">
            <span>${item.comment}</span>
            <span class="comment-sentiment">Sentiment: ${item.sentiment}</span>
          </li>`).join('')}
      </ul>`;
  }

  // --- VISUALIZATION FUNCTIONS (Updating selectors to match Lavender IDs) ---

  async function fetchAndDisplayChart(predictions) {
    const response = await fetch(`${API_URL}/generate_chart`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ predictions: predictions.map(p => p.sentiment) })
    });
    const blob = await response.blob();
    const img = document.createElement('img');
    img.src = URL.createObjectURL(blob);
    document.getElementById('chart-container').appendChild(img);
  }

  async function fetchAndDisplayTrendGraph(predictions) {
    const response = await fetch(`${API_URL}/generate_trend_graph`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sentiment_data: predictions })
    });
    const blob = await response.blob();
    const img = document.createElement('img');
    img.src = URL.createObjectURL(blob);
    document.getElementById('trend-graph-container').appendChild(img);
  }

  async function fetchAndDisplayWordCloud(textArray) {
    const response = await fetch(`${API_URL}/generate_wordcloud`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ comments: textArray })
    });
    const blob = await response.blob();
    const img = document.createElement('img');
    img.src = URL.createObjectURL(blob);
    document.getElementById('wordcloud-container').appendChild(img);
  }
});