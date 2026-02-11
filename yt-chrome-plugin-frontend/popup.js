// popup.js - Lavender UI Compatible (Fixed Version)
document.addEventListener("DOMContentLoaded", async () => {
  const statusDiv = document.getElementById("output");
  const metricsContainer = document.getElementById("metrics-container");
  const predictionsList = document.getElementById("predictions-list");
  
  const API_KEY = 'AIzaSyAbEUm9rM6ShlZsLSUb3CUTte7gISrH_zg'; 
  const API_URL = 'http://56.228.29.222:5000';

  chrome.tabs.query({ active: true, currentWindow: true }, async (tabs) => {
    const url = tabs[0].url;
    // FIXED: Removed the '價' typo and improved regex
    const youtubeRegex = /^https:\/\/(?:www\.)?youtube\.com\/watch\?v=([\w-]{11})/;
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
      
      statusDiv.innerHTML = `<span class="section-title">Status</span><p>Analysis Complete!</p>`;

      // 1. Metrics Calculation
      displayMetrics(predictions);

      // 2. Chart Logic - FIXED: Now calculating counts to match app.py
      const counts = { "1": 0, "0": 0, "-1": 0 };
      predictions.forEach(p => { counts[p.sentiment.toString()]++; });
      
      await fetchAndDisplayChart(counts);
      await fetchAndDisplayTrendGraph(predictions);
      await fetchAndDisplayWordCloud(comments.map(c => c.text));

      // 3. Comment Feed
      renderCommentList(predictions);

    } else {
      statusDiv.innerHTML = `<p>Please open a YouTube video page.</p>`;
    }
  });

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
    } catch (e) { console.error("Fetch error:", e); }
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
    const total = predictions.length;
    const scores = predictions.map(p => p.sentiment);
    const avg = scores.reduce((a, b) => a + b, 0) / total;
    metricsContainer.innerHTML = `
      <div class="metric"><h3>Total Comments</h3><p>${total}</p></div>
      <div class="metric"><h3>Avg Sentiment</h3><p>${avg.toFixed(2)}</p></div>`;
  }

  function renderCommentList(predictions) {
    predictionsList.innerHTML = `<ul class="comment-list">
      ${predictions.slice(0, 10).map(item => `
        <li class="comment-item">
          <span>${item.comment}</span>
          <span class="comment-sentiment">Type: ${item.sentiment}</span>
        </li>`).join('')}</ul>`;
  }

  async function fetchAndDisplayChart(counts) {
    const response = await fetch(`${API_URL}/generate_chart`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sentiment_counts: counts }) // FIXED: Correct key for app.py
    });
    const blob = await response.blob();
    const img = document.createElement('img');
    img.src = URL.createObjectURL(blob);
    document.getElementById('chart-container').appendChild(img);
  }

  async function fetchAndDisplayTrendGraph(data) {
    const response = await fetch(`${API_URL}/generate_trend_graph`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sentiment_data: data })
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