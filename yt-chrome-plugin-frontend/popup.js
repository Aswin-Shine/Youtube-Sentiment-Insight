// popup.js - Professional Lavender UI (Fully Integrated Version)
document.addEventListener("DOMContentLoaded", async () => {
    const statusDiv = document.getElementById("output");
    const metricsContainer = document.getElementById("metrics-container");
    const predictionsList = document.getElementById("predictions-list");
    
    // Configurations
    const API_KEY = 'AIzaSyAbEUm9rM6ShlZsLSUb3CUTte7gISrH_zg'; 
    const API_URL = 'http://56.228.29.222:5000';

    chrome.tabs.query({ active: true, currentWindow: true }, async (tabs) => {
        if (!tabs[0] || !tabs[0].url) return;
        
        const urlString = tabs[0].url;
        let videoId = null;

        // --- ROBUST VIDEO ID EXTRACTION ---
        try {
            const url = new URL(urlString);
            if (url.hostname.includes("youtube.com")) {
                videoId = url.searchParams.get("v");
            } else if (url.hostname.includes("youtu.be")) {
                videoId = url.pathname.substring(1);
            }
        } catch (e) {
            console.error("URL Parsing error", e);
        }

        if (videoId && videoId.length === 11) {
            updateStatus(`<p>Video ID: <b>${videoId}</b></p><p class="pulse">Step 1: Fetching comments...</p>`);

            try {
                // 1. Fetch Comments
                const comments = await fetchComments(videoId);
                if (comments.length === 0) {
                    updateStatus(`<p style="color:var(--accent)">No public comments found.</p>`);
                    return;
                }

                updateStatus(`<p class="pulse">Step 2: Analyzing ${comments.length} comments...</p>`);

                // 2. Get Sentiment Analysis
                const predictions = await getSentimentPredictions(comments);
                updateStatus(`<span style="color:var(--lavender); font-weight:bold;">✔ Analysis Complete!</span>`);

                // 3. Display Metrics Grid
                displayMetrics(predictions);

                // 4. Local Calculation for Chart Data
                const counts = { "1": 0, "0": 0, "-1": 0 };
                predictions.forEach(p => {
                    const s = p.sentiment.toString();
                    if (counts.hasOwnProperty(s)) counts[s]++;
                });

                // 5. Trigger Visualizations
                // Fixed: Explicitly extracting text array for WordCloud
                await fetchAndDisplayChart(counts);
                await fetchAndDisplayTrendGraph(predictions);
                await fetchAndDisplayWordCloud(comments.map(c => c.text));

                // 6. Render Recent Feedback List
                renderCommentList(predictions);

            } catch (err) {
                console.error("Pipeline Error:", err);
                updateStatus(`<p style="color:var(--accent)">Error: Unable to reach analysis server.</p>`);
            }
        } else {
            updateStatus(`<p>Please open a YouTube video page to begin analysis.</p>`);
        }
    });

    // --- HELPER FUNCTIONS ---

    function updateStatus(html) {
        if (statusDiv) {
            // Maintains the "Status" label from your Bento-style UI
            statusDiv.innerHTML = `<span class="section-title">Status</span>${html}`;
        }
    }

    async function fetchComments(videoId) {
        try {
            const response = await fetch(`https://www.googleapis.com/youtube/v3/commentThreads?part=snippet&videoId=${videoId}&maxResults=100&key=${API_KEY}`);
            const data = await response.json();
            return data.items ? data.items.map(item => ({
                text: item.snippet.topLevelComment.snippet.textOriginal,
                timestamp: item.snippet.topLevelComment.snippet.publishedAt
            })) : [];
        } catch (error) {
            console.error("Fetch error:", error);
            return [];
        }
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
        if (!metricsContainer) return;
        const total = predictions.length;
        const scores = predictions.map(p => parseInt(p.sentiment));
        const avg = scores.reduce((a, b) => a + b, 0) / total;
        
        metricsContainer.innerHTML = `
            <div class="metric">
                <h3>Total Analyzed</h3>
                <p>${total}</p>
            </div>
            <div class="metric">
                <h3>Avg Sentiment</h3>
                <p>${avg.toFixed(2)}</p>
            </div>
        `;
    }

    function renderCommentList(predictions) {
        if (!predictionsList) return;
        predictionsList.innerHTML = `
            <ul class="comment-list">
                ${predictions.slice(0, 10).map(item => `
                    <li class="comment-item">
                        <span>${item.comment}</span>
                        <span class="comment-sentiment">Sentiment: ${item.sentiment}</span>
                    </li>`).join('')}
            </ul>`;
    }

    // --- VISUALIZATION HANDLERS ---

    async function fetchAndDisplayChart(counts) {
        await updateImageContainer('chart-container', `${API_URL}/generate_chart`, { sentiment_counts: counts });
    }

    async function fetchAndDisplayTrendGraph(data) {
        await updateImageContainer('trend-graph-container', `${API_URL}/generate_trend_graph`, { sentiment_data: data });
    }

    async function fetchAndDisplayWordCloud(textArray) {
        await updateImageContainer('wordcloud-container', `${API_URL}/generate_wordcloud`, { comments: textArray });
    }

    async function updateImageContainer(containerId, url, body) {
        const container = document.getElementById(containerId);
        if (!container) return;
        
        try {
            const response = await fetch(url, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(body)
            });
            const blob = await response.blob();
            const img = document.createElement('img');
            img.src = URL.createObjectURL(blob);
            container.innerHTML = ''; // Clear previous images or broken icons
            container.appendChild(img);
        } catch (err) {
            console.error(`Failed to load image for ${containerId}:`, err);
        }
    }
});