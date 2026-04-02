from flask import Flask, request, render_template_string
import os

app = Flask(__name__)
URL_FILE = "/tmp/youtube_url"

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>RTSP Simulator Dashboard</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #121212; color: #e0e0e0; max-width: 600px; margin: 40px auto; padding: 20px; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.5); }
        h1 { color: #00e676; text-align: center; }
        .card { background: #1e1e1e; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        input[type="text"] { width: 100%; padding: 12px; margin: 10px 0; border: 1px solid #333; background: #2c2c2c; color: #fff; border-radius: 4px; box-sizing: border-box; }
        button { background: #00e676; color: #000; padding: 12px 20px; border: none; border-radius: 4px; cursor: pointer; font-weight: bold; width: 100%; }
        button:hover { background: #00c853; }
        .status { margin-top: 15px; font-size: 0.9em; color: #888; text-align: center; }
        .presets { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 20px; justify-content: center; }
        .preset-btn { background: #333; color: #fff; padding: 8px 12px; border: none; border-radius: 4px; cursor: pointer; font-size: 0.8em; }
        .preset-btn:hover { background: #444; }
    </style>
</head>
<body>
    <h1>🚢 RTSP Simulator</h1>
    <div class="card">
        <form method="POST" action="/stream">
            <label>YouTube URL:</label>
            <input type="text" name="url" placeholder="Paste YouTube link here..." value="{{ current_url }}">
            <button type="submit">Update Stream</button>
        </form>
        <div class="status">Current Source: {{ current_url }}</div>
    </div>

    <h3>Maritime Presets</h3>
    <div class="presets">
        <button class="preset-btn" onclick="updateUrl('https://www.youtube.com/watch?v=1tPNNOwZ2F0')">Harbor Cam 1</button>
        <button class="preset-btn" onclick="updateUrl('https://www.youtube.com/watch?v=U-MFYTeJZqc')">Miami Beach</button>
        <button class="preset-btn" onclick="updateUrl('https://www.youtube.com/watch?v=21X5lGlDOfg')">Harbor Cam 2</button>
    </div>

    <script>
        function updateUrl(url) {
            document.querySelector('input[name="url"]').value = url;
        }
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    current_url = "Unknown"
    if os.path.exists(URL_FILE):
        with open(URL_FILE, "r") as f:
            current_url = f.read().strip()
    return render_template_string(HTML_TEMPLATE, current_url=current_url)

@app.route("/stream", methods=["POST"])
def stream():
    url = request.form.get("url")
    if url:
        with open(URL_FILE, "w") as f:
            f.write(url)
        return """
        <html><body style="background:#121212; color:#00e676; font-family:sans-serif; text-align:center; padding:50px;">
        <h2>Signal Sent!</h2>
        <p>FFmpeg is restarting with the new source...</p>
        <script>setTimeout(() => { window.location.href = '/'; }, 2000);</script>
        </body></html>
        """
    return "No URL provided", 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
