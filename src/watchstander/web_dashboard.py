import os
import json
import redis
from datetime import datetime
from flask import Flask, jsonify, render_template_string, request

app = Flask(__name__)

REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
REDIS_STREAM_KEY = "vision_events"

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sentinel Vision Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <style>
        body { background-color: #f8f9fa; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
        .dashboard-header { background-color: #2c3e50; color: white; padding: 1rem 0; margin-bottom: 2rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        .event-card { margin-bottom: 1.5rem; border: none; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); transition: transform 0.2s; }
        .event-card:hover { transform: translateY(-2px); box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
        .card-header { background-color: #fff; border-bottom: 1px solid #edf2f9; font-weight: bold; border-radius: 10px 10px 0 0 !important; }
        .card-body { padding: 1.5rem; }
        .markdown-body { font-size: 0.95rem; line-height: 1.6; color: #333; }
        .markdown-body h1, .markdown-body h2, .markdown-body h3 { margin-top: 1rem; margin-bottom: 0.5rem; font-weight: 600; color: #2c3e50; }
        .markdown-body ul, .markdown-body ol { padding-left: 1.5rem; }
        .markdown-body p { margin-bottom: 0.8rem; }
        .badge-custom { background-color: #e74c3c; font-size: 0.8rem; padding: 0.4em 0.8em; }
        .badge-time { background-color: #34495e; font-size: 0.8rem; }
    </style>
</head>
<body>
    <div class="dashboard-header">
        <div class="container">
            <div class="d-flex justify-content-between align-items-center">
                <h2 class="m-0">Sentinel Vision Dashboard</h2>
                <div>
                    <button class="btn btn-outline-light btn-sm me-3" onclick="triggerAnalyzeScene()" id="btnAnalyze">Analyze Scene</button>
                    <div class="spinner-grow spinner-grow-sm text-light" role="status" id="loadingIndicator">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <div class="container mb-5">
        <ul class="nav nav-tabs" id="dashboardTabs" role="tablist">
            <li class="nav-item" role="presentation">
                <button class="nav-link active fw-bold" id="events-tab" data-bs-toggle="tab" data-bs-target="#events-pane" type="button" role="tab">Event Feed</button>
            </li>
            <li class="nav-item" role="presentation">
                <button class="nav-link fw-bold" id="prompts-tab" data-bs-toggle="tab" data-bs-target="#prompts-pane" type="button" role="tab">Configure Prompts</button>
            </li>
        </ul>

        <div class="tab-content pt-4" id="dashboardTabsContent">
            <!-- Events Pane -->
            <div class="tab-pane fade show active" id="events-pane" role="tabpanel" tabindex="0">
                <div class="row mb-3">
                    <div class="col-md-4">
                        <label for="eventTypeFilter" class="form-label fw-bold">Filter by Event Type:</label>
                        <select id="eventTypeFilter" class="form-select" onchange="applyFilter()">
                            <option value="all">All Events</option>
                        </select>
                    </div>
                </div>
                <div id="events-container">
                    <!-- Events will be injected here -->
                </div>
            </div>

            <!-- Prompts Pane -->
            <div class="tab-pane fade" id="prompts-pane" role="tabpanel" tabindex="0">
                <div class="card shadow-sm border-0 mb-4">
                    <div class="card-header bg-white border-bottom">
                        <h5 class="m-0 text-dark fw-bold">Cosmos VLM Prompts <span class="badge bg-secondary ms-2 align-middle">Live</span></h5>
                    </div>
                    <div class="card-body">
                        <form id="promptsForm">
                            <div class="mb-3">
                                <label class="form-label fw-bold text-muted small">SYSTEM INSTRUCTION</label>
                                <textarea class="form-control" id="sysPrompt" rows="3"></textarea>
                            </div>
                            <div class="mb-3">
                                <label class="form-label fw-bold text-muted small">USER PROMPT (PER VIDEO CLIP)</label>
                                <textarea class="form-control" id="userPrompt" rows="4"></textarea>
                            </div>
                            <button type="button" class="btn btn-primary" onclick="savePrompts()">Update Prompts</button>
                            <span id="saveStatus" class="ms-3 fw-bold"></span>
                        </form>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        function triggerAnalyzeScene() {
            const btn = document.getElementById('btnAnalyze');
            btn.disabled = true;
            btn.innerText = 'Requesting...';
            
            fetch('/api/analyze', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    setTimeout(() => {
                        btn.innerText = 'Analyze Scene';
                        btn.disabled = false;
                    }, 2000);
                })
                .catch(error => {
                    console.error('Error triggering analyze:', error);
                    btn.innerText = 'Error';
                    setTimeout(() => {
                        btn.innerText = 'Analyze Scene';
                        btn.disabled = false;
                    }, 3000);
                });
        }

        let currentFilter = 'status_request';
        let knownEventTypes = new Set(['status_request', 'contact_report', 'scene_summary', 'watchstander_report', 'sentry_telemetry', 'person', 'vessel']); // pre-populate some
        
        function applyFilter() {
            currentFilter = document.getElementById('eventTypeFilter').value;
            loadEvents(); // Reload to apply filter immediately
        }

        function updateFilterOptions() {
            const select = document.getElementById('eventTypeFilter');
            const currentVal = select.value;
            let optionsHtml = '<option value="all">All Events</option>';
            
            // Sort known types for consistent ordering
            const sortedTypes = Array.from(knownEventTypes).sort();
            sortedTypes.forEach(type => {
                optionsHtml += `<option value="${type}">${type.toUpperCase()}</option>`;
            });
            
            select.innerHTML = optionsHtml;
            select.value = currentFilter; // restore selected value
        }

        function loadEvents() {
            fetch('/api/events?type=' + encodeURIComponent(currentFilter))
                .then(response => response.json())
                .then(data => {
                    const container = document.getElementById('events-container');
                    
                    if (data.length === 0) {
                        container.innerHTML = '<div class="alert alert-info border-0 shadow-sm">No events found in Redis stream. Waiting for Watchstander...</div>';
                        return;
                    }
                    
                    let hasNewOptions = false;
                    data.forEach(event => {
                        if (!knownEventTypes.has(event.class)) {
                            knownEventTypes.add(event.class);
                            hasNewOptions = true;
                        }
                    });
                    
                    if (hasNewOptions) {
                        updateFilterOptions();
                    }

                    // Apply active filter
                    if (currentFilter !== 'all') {
                        data = data.filter(e => e.class === currentFilter);
                    }
                    
                    container.innerHTML = ''; // Clear current

                    if (data.length === 0) {
                        container.innerHTML = `<div class="alert alert-warning border-0 shadow-sm">No events matching filter: ${currentFilter}</div>`;
                        return;
                    }

                    data.forEach(event => {
                        const card = document.createElement('div');
                        card.className = 'card event-card';
                        
                        let detailsHtml = '';

                        if (['scene_summary', 'watchstander_report', 'status_request', 'contact_report'].includes(event.class)) {
                            detailsHtml = marked.parse(event.content);
                            if (event.image_base64) {
                                detailsHtml += `<img src="data:image/jpeg;base64,${event.image_base64}" class="img-fluid rounded mt-3" style="max-height: 400px; border: 1px solid #ddd;" alt="Scene Snapshot"/>`;
                            }
                        } else if (event.class === 'sentry_telemetry') {

                            detailsHtml = `<p><strong>Telemetry:</strong> ${event.content}</p>`;
                        } else {
                            detailsHtml = `<p><strong>Bearing:</strong> ${event.bearing}° | <strong>Range:</strong> ${event.range}m | <strong>Nav Status:</strong> ${event.nav_status}</p>`;
                            if (event.image_base64) {
                                detailsHtml += `<img src="data:image/jpeg;base64,${event.image_base64}" class="img-fluid rounded mt-2" style="max-height: 200px;" alt="Detection Snapshot"/>`;
                            }
                        }
                        
                        card.innerHTML = `
                            <div class="card-header d-flex justify-content-between align-items-center">
                                <span><span class="badge badge-custom rounded-pill me-2">${event.class.toUpperCase()}</span></span>
                                <span class="badge badge-time rounded-pill text-bg-secondary">${event.timestamp}</span>
                            </div>
                            <div class="card-body markdown-body">
                                ${detailsHtml}
                            </div>
                        `;
                        container.appendChild(card);
                    });
                })
                .catch(error => console.error('Error fetching events:', error));
        }

        // Load immediately and poll every 3 seconds
        updateFilterOptions(); // Initialize dropdown with known types
        loadEvents();
        setInterval(loadEvents, 3000);

        function loadPrompts() {
            fetch('/api/prompts')
                .then(r => r.json())
                .then(data => {
                    document.getElementById('sysPrompt').value = data.system;
                    document.getElementById('userPrompt').value = data.user;
                });
        }
        loadPrompts();

        function savePrompts() {
            const sys = document.getElementById('sysPrompt').value;
            const usr = document.getElementById('userPrompt').value;
            const status = document.getElementById('saveStatus');
            
            status.innerText = "Saving...";
            status.className = "ms-3 fw-bold text-muted";
            
            fetch('/api/prompts', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ system: sys, user: usr })
            })
            .then(r => r.json())
            .then(data => {
                status.innerText = "Saved!";
                status.className = "ms-3 fw-bold text-success";
                setTimeout(() => status.innerText = "", 2000);
            })
            .catch(err => {
                status.innerText = "Error saving";
                status.className = "ms-3 fw-bold text-danger";
            });
        }
    </script>
</body>
</html>
"""

def get_redis_client():
    return redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    try:
        r = get_redis_client()
        r.publish('vision_control', json.dumps({"command": "analyze_scene"}))
        return jsonify({"status": "success", "message": "Analyze scene command published."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/prompts', methods=['GET', 'POST'])
def api_prompts():
    r = get_redis_client()
    if request.method == 'POST':
        data = request.json
        if 'system' in data:
            r.set("prompt:cosmos:system", data['system'])
        if 'user' in data:
            r.set("prompt:cosmos:user", data['user'])
        return jsonify({"status": "success"})
    else:
        sys_prompt = r.get("prompt:cosmos:system") or "You are Sentinel, an autonomous AI watchstander. Your duty is to continuously monitor video feeds, detect anomalies, track moving objects (especially people and vessels), and provide clear, structured situation reports."
        usr_prompt = r.get("prompt:cosmos:user") or "Observe this short video clip. Please provide:\n1. A detailed scene analysis.\n2. Any objects or people of interest.\n3. Anomaly detection (is anything out of the ordinary?).\nStructure your response clearly and explain your reasoning."
        return jsonify({
            "system": sys_prompt,
            "user": usr_prompt
        })

@app.route('/api/events')
def api_events():
    try:
        event_type = request.args.get('type', 'all')
        r = get_redis_client()
        # Retrieve the latest 100 events from the stream
        raw_events = r.xrevrange(REDIS_STREAM_KEY, max='+', min='-', count=100)
        
        parsed_events = []
        for event_id, event_data in raw_events:
            event_class = event_data.get('class', 'unknown')
            if event_type != 'all' and event_class != event_type:
                continue
                
            ts_str = event_data.get('timestamp', '0')
            try:
                local_ts = datetime.fromtimestamp(float(ts_str)).strftime('%Y-%m-%d %H:%M:%S')
            except:
                local_ts = event_id.split('-')[0]
                
            parsed_event = {
                'id': event_id,
                'class': event_data.get('class', 'unknown'),
                'timestamp': local_ts
            }
            
            if parsed_event['class'] in ['scene_summary', 'watchstander_report', 'sentry_telemetry', 'status_request', 'contact_report']:
                parsed_event['content'] = event_data.get('content', 'No content provided.')
            else:
                parsed_event['bearing'] = event_data.get('bearing', '?')
                parsed_event['range'] = event_data.get('range', '?')
                parsed_event['nav_status'] = event_data.get('nav_status', '?')
            
            # Allow image_base64 for any event type, including scene_summary
            if 'image_base64' in event_data:
                parsed_event['image_base64'] = event_data['image_base64']
                
            parsed_events.append(parsed_event)
            
        return jsonify(parsed_events)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)
