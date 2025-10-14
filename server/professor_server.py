#!/usr/bin/env python3
"""
Professor Server for BAI SVM Prototype
Receives connections from student Codespaces and provides live dashboard
"""

from flask import Flask, request, jsonify, render_template_string
from flask_socketio import SocketIO, emit
import json
import os
from datetime import datetime
import logging
import csv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'bai-svm-prototype-secret'
socketio = SocketIO(app, cors_allowed_origins="*")

# Store connected students
connected_students = {}

# CSV audit log file path
AUDIT_LOG_PATH = '../audit_log.csv'

def init_audit_log():
    """Initialize the audit log CSV file with headers if it doesn't exist"""
    if not os.path.exists(AUDIT_LOG_PATH):
        with open(AUDIT_LOG_PATH, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'timestamp',
                'github_username', 
                'codespace_name',
                'codespace_url',
                'repository',
                'chapter',
                'event_type',
                'live_share_url'
            ])
        logger.info(f"Created audit log file: {AUDIT_LOG_PATH}")

def log_student_event(student_data, event_type):
    """Log student connection/disconnection events to CSV"""
    try:
        with open(AUDIT_LOG_PATH, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                datetime.now().isoformat(),
                student_data.get('github_username', 'Unknown'),
                student_data.get('codespace_name', 'Unknown'),
                student_data.get('codespace_url', ''),
                student_data.get('repository', ''),
                student_data.get('chapter', 'Unknown'),
                event_type,
                student_data.get('live_share_url', '')
            ])
        logger.info(f"Logged {event_type} event for {student_data.get('github_username', 'Unknown')}")
    except Exception as e:
        logger.error(f"Failed to log audit event: {str(e)}")

@app.route('/')
def dashboard():
    """Main dashboard for professor"""
    return render_template_string(DASHBOARD_HTML)

@app.route('/register', methods=['POST'])
def register_student():
    """Register a student connection"""
    try:
        data = request.get_json()
        
        # Extract student information
        github_username = data.get('github_username', 'Unknown')
        codespace_name = data.get('codespace_name', 'Unknown')
        codespace_url = data.get('codespace_url', '')
        repository = data.get('repository', '')
        chapter = data.get('chapter', 'Unknown')
        timestamp = data.get('timestamp', datetime.now().isoformat())
        live_share_url = data.get('live_share_url', None)
        
        # Store student info
        student_info = {
            'github_username': github_username,
            'codespace_name': codespace_name,
            'codespace_url': codespace_url,
            'repository': repository,
            'chapter': chapter,
            'timestamp': timestamp,
            'status': 'connected',
            'last_seen': datetime.now().isoformat(),
            'live_share_url': live_share_url
        }
        
        connected_students[github_username] = student_info
        
        # Log the connection event to CSV
        log_student_event(student_info, 'connected')
        
        logger.info(f"Student registered: {github_username} from {codespace_name}")
        
        # Notify dashboard of new connection
        socketio.emit('student_connected', student_info)
        
        return jsonify({
            'status': 'success',
            'message': f'Welcome {github_username}! Professor can now see your workspace.',
            'server_time': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'Registration failed'
        }), 500

@app.route('/students')
def get_students():
    """Get list of connected students"""
    return jsonify(connected_students)

@app.route('/ping/<username>')
def ping_student(username):
    """Ping a specific student"""
    if username in connected_students:
        connected_students[username]['last_seen'] = datetime.now().isoformat()
        # Log ping event
        log_student_event(connected_students[username], 'pinged')
        socketio.emit('student_pinged', {'username': username})
        return jsonify({'status': 'pinged'})
    return jsonify({'status': 'not_found'}), 404

@socketio.on('connect')
def handle_connect():
    """Handle dashboard connection"""
    logger.info("Dashboard connected")
    emit('initial_data', connected_students)

@socketio.on('request_student_list')
def handle_student_list_request():
    """Send current student list to dashboard"""
    emit('student_list_update', connected_students)

# Dashboard HTML Template
DASHBOARD_HTML = '''
<!DOCTYPE html>
<html>
<head>
    <title>🎓 BAI SVM Live Dashboard</title>
    <script src="https://cdn.socket.io/4.7.2/socket.io.min.js"></script>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            min-height: 100vh;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        .header h1 {
            font-size: 2.5em;
            margin: 0;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .subtitle {
            font-size: 1.2em;
            opacity: 0.9;
            margin-top: 10px;
        }
        .student-table {
            width: 100%;
            margin-top: 30px;
            background: rgba(255,255,255,0.1);
            border-radius: 15px;
            overflow: hidden;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
        }
        .student-table table {
            width: 100%;
            border-collapse: collapse;
        }
        .student-table th {
            background: rgba(255,255,255,0.2);
            padding: 15px;
            text-align: left;
            font-weight: bold;
            border-bottom: 1px solid rgba(255,255,255,0.2);
        }
        .student-table td {
            padding: 12px 15px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
            vertical-align: middle;
        }
        .student-table tr:hover {
            background: rgba(255,255,255,0.1);
        }
        .student-avatar {
            width: 35px;
            height: 35px;
            border-radius: 50%;
            background: #3498db;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            font-size: 0.9em;
            margin-right: 10px;
        }
        .student-status {
            padding: 4px 12px;
            border-radius: 15px;
            font-size: 0.8em;
            background: #27ae60;
            color: white;
            display: inline-block;
        }
        .action-buttons {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }
        .btn {
            background: rgba(52, 152, 219, 0.8);
            border: none;
            color: white;
            padding: 8px 15px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 0.9em;
            transition: background 0.3s;
            text-decoration: none;
            display: inline-block;
            text-align: center;
        }
        .btn:hover {
            background: rgba(52, 152, 219, 1);
        }
        .btn-success {
            background: rgba(46, 204, 113, 0.8);
        }
        .btn-success:hover {
            background: rgba(46, 204, 113, 1);
        }
        .no-students {
            text-align: center;
            font-size: 1.2em;
            opacity: 0.7;
            margin-top: 50px;
        }
        .stats {
            display: flex;
            justify-content: center;
            gap: 40px;
            margin: 20px 0;
        }
        .stat {
            text-align: center;
        }
        .stat-number {
            font-size: 2em;
            font-weight: bold;
        }
        .stat-label {
            opacity: 0.8;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎓 BAI SVM Live Dashboard</h1>
            <div class="subtitle">Real-time Student Codespace Connections</div>
            <div class="stats">
                <div class="stat">
                    <div class="stat-number" id="student-count">0</div>
                    <div class="stat-label">Connected Students</div>
                </div>
                <div class="stat">
                    <div class="stat-number" id="active-count">0</div>
                    <div class="stat-label">Active Sessions</div>
                </div>
            </div>
        </div>
        
        <div id="student-container">
            <div class="no-students">
                <h3>⏳ Waiting for student connections...</h3>
                <p>Students should run: <code>python student/connect_to_professor.py</code></p>
            </div>
        </div>
    </div>

    <script>
        const socket = io();
        const studentContainer = document.getElementById('student-container');
        const studentCountEl = document.getElementById('student-count');
        const activeCountEl = document.getElementById('active-count');
        
        let students = {};

        socket.on('connect', function() {
            console.log('Connected to server');
            socket.emit('request_student_list');
        });

        socket.on('initial_data', function(data) {
            students = data;
            updateDisplay();
        });

        socket.on('student_connected', function(studentInfo) {
            students[studentInfo.github_username] = studentInfo;
            updateDisplay();
            showNotification(`${studentInfo.github_username} connected!`);
        });

        socket.on('student_list_update', function(data) {
            students = data;
            updateDisplay();
        });

        function updateDisplay() {
            const studentCount = Object.keys(students).length;
            const activeCount = Object.values(students).filter(s => s.status === 'connected').length;
            
            studentCountEl.textContent = studentCount;
            activeCountEl.textContent = activeCount;
            
            if (studentCount === 0) {
                studentContainer.innerHTML = `
                    <div class="no-students">
                        <h3>⏳ Waiting for student connections...</h3>
                        <p>Students should run: <code>python student/connect_to_professor.py</code></p>
                    </div>
                `;
                return;
            }
            
            const table = createStudentTable(students);
            studentContainer.innerHTML = '';
            studentContainer.appendChild(table);
        }

        function createStudentTable(students) {
            const tableContainer = document.createElement('div');
            tableContainer.className = 'student-table';
            
            let tableHTML = `
                <table>
                    <thead>
                        <tr>
                            <th>Student</th>
                            <th>Codespace</th>
                            <th>Chapter</th>
                            <th>Status</th>
                            <th>Connected</th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody>
            `;
            
            Object.values(students).forEach(student => {
                const initials = student.github_username.substring(0, 2).toUpperCase();
                const connectedTime = new Date(student.timestamp).toLocaleTimeString();
                
                tableHTML += `
                    <tr>
                        <td>
                            <div style="display: flex; align-items: center;">
                                <div class="student-avatar">${initials}</div>
                                <strong>${student.github_username}</strong>
                            </div>
                        </td>
                        <td>${student.codespace_name}</td>
                        <td>📁 ${student.chapter}</td>
                        <td><span class="student-status">${student.status}</span></td>
                        <td>⏰ ${connectedTime}</td>
                        <td>
                            <div class="action-buttons">
                                <a href="${student.codespace_url}" target="_blank" class="btn">🔗 Codespace</a>
                                ${student.live_share_url ? 
                                    `<a href="${student.live_share_url}" target="_blank" class="btn btn-success">🤝 Live Share</a>` : 
                                    '<span class="btn" style="opacity:0.5; cursor:default;">🤝 No Share</span>'
                                }
                                <button onclick="pingStudent('${student.github_username}')" class="btn btn-success">📡 Ping</button>
                            </div>
                        </td>
                    </tr>
                `;
            });
            
            tableHTML += `
                    </tbody>
                </table>
            `;
            
            tableContainer.innerHTML = tableHTML;
            return tableContainer;
        }

        function pingStudent(username) {
            fetch(`/ping/${username}`)
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'pinged') {
                        showNotification(`Pinged ${username}!`);
                    }
                });
        }

        function showNotification(message) {
            // Simple notification - could be enhanced with a proper notification system
            console.log('Notification:', message);
        }

        // Refresh student list every 30 seconds
        setInterval(() => {
            socket.emit('request_student_list');
        }, 30000);
    </script>
</body>
</html>
'''

if __name__ == '__main__':
    # Initialize audit log
    init_audit_log()
    
    print("🎓 Starting BAI SVM Professor Server...")
    print("📊 Dashboard: http://localhost:8001")
    print("🔗 Student Registration: http://localhost:8001/register")
    print("👥 Students API: http://localhost:8001/students")
    print("📋 Audit Log: audit_log.csv")
    print()
    
    socketio.run(app, host='0.0.0.0', port=8001, debug=True)