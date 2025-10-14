#!/usr/bin/env python3
"""
Student Connection Script for BAI SVM Prototype
Automatically detects Codespace info and connects to professor's server

Usage:
    python student/connect_to_professor.py                    # Use default server
    python student/connect_to_professor.py [SERVER_URL]       # Use custom server
    python student/connect_to_professor.py --test            # Test connection
"""

import os
import requests
import subprocess
import json
from datetime import datetime
import sys

# Default server configuration
DEFAULT_SERVER_URL = "https://ideal-space-dollop-9gv74g94p56f76qp-8001.app.github.dev"

def get_github_username():
    """Get GitHub username from multiple sources"""
    try:
        # Method 1: Try GitHub CLI
        result = subprocess.run(['gh', 'api', 'user'], 
                              capture_output=True, text=True, check=True)
        user_data = json.loads(result.stdout)
        return user_data.get('login', 'Unknown')
    except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError):
        pass
    
    try:
        # Method 2: Try git config
        result = subprocess.run(['git', 'config', 'user.name'], 
                              capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        pass
    
    try:
        # Method 3: Try environment variable (Codespaces)
        if 'GITHUB_USER' in os.environ:
            return os.environ['GITHUB_USER']
    except:
        pass
    
    # Method 4: Fallback - ask user
    username = input("Please enter your GitHub username: ").strip()
    return username if username else "Unknown"

def get_codespace_info():
    """Get Codespace information if running in Codespace"""
    codespace_name = os.environ.get('CODESPACE_NAME', 'local-environment')
    github_repo = os.environ.get('GITHUB_REPOSITORY', 'local-repo')
    
    if codespace_name != 'local-environment':
        # Running in Codespace
        codespace_url = f"https://{codespace_name}.github.dev"
        is_codespace = True
    else:
        # Running locally
        codespace_url = "http://localhost:8888"  # Typical Jupyter port
        is_codespace = False
    
    return {
        'codespace_name': codespace_name,
        'codespace_url': codespace_url,
        'repository': github_repo,
        'is_codespace': is_codespace
    }

def detect_chapter():
    """Try to detect which chapter/module we're working on"""
    try:
        # Check current directory for clues
        current_dir = os.getcwd()
        if 'svm' in current_dir.lower():
            return 'chapter5-svm'
        elif 'tree' in current_dir.lower():
            return 'chapter6-trees'
        elif 'neural' in current_dir.lower():
            return 'neural-networks'
        else:
            return 'svm-prototype'
    except:
        return 'unknown'

def get_server_url():
    """Get professor's server URL"""
    # Check if URL provided as command line argument
    if len(sys.argv) > 1 and sys.argv[1].startswith('http'):
        return sys.argv[1]
    
    # Check for environment variable
    server_url = os.environ.get('PROFESSOR_SERVER_URL')
    if server_url:
        return server_url
    
    # Check for config file
    config_file = os.path.join(os.path.dirname(__file__), 'server_config.txt')
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            return f.read().strip()
    
    # Use default without prompting
    print(f"🔗 Using default server: {DEFAULT_SERVER_URL}")
    
    # Save for next time
    try:
        with open(config_file, 'w') as f:
            f.write(DEFAULT_SERVER_URL)
    except:
        pass
    
    return DEFAULT_SERVER_URL

def get_live_share_url():
    """Get Live Share URL from student - auto-skip for streamlined connection"""
    print("⏭️  Skipping Live Share setup (streamlined mode)")
    return None

def connect_to_professor():
    """Main connection function"""
    print("🎓 BAI SVM Student Connection")
    print("=" * 40)
    
    # Gather all information
    print("📊 Gathering environment information...")
    github_username = get_github_username()
    codespace_info = get_codespace_info()
    chapter = detect_chapter()
    server_url = get_server_url()
    live_share_url = get_live_share_url()
    
    # Prepare registration data
    registration_data = {
        'github_username': github_username,
        'codespace_name': codespace_info['codespace_name'],
        'codespace_url': codespace_info['codespace_url'],
        'repository': codespace_info['repository'],
        'chapter': chapter,
        'timestamp': datetime.now().isoformat(),
        'environment_type': 'codespace' if codespace_info['is_codespace'] else 'local',
        'live_share_url': live_share_url
    }
    
    print(f"👤 GitHub Username: {github_username}")
    print(f"💻 Environment: {codespace_info['codespace_name']}")
    print(f"📁 Chapter: {chapter}")
    print(f"🔗 Connecting to: {server_url}")
    print()
    
    # Attempt connection
    try:
        print("🚀 Connecting to professor's server...")
        response = requests.post(
            f"{server_url}/register",
            json=registration_data,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ CONNECTION SUCCESSFUL!")
            print(f"📝 {result.get('message', 'Connected successfully')}")
            print("🎯 Professor can now access your workspace!")
            
            if codespace_info['is_codespace']:
                print(f"🔗 Your Codespace: {codespace_info['codespace_url']}")
            else:
                print("⚠️  Running locally - some features may be limited")
            
            return True
            
        else:
            print(f"❌ CONNECTION FAILED: {response.status_code}")
            try:
                error_data = response.json()
                print(f"📝 Error: {error_data.get('message', 'Unknown error')}")
            except:
                print(f"📝 Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ CONNECTION FAILED: Cannot reach professor's server")
        print("🔍 Check that:")
        print("   1. Professor's server is running")
        print("   2. Server URL is correct")
        print("   3. Network connection is working")
        return False
        
    except requests.exceptions.Timeout:
        print("❌ CONNECTION FAILED: Request timed out")
        print("🔍 Server may be overloaded or unreachable")
        return False
        
    except Exception as e:
        print(f"❌ CONNECTION FAILED: {str(e)}")
        return False

def test_connection():
    """Test if we can reach the professor's server"""
    server_url = get_server_url()
    
    try:
        print(f"🧪 Testing connection to {server_url}...")
        response = requests.get(f"{server_url}/students", timeout=5)
        
        if response.status_code == 200:
            print("✅ Server is reachable!")
            students = response.json()
            print(f"👥 Currently connected students: {len(students)}")
            for username in students.keys():
                print(f"   - {username}")
            return True
        else:
            print(f"⚠️  Server responded with status: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Cannot reach server: {str(e)}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_connection()
    else:
        success = connect_to_professor()
        
        if success:
            print("\n" + "=" * 50)
            print("🎉 You're now connected to the class!")
            print("💻 Continue working in your Codespace")
            print("👨‍🏫 Professor can see your environment")
            print("=" * 50)
        else:
            print("\n" + "=" * 50)
            print("😞 Connection failed")
            print("💡 Try running with --test flag to diagnose")
            print("📞 Contact professor if issues persist")
            print("=" * 50)
            sys.exit(1)