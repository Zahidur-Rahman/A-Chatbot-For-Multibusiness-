#!/usr/bin/env python3
"""
Development startup script for Multi-Business Conversational Chatbot.
This script helps you get started with the development environment.
"""

import os
import sys
import subprocess
import time
from pathlib import Path
from pydantic_settings import BaseSettings

def print_banner():
    """Print startup banner"""
    print("=" * 60)
    print("🚀 Multi-Business Conversational Chatbot")
    print("   Development Environment Setup")
    print("=" * 60)

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 9):
        print("❌ Python 3.9+ is required")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import fastapi
        import uvicorn
        import psycopg2
        print("✅ Core dependencies found")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please run: pip install -r backend/requirements.txt")
        return False
    return True

def setup_environment():
    """Setup environment variables"""
    env_file = Path(".env")
    env_example = Path("env.example")
    
    if not env_file.exists() and env_example.exists():
        print("📝 Creating .env file from template...")
        with open(env_example, 'r') as f:
            content = f.read()
        with open(".env", 'w') as f:
            f.write(content)
        print("✅ .env file created")
    elif env_file.exists():
        print("✅ .env file already exists")
    else:
        print("⚠️  No .env file found. Please create one manually.")

def run_tests():
    """Run basic tests"""
    print("🧪 Running basic tests...")
    try:
        result = subprocess.run([sys.executable, "tests/test_basic.py"], 
                              capture_output=True, text=True, cwd=".")
        if result.returncode == 0:
            print("✅ Basic tests passed")
        else:
            print("❌ Basic tests failed")
            print(result.stderr)
    except Exception as e:
        print(f"⚠️  Could not run tests: {e}")

def start_services():
    """Start development services"""
    print("\n🔧 Starting development services...")
    print("Choose an option:")
    print("1. Start with Docker Compose (recommended)")
    print("2. Start FastAPI server only")
    print("3. Exit")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        print("🐳 Starting services with Docker Compose...")
        try:
            subprocess.run(["docker-compose", "up", "-d"], check=True)
            print("✅ Services started successfully!")
            print("📊 Services available at:")
            print("   - FastAPI: http://localhost:8000")
            print("   - API Docs: http://localhost:8000/docs")
            print("   - PostgreSQL: localhost:5432")
            print("   - MongoDB: localhost:27017")
            print("   - Redis: localhost:6379")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to start services: {e}")
        except FileNotFoundError:
            print("❌ Docker Compose not found. Please install Docker and Docker Compose.")
    
    elif choice == "2":
        print("🚀 Starting FastAPI server...")
        try:
            os.chdir("backend")
            subprocess.run([sys.executable, "-m", "uvicorn", "app.main:app", 
                          "--host", "0.0.0.0", "--port", "8000", "--reload"])
        except KeyboardInterrupt:
            print("\n👋 Server stopped")
        except Exception as e:
            print(f"❌ Failed to start server: {e}")
    
    elif choice == "3":
        print("👋 Goodbye!")
        sys.exit(0)
    
    else:
        print("❌ Invalid choice")

def main():
    """Main function"""
    print_banner()
    
    # Check prerequisites
    check_python_version()
    
    if not check_dependencies():
        print("\n💡 To install dependencies, run:")
        print("   pip install -r backend/requirements.txt")
        return
    
    # Setup environment
    setup_environment()
    
    # Run tests
    run_tests()
    
    # Start services
    start_services()

if __name__ == "__main__":
    main() 