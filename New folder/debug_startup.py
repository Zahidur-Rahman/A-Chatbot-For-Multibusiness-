#!/usr/bin/env python3
"""
Debug script to identify startup issues with the FastAPI application.
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

# Add the backend directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

def setup_logging():
    """Setup logging for debugging"""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

async def test_imports():
    """Test if all required modules can be imported"""
    print("🔍 Testing imports...")
    
    try:
        from backend.app.config import get_settings
        print("✅ Config module imported successfully")
    except Exception as e:
        print(f"❌ Failed to import config: {e}")
        return False
    
    try:
        from backend.app.services.mongodb_service import get_mongodb_service
        print("✅ MongoDB service imported successfully")
    except Exception as e:
        print(f"❌ Failed to import MongoDB service: {e}")
        return False
    
    try:
        from backend.app.services.vector_search import FaissVectorSearchService
        print("✅ Vector search service imported successfully")
    except Exception as e:
        print(f"❌ Failed to import vector search service: {e}")
        return False
    
    try:
        from backend.app.services.mistral_llm_service import MistralLLMService
        print("✅ Mistral LLM service imported successfully")
    except Exception as e:
        print(f"❌ Failed to import Mistral LLM service: {e}")
        return False
    
    return True

async def test_config():
    """Test configuration loading"""
    print("\n🔍 Testing configuration...")
    
    try:
        from backend.app.config import get_settings
        settings = get_settings()
        print("✅ Settings loaded successfully")
        print(f"   Debug mode: {settings.server.debug}")
        print(f"   MongoDB URI: {settings.mongodb.uri}")
        print(f"   Redis host: {settings.redis.host}")
        print(f"   Mistral model: {settings.mistral.model}")
        return True
    except Exception as e:
        print(f"❌ Failed to load settings: {e}")
        return False

async def test_mongodb_connection():
    """Test MongoDB connection"""
    print("\n🔍 Testing MongoDB connection...")
    
    try:
        from backend.app.services.mongodb_service import get_mongodb_service
        mongo_service = await get_mongodb_service()
        print("✅ MongoDB connection successful")
        return True
    except Exception as e:
        print(f"❌ Failed to connect to MongoDB: {e}")
        return False

async def test_vector_search():
    """Test vector search service initialization"""
    print("\n🔍 Testing vector search service...")
    
    try:
        from backend.app.services.vector_search import FaissVectorSearchService
        vector_service = FaissVectorSearchService()
        print("✅ Vector search service initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize vector search service: {e}")
        return False

async def test_fastapi_app():
    """Test FastAPI app creation"""
    print("\n🔍 Testing FastAPI app creation...")
    
    try:
        from backend.app.main import create_app
        app = create_app()
        print("✅ FastAPI app created successfully")
        print(f"   Docs URL: {app.docs_url}")
        print(f"   Redoc URL: {app.redoc_url}")
        return True
    except Exception as e:
        print(f"❌ Failed to create FastAPI app: {e}")
        return False

async def main():
    """Main debug function"""
    print("🚀 FastAPI Startup Debug Tool")
    print("=" * 50)
    
    setup_logging()
    
    # Test imports
    if not await test_imports():
        print("\n❌ Import tests failed. Please check your dependencies.")
        return
    
    # Test configuration
    if not await test_config():
        print("\n❌ Configuration test failed. Please check your .env file.")
        return
    
    # Test MongoDB connection
    if not await test_mongodb_connection():
        print("\n❌ MongoDB connection failed. Please check your MongoDB setup.")
        return
    
    # Test vector search
    if not await test_vector_search():
        print("\n❌ Vector search test failed. Please check your dependencies.")
        return
    
    # Test FastAPI app
    if not await test_fastapi_app():
        print("\n❌ FastAPI app creation failed.")
        return
    
    print("\n✅ All tests passed! Your application should start successfully.")
    print("\n💡 To start the server, run:")
    print("   cd backend")
    print("   python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload")

if __name__ == "__main__":
    asyncio.run(main()) 