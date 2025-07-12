"""
Main FastAPI application for the Multi-Business Conversational Chatbot.
"""

from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from contextlib import asynccontextmanager
import logging
import sys
from typing import Dict, Any, List, Optional
import re
import os
import json
from datetime import datetime, timezone, timedelta
from pydantic import BaseModel, Field, validator
from bson import ObjectId
from fastapi.responses import JSONResponse

from backend.app.config import get_settings, Settings
from backend.app.auth.routes import router as auth_router
from backend.app.services.business_service import router as business_admin_router
from backend.app.services.mistral_llm_service import MistralLLMService
from backend.app.services.vector_search import FaissVectorSearchService
from backend.app.services.mongodb_service import MongoDBService, get_mongodb_service
from backend.app.models.conversation import ConversationSession
from backend.app.mcp.mcp_client import MCPClient
from backend.app.auth.jwt_handler import get_current_user, require_business_access, require_admin
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
import redis.asyncio as aioredis
from backend.app.models.business import BusinessConfig, BusinessConfigCreate, BusinessConfigUpdate
from backend.app.models.user import User, UserCreate, UserUpdate
from backend.app.utils import hash_password
from backend.app.auth.routes import require_role

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

MCP_SERVER_PATH = os.path.join(os.path.dirname(__file__), "mcp", "server_enhanced.py")
mcp_client = MCPClient(MCP_SERVER_PATH)

# Global service instances
vector_search_service = FaissVectorSearchService()
llm_service = MistralLLMService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Multi-Business Conversational Chatbot...")

    logger.info("Connecting to Redis...")
    settings = get_settings()
    redis_url = f"redis://{settings.redis.host}:{settings.redis.port}/0"
    redis = await aioredis.from_url(redis_url, encoding="utf8", decode_responses=True)
    await FastAPILimiter.init(redis)
    logger.info("Connected to Redis.")

    try:
        logger.info("Connecting to MongoDB...")
        mongo_service = await get_mongodb_service()
        logger.info("Connected to MongoDB.")
        business_configs = await mongo_service.get_all_business_configs()
        business_ids = [b.business_id for b in business_configs]
        logger.info(f"Found business IDs: {business_ids}")
        for business_id in business_ids:
            logger.info(f"Indexing schemas for business: {business_id}")
            await vector_search_service.index_business_schemas(business_id)
            logger.info(f"Finished indexing for business: {business_id}")
    except Exception as e:
        logger.error(f"Startup failed: {e}", exc_info=True)
        # Don't raise the exception - let the app start even if indexing fails
        logger.warning("Continuing startup despite indexing errors")

    logger.info("Application startup complete")
    yield  # Don't forget this!

def create_app() -> FastAPI:
    """Create and configure the FastAPI application"""
    settings = get_settings()
    
    app = FastAPI(
        title="Multi-Business Conversational Chatbot",
        description="A production-ready, dynamic multi-business conversational chatbot with PostgreSQL integration, vector-based schema discovery, and LangChain conversational AI.",
        version="1.0.0",
        docs_url="/docs",  # Always enable docs for debugging
        redoc_url="/redoc",  # Always enable redoc for debugging
        lifespan=lifespan
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.server.allowed_origins,
        allow_credentials=True,
        allow_methods=settings.server.allowed_methods,
        allow_headers=settings.server.allowed_headers,
    )
    
    # Add trusted host middleware for production
    if not settings.server.debug:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["*"]  # Configure based on your domain
        )
    
    # Include authentication routes
    app.include_router(auth_router)
    # Include business/user admin routes
    app.include_router(business_admin_router)
    
    return app

# Create the application instance
app = create_app()

# Connect to MongoDB at startup
# @app.on_event("startup")
# async def startup_event():
#     # Get all business IDs from MongoDB
#     mongo_service = get_mongodb_service()
#     business_configs = await mongo_service.get_all_business_configs()
#     business_ids = [b.business_id for b in business_configs]
#     # Index schemas for each business using the global instance
#     for business_id in business_ids:
#         await vector_search_service.index_business_schemas(business_id)

# =============================================================================
# HEALTH CHECK ENDPOINTS
# =============================================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Multi-Business Conversational Chatbot API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "multi-business-chatbot",
        "version": "1.0.0"
    }

@app.get("/health/detailed")
async def detailed_health_check(settings: Settings = Depends(get_settings)):
    """Detailed health check with service status"""
    health_status = {
        "status": "healthy",
        "service": "multi-business-chatbot",
        "version": "1.0.0",
        "features": {
            "multi_business": settings.features.enable_multi_business,
            "vector_search": settings.features.enable_vector_search,
            "schema_discovery": settings.features.enable_schema_discovery,
            "conversation_memory": settings.features.enable_conversation_memory,
            "analytics": settings.features.enable_analytics
        },
        "businesses": settings.business_ids,
        "services": {
            "database": "unknown",  # TODO: Add database health check
            "mongodb": "unknown",   # TODO: Add MongoDB health check
            "redis": "unknown",     # TODO: Add Redis health check
            "mcp_server": "unknown" # TODO: Add MCP server health check
        }
    }
    
    return health_status

# =============================================================================
# BUSINESS MANAGEMENT ENDPOINTS
# =============================================================================

@app.get("/businesses")
async def list_businesses(mongo_service: MongoDBService = Depends(get_mongodb_service)):
    businesses = await mongo_service.get_all_business_configs()
    return {
        "businesses": [b.business_id for b in businesses],
        "count": len(businesses)
    }

@app.get("/businesses/{business_id}/config")
async def get_business_config(business_id: str, mongo_service: MongoDBService = Depends(get_mongodb_service)):
    config = await mongo_service.get_business_config(business_id)
    if not config:
        raise HTTPException(status_code=404, detail=f"Business '{business_id}' not found")
    return {
        "business_id": business_id,
        "host": config.host,
        "database": config.database,
        "port": config.port,
        "user": config.user,
        # Don't expose password in response
    }

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def clean_session_id(session_id: str) -> str:
    """Clean session ID by removing escaped quotes and normalizing format."""
    if not session_id:
        return session_id
    
    logger.info(f"[CleanSessionID] Original session_id: '{session_id}'")
    
    # Remove escaped quotes that Swagger UI might add
    cleaned = session_id.replace('\\"', '').replace('"', '')
    
    logger.info(f"[CleanSessionID] After quote removal: '{cleaned}'")
    
    # Ensure it starts with 'sess_'
    if not cleaned.startswith('sess_'):
        logger.warning(f"[CleanSessionID] Session ID doesn't start with 'sess_': '{cleaned}'")
        return session_id  # Return original if it doesn't match expected format
    
    logger.info(f"[CleanSessionID] Final cleaned session_id: '{cleaned}'")
    return cleaned

# =============================================================================
# CONVERSATION ENDPOINTS (Placeholder)
# =============================================================================

@app.post("/conversations")
async def create_conversation():
    """Create a new conversation session"""
    # TODO: Implement conversation creation
    return {
        "message": "Conversation creation endpoint - to be implemented",
        "status": "placeholder"
    }

@app.post("/conversations/{conversation_id}/messages")
async def send_message(conversation_id: str):
    """Send a message in a conversation"""
    # TODO: Implement message handling
    return {
        "message": "Message handling endpoint - to be implemented",
        "conversation_id": conversation_id,
        "status": "placeholder"
    }

@app.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get conversation details"""
    # TODO: Implement conversation retrieval
    return {
        "message": "Conversation retrieval endpoint - to be implemented",
        "conversation_id": conversation_id,
        "status": "placeholder"
    }

# =============================================================================
# SCHEMA DISCOVERY ENDPOINTS (Placeholder)
# =============================================================================

@app.get("/businesses/{business_id}/schemas")
async def list_schemas(business_id: str):
    """List all schemas for a business"""
    # TODO: Implement schema listing
    return {
        "message": "Schema listing endpoint - to be implemented",
        "business_id": business_id,
        "status": "placeholder"
    }

@app.get("/businesses/{business_id}/schemas/{table_name}")
async def get_schema(business_id: str, table_name: str):
    """Get schema for a specific table"""
    # TODO: Implement schema retrieval
    return {
        "message": "Schema retrieval endpoint - to be implemented",
        "business_id": business_id,
        "table_name": table_name,
        "status": "placeholder"
    }

# =============================================================================
# ADMIN-ONLY BUSINESS & USER MANAGEMENT ENDPOINTS
# =============================================================================

from pydantic import BaseModel
from typing import Dict, Any

class BusinessCreateRequest(BaseModel):
    business_id: str
    config: Dict[str, Any]

@app.post("/admin/businesses", dependencies=[Depends(require_admin)])
async def admin_add_business(
    request: BusinessCreateRequest,
    current_user: dict = Depends(get_current_user),
    mongo_service: MongoDBService = Depends(get_mongodb_service)
):
    logger.info(f"Adding business: {request.business_id}")
    logger.info(f"Adding businessx: {request.business_id}")
    success = await mongo_service.add_business(request.business_id, request.config)
    logger.info(f"Success: {success}")
    
    if not success:
        raise HTTPException(status_code=400, detail="Failed to add business")
    return {"message": f"Business '{request.business_id}' added", "status": "success"}

@app.delete("/admin/businesses/{business_id}", dependencies=[Depends(require_admin)])
async def admin_remove_business(business_id: str, current_user: dict = Depends(get_current_user), mongo_service: MongoDBService = Depends(get_mongodb_service)):
    """Admin: Remove a business"""
    success = await mongo_service.remove_business(business_id)
    if not success:
        raise HTTPException(status_code=404, detail="Business not found or could not be removed")
    return {"message": f"Business '{business_id}' removed", "status": "success"}

@app.post("/admin/users", dependencies=[Depends(require_admin)])
async def admin_add_user(user: Dict[str, Any], current_user: dict = Depends(get_current_user), mongo_service: MongoDBService = Depends(get_mongodb_service)):
    """Admin: Add a new user and assign to business(es)"""
    success = await mongo_service.add_user(user)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to add user")
    return {"message": f"User '{user.get('username')}' added", "status": "success"}

@app.delete("/admin/users/{user_id}", dependencies=[Depends(require_admin)])
async def admin_remove_user(user_id: str, current_user: dict = Depends(get_current_user), mongo_service: MongoDBService = Depends(get_mongodb_service)):
    """Admin: Remove a user"""
    success = await mongo_service.remove_user(user_id)
    if not success:
        raise HTTPException(status_code=404, detail="User not found or could not be removed")
    return {"message": f"User '{user_id}' removed", "status": "success"}

@app.post("/admin/users/{user_id}/assign", dependencies=[Depends(require_admin)])
async def admin_assign_user_to_business(user_id: str, business_ids: List[str], current_user: dict = Depends(get_current_user), mongo_service: MongoDBService = Depends(get_mongodb_service)):
    """Admin: Assign user to one or more businesses"""
    success = await mongo_service.assign_user_to_businesses(user_id, business_ids)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to assign user to businesses")
    return {"message": f"User '{user_id}' assigned to businesses {business_ids}", "status": "success"}

@app.post("/admin/businesses/{business_id}/schemas", dependencies=[Depends(require_admin)])
async def admin_add_or_update_business_schema(business_id: str, schema: Dict[str, Any], current_user: dict = Depends(get_current_user), mongo_service: MongoDBService = Depends(get_mongodb_service)):
    """Admin: Add or update a business schema"""
    success = await mongo_service.add_or_update_business_schema(business_id, schema)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to add/update schema")
    # AUTOMATION: Re-index vector embeddings after schema change using the global instance
    await vector_search_service.index_business_schemas(business_id)
    return {"message": f"Schema for business '{business_id}' added/updated and vector index refreshed", "status": "success"}

# =============================================================================
# ERROR HANDLERS
# =============================================================================

@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors"""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not found",
            "message": "The requested resource was not found",
            "status_code": 404
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "status_code": 500
        }
    )

# =============================================================================
# DEVELOPMENT ENDPOINTS (only in debug mode)
# =============================================================================

@app.get("/debug/config")
async def debug_config(mongo_service: MongoDBService = Depends(get_mongodb_service)):
    businesses = await mongo_service.get_all_business_configs()
    return {
        "businesses": [b.business_id for b in businesses],
        "count": len(businesses)
    }

@app.get("/debug/schemas/{business_id}")
async def debug_schemas(business_id: str, mongo_service: MongoDBService = Depends(get_mongodb_service)):
    """Debug endpoint to check available schemas for a business"""
    schemas = await mongo_service.get_business_schemas(business_id)
    return {
        "business_id": business_id,
        "schemas": [
            {
                "table_name": s.table_name,
                "description": s.schema_description,
                "columns": [{"name": c.name, "type": c.type, "description": c.description} for c in s.columns]
            }
            for s in schemas
        ],
        "count": len(schemas)
    }

@app.get("/debug/vector-search/{business_id}")
async def debug_vector_search(
    business_id: str, 
    query: str, 
    mongo_service: MongoDBService = Depends(get_mongodb_service)
):
    """Debug endpoint to test vector search for a business"""
    results = await vector_search_service.search_schemas(business_id, query, top_k=5)
    return {
        "business_id": business_id,
        "query": query,
        "results": [
            {
                "table_name": r.get("table_name"),
                "description": r.get("schema_description"),
                "columns": [c.get("name") for c in r.get("columns", [])]
            }
            for r in results
        ],
        "count": len(results)
    }

@app.get("/debug/conversations/user/{user_id}")
async def debug_user_conversations(
    user_id: str,
    business_id: str,
    mongo_service: MongoDBService = Depends(get_mongodb_service)
):
    """Debug endpoint to list all conversations for a user"""
    try:
        conversations = await mongo_service.get_user_conversations(user_id, business_id, limit=10)
        return {
            "user_id": user_id,
            "business_id": business_id,
            "conversation_count": len(conversations),
            "conversations": [
                {
                    "session_id": conv.session_id,
                    "message_count": len(conv.conversation_memory.messages) if conv.conversation_memory else 0,
                    "last_activity": conv.last_activity,
                    "status": conv.status
                }
                for conv in conversations
            ]
        }
    except Exception as e:
        logger.error(f"Error retrieving user conversations: {e}")
        return {"error": str(e)}

@app.get("/debug/session-info/{session_id}")
async def debug_session_info(
    session_id: str,
    mongo_service: MongoDBService = Depends(get_mongodb_service)
):
    """Debug endpoint to show session ID variations and find matching sessions"""
    # Clean session ID to handle Swagger UI escaping
    original_session_id = session_id
    cleaned_session_id = clean_session_id(session_id)
    
    # Try different variations
    variations = [
        session_id,
        f'"{session_id}"',
        f'\\"{session_id}\\"',
        cleaned_session_id
    ]
    
    results = {}
    for var in variations:
        try:
            session = await mongo_service.get_conversation_session(var)
            if session:
                results[var] = {
                    "found": True,
                    "message_count": len(session.conversation_memory.messages) if session.conversation_memory and session.conversation_memory.messages else 0,
                    "session_id_in_db": session.session_id
                }
            else:
                results[var] = {"found": False}
        except Exception as e:
            results[var] = {"found": False, "error": str(e)}
    
    return {
        "original_session_id": original_session_id,
        "cleaned_session_id": cleaned_session_id,
        "variations_tested": variations,
        "results": results
    }

@app.get("/debug/cleanup-sessions")
async def debug_cleanup_sessions(
    mongo_service: MongoDBService = Depends(get_mongodb_service)
):
    """Debug endpoint to clean up sessions with escaped quotes"""
    try:
        # Find all sessions with escaped quotes
        cursor = mongo_service._collections['conversation_sessions'].find({
            "session_id": {"$regex": r'^".*"$'}
        })
        
        migrated_sessions = []
        async for doc in cursor:
            old_session_id = doc['session_id']
            new_session_id = old_session_id.replace('"', '').replace('\\"', '')
            
            if old_session_id != new_session_id:
                success = await mongo_service.migrate_session_id_format(old_session_id, new_session_id)
                if success:
                    migrated_sessions.append({
                        "old_id": old_session_id,
                        "new_id": new_session_id
                    })
        
        return {
            "message": f"Migrated {len(migrated_sessions)} sessions",
            "migrated_sessions": migrated_sessions
        }
    except Exception as e:
        return {"error": f"Error cleaning up sessions: {str(e)}"}

@app.get("/debug/migrate-escaped-sessions")
async def debug_migrate_escaped_sessions(
    mongo_service: MongoDBService = Depends(get_mongodb_service)
):
    """Debug endpoint to migrate all sessions with escaped quotes to clean format"""
    try:
        # Find all sessions with escaped quotes
        escaped_sessions = []
        cursor = mongo_service._collections['conversation_sessions'].find({
            "session_id": {"$regex": r'^".*"$'}  # Session IDs wrapped in quotes
        })
        
        async for doc in cursor:
            escaped_sessions.append(doc)
        
        logger.info(f"[Debug] Found {len(escaped_sessions)} sessions with escaped quotes")
        
        migrated_count = 0
        for doc in escaped_sessions:
            old_session_id = doc['session_id']
            # Clean the session ID
            new_session_id = old_session_id.replace('"', '').replace('\\"', '')
            
            if new_session_id != old_session_id:
                success = await mongo_service.migrate_session_id_format(old_session_id, new_session_id)
                if success:
                    migrated_count += 1
                    logger.info(f"[Debug] Migrated session from '{old_session_id}' to '{new_session_id}'")
        
        return {
            "message": "Session migration completed",
            "total_escaped_sessions": len(escaped_sessions),
            "migrated_sessions": migrated_count,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Session migration failed: {e}")
        return {
            "message": f"Session migration failed: {str(e)}",
            "status": "error"
        }

@app.get("/debug/list-all-sessions")
async def debug_list_all_sessions(
    mongo_service: MongoDBService = Depends(get_mongodb_service)
):
    """Debug endpoint to list all sessions and their formats"""
    try:
        cursor = mongo_service._collections['conversation_sessions'].find({})
        
        sessions = []
        async for doc in cursor:
            session_info = {
                "session_id": doc.get('session_id'),
                "user_id": doc.get('user_id'),
                "business_id": doc.get('business_id'),
                "has_escaped_quotes": doc.get('session_id', '').startswith('"') and doc.get('session_id', '').endswith('"'),
                "message_count": len(doc.get('conversation_memory', {}).get('messages', [])),
                "created_at": doc.get('created_at'),
                "last_activity": doc.get('last_activity')
            }
            sessions.append(session_info)
        
        return {
            "total_sessions": len(sessions),
            "sessions": sessions,
            "escaped_sessions": [s for s in sessions if s['has_escaped_quotes']],
            "clean_sessions": [s for s in sessions if not s['has_escaped_quotes']]
        }
    except Exception as e:
        logger.error(f"Failed to list sessions: {e}")
        return {
            "message": f"Failed to list sessions: {str(e)}",
            "status": "error"
        }

@app.get("/debug/conversations/{session_id}")
async def debug_conversation(
    session_id: str,
    mongo_service: MongoDBService = Depends(get_mongodb_service)
):
    # Clean session ID to handle Swagger UI escaping
    original_session_id = session_id
    session_id = clean_session_id(session_id)
    if original_session_id != session_id:
        logger.info(f"[Debug] Cleaned session_id from '{original_session_id}' to '{session_id}'")
    """Debug endpoint to check conversation storage"""
    try:
        session = await mongo_service.get_conversation_session(session_id)
        if session:
            return {
                "session_id": session.session_id,
                "user_id": session.user_id,
                "business_id": session.business_id,
                "status": session.status,
                "created_at": session.created_at,
                "last_activity": session.last_activity,
                "expires_at": session.expires_at,
                "conversation_memory": {
                    "message_count": len(session.conversation_memory.messages) if session.conversation_memory and session.conversation_memory.messages else 0,
                    "messages": [
                        {
                            "role": msg.role,
                            "content": msg.content[:100] + "..." if len(msg.content) > 100 else msg.content,
                            "timestamp": msg.timestamp
                        }
                        for msg in (session.conversation_memory.messages if session.conversation_memory and session.conversation_memory.messages else [])
                    ]
                },
                "cached_schema_context": len(session.cached_schema_context) if session.cached_schema_context else 0
            }
        else:
            return {"error": f"Session '{session_id}' not found"}
    except Exception as e:
        return {"error": f"Error retrieving session: {str(e)}"}

@app.get("/debug/session/{session_id}")
async def debug_session(
    session_id: str,
    mongo_service: MongoDBService = Depends(get_mongodb_service)
):
    """Debug endpoint to check the current state of a session"""
    # Clean session ID to handle Swagger UI escaping
    original_session_id = session_id
    session_id = clean_session_id(session_id)
    if original_session_id != session_id:
        logger.info(f"[Debug] Cleaned session_id from '{original_session_id}' to '{session_id}'")
    
    try:
        session = await mongo_service.get_conversation_session(session_id)
        if session:
            return {
                "session_id": session.session_id,
                "user_id": session.user_id,
                "business_id": session.business_id,
                "status": session.status,
                "created_at": session.created_at,
                "last_activity": session.last_activity,
                "expires_at": session.expires_at,
                "conversation_memory": {
                    "message_count": len(session.conversation_memory.messages) if session.conversation_memory and session.conversation_memory.messages else 0,
                    "messages": [
                        {
                            "role": msg.role,
                            "content": msg.content[:100] + "..." if len(msg.content) > 100 else msg.content,
                            "timestamp": msg.timestamp
                        }
                        for msg in (session.conversation_memory.messages if session.conversation_memory and session.conversation_memory.messages else [])
                    ]
                },
                "cached_schema_context": len(session.cached_schema_context) if session.cached_schema_context else 0
            }
        else:
            return {"error": f"Session '{session_id}' not found"}
    except Exception as e:
        return {"error": f"Error retrieving session: {str(e)}"}

@app.get("/debug/generate-session-id")
async def debug_generate_session_id(
    current_user: dict = Depends(get_current_user)
):
    """Debug endpoint to generate a clean session ID for testing"""
    user_id = current_user["user_id"]
    business_id = current_user["business_id"]
    session_id = f"sess_{user_id}_{business_id}_{int(datetime.now().timestamp())}"
    return {
        "session_id": session_id,
        "user_id": user_id,
        "business_id": business_id,
        "timestamp": int(datetime.now().timestamp()),
        "note": "Use this session_id in your chat requests to maintain conversation context"
    }

@app.get("/debug/mcp-test/{business_id}")
async def debug_mcp_test(
    business_id: str, 
    query: str = "SELECT 1 as test",
    mongo_service: MongoDBService = Depends(get_mongodb_service)
):
    """Debug endpoint to test MCP connection directly"""
    try:
        result = await mcp_client.execute_query(query, business_id)
        return {
            "business_id": business_id,
            "query": query,
            "result": result,
            "status": "success"
        }
    except Exception as e:
        return {
            "business_id": business_id,
            "query": query,
            "error": str(e),
            "status": "error"
        }

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None  # Made optional
    business_id: Optional[str] = None  # Optional business_id for admin users
    conversation_history: Optional[List[ChatMessage]] = []
    refresh_schema_context: Optional[bool] = False

class ChatResponse(BaseModel):
    response: str
    session_id: Optional[str] = None
    
    @validator('session_id', pre=True)
    def clean_session_id(cls, v):
        if v is None:
            return v
        # Remove escaped quotes that might be present
        cleaned = v.replace('\\"', '').replace('"', '')
        return cleaned
    
    class Config:
        # Ensure proper JSON serialization
        json_encoders = {
            str: lambda v: v.replace('\\"', '').replace('"', '') if isinstance(v, str) else v
        }

llm_service = MistralLLMService()
# mongo_service = mongodb_service # This line is no longer needed

# Add this function to extract user_id from the request state (assuming authentication middleware sets it)
async def get_user_id(request: Request):
    # If you use request.state.user, adjust as needed
    user = getattr(request.state, 'user', None)
    if user and hasattr(user, 'user_id'):
        return str(user.user_id)
    # Fallback: use IP if user_id is not available
    return request.client.host

async def is_database_related_query_dynamic(message: str, business_id: str, vector_search_service) -> bool:
    """
    Fully LLM-based dynamic classification - no hardcoded rules.
    Uses the LLM to intelligently determine if a query should generate SQL.
    """
    try:
        # 1. Get schema context for the query
        schema_results = await vector_search_service.search_schemas(business_id, message, top_k=3)
        
        # 2. Create schema context for the LLM
        schema_context = ""
        if schema_results and len(schema_results) > 0:
            schema_context = "Available database schemas:\n"
            for schema in schema_results:
                schema_context += f"- {schema.get('table_name', 'Unknown')}: {schema.get('schema_description', 'No description')}\n"
        else:
            schema_context = "No relevant database schemas found."
        
        # 3. Use LLM to classify the query
        classification_prompt = f"""
You are an intelligent query classifier for a business chatbot system.

AVAILABLE DATABASE SCHEMAS:
{schema_context}

USER QUERY: "{message}"

TASK: Determine if this query should generate a SQL database query or be treated as general conversation.

CLASSIFICATION RULES:
- If the user is asking for specific data, records, information, or facts that could be retrieved from the database → CLASSIFY AS DATABASE QUERY
- If the user is asking for general help, explanations, opinions, or casual conversation → CLASSIFY AS GENERAL CONVERSATION
- If the user mentions specific names, IDs, or entities that could be looked up → CLASSIFY AS DATABASE QUERY
- If the user is asking "how to" questions or seeking advice → CLASSIFY AS GENERAL CONVERSATION
- If the user asks "about" someone/something and relevant database tables exist → CLASSIFY AS DATABASE QUERY
- If the user asks for information that could be found in customer, menu, or other business data → CLASSIFY AS DATABASE QUERY

EXAMPLES:
- "give me all information about Zahid" → DATABASE_QUERY (looking for specific customer data)
- "About Zahid" → DATABASE_QUERY (asking about specific person, likely customer data)
- "show me the menu" → DATABASE_QUERY (looking for menu data)
- "how do I reset my password?" → GENERAL CONVERSATION (seeking instructions)
- "what's the weather like?" → GENERAL CONVERSATION (not database related)
- "find customer John Smith" → DATABASE_QUERY (looking for specific customer)
- "can you help me?" → GENERAL CONVERSATION (general assistance request)
- "tell me about the restaurant" → DATABASE_QUERY (asking about business data)
- "hello" → GENERAL CONVERSATION (greeting)

RESPONSE FORMAT: Respond with ONLY "DATABASE_QUERY" or "GENERAL_CONVERSATION" (no other text).

CLASSIFICATION:"""

        # Use the LLM to classify
        llm_service = MistralLLMService()
        classification_response = await llm_service.chat([
            {"role": "system", "content": classification_prompt},
            {"role": "user", "content": message}
        ])
        
        # Parse the response
        classification = classification_response.strip().upper()
        is_database_query = "DATABASE_QUERY" in classification
        
        logger.info(f"[DynamicClassifier] Query: '{message}' | LLM Classification: {classification} | DB Query: {is_database_query}")
        logger.info(f"[DynamicClassifier] Schema matches: {len(schema_results) if schema_results else 0}")
        
        return is_database_query
        
    except Exception as e:
        logger.error(f"[DynamicClassifier] Error in LLM-based classification: {e}")
        # Conservative fallback - if LLM fails, check if we have schema matches
        try:
            schema_results = await vector_search_service.search_schemas(business_id, message, top_k=3)
            fallback_result = schema_results and len(schema_results) > 0
            logger.info(f"[DynamicClassifier] Using fallback classification: {fallback_result}")
            return fallback_result
        except Exception as fallback_error:
            logger.error(f"[DynamicClassifier] Fallback classification also failed: {fallback_error}")
            return False

@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(RateLimiter(times=10, seconds=60, identifier=get_user_id))])
async def chat_endpoint(
    request: ChatRequest,
    current_user: dict = Depends(get_current_user),
    mongo_service: MongoDBService = Depends(get_mongodb_service)
):
    """
    Chat endpoint for conversational AI with database query capabilities.
    
    Session Management:
    - If session_id is not provided, a new session will be created automatically
    - To maintain conversation context, reuse the same session_id in subsequent requests
    - Session IDs are automatically cleaned to handle Swagger UI escaping
    - Use /debug/generate-session-id to get a clean session ID for testing
    
    Database Queries:
    - The system automatically detects database-related queries
    - Only SELECT queries are allowed for security
    - Results are formatted as tables for easy reading
    - Conversation history is maintained within the session
    """
    user_id = current_user["user_id"]
    role = current_user["role"]
    
    # Handle admin users with business_id: "all"
    if current_user["business_id"] == "all" and role == "admin":
        # For admin users with "all" access, business_id is optional
        # If provided, validate it exists. If not provided, we'll handle it later based on query type
        if request.business_id:
            business_id = request.business_id
            # Validate that the specified business exists
            try:
                await mongo_service.get_business_config(business_id)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Business '{business_id}' not found"
                )
        else:
            # No business_id provided - we'll determine if it's needed based on the query
            business_id = None
    else:
        business_id = current_user["business_id"]
    
    # Only check business access if we have a specific business_id
    if business_id:
        await require_business_access(user_id, business_id)
    
    # Helper function to get storage business_id (for admin users without business_id, use "admin")
    def get_storage_business_id():
        return business_id if business_id else "admin"
    
    # Auto-generate session_id if not provided
    session_id = request.session_id
    logger.info(f"[Chat] Original request session_id: '{session_id}'")
    
    if not session_id:
        # Generate clean session ID without any escaping issues
        # For admin users without business_id, use "admin" as business identifier
        business_identifier = business_id if business_id else "admin"
        session_id = f"sess_{user_id}_{business_identifier}_{int(datetime.now().timestamp())}"
        logger.info(f"[Chat] Auto-generated session_id: {session_id}")
    else:
        # Clean session ID to handle Swagger UI escaping
        original_session_id = session_id
        session_id = clean_session_id(session_id)
        if original_session_id != session_id:
            logger.info(f"[Chat] Cleaned session_id from '{original_session_id}' to '{session_id}'")
    
    # Store cleaned session_id for response
    cleaned_session_id = session_id
    logger.info(f"[Chat] Final cleaned_session_id for response: '{cleaned_session_id}'")
    
    # Try to get session with multiple format attempts
    session: Optional[ConversationSession] = None
    
    # First try with the cleaned session_id
    session = await mongo_service.get_conversation_session(session_id)
    logger.info(f"[Chat] Session lookup for cleaned '{session_id}': {'Found' if session else 'Not found'}")
    
    # If not found, try with escaped quotes (common in Swagger UI)
    if not session:
        escaped_session_id = f'"{session_id}"'
        session = await mongo_service.get_conversation_session(escaped_session_id)
        logger.info(f"[Chat] Session lookup for escaped '{escaped_session_id}': {'Found' if session else 'Not found'}")
        
        # If found with escaped format, migrate it to clean format
        if session:
            logger.info(f"[Chat] Found session with escaped format, migrating from '{session.session_id}' to '{session_id}'")
            await mongo_service.migrate_session_id_format(session.session_id, session_id)
            # Update the session object to use the clean ID
            session.session_id = session_id
    
    # If still not found, try with double escaped quotes
    if not session:
        double_escaped_session_id = f'\\"{session_id}\\"'
        session = await mongo_service.get_conversation_session(double_escaped_session_id)
        logger.info(f"[Chat] Session lookup for double escaped '{double_escaped_session_id}': {'Found' if session else 'Not found'}")
        
        # If found with double escaped format, migrate it to clean format
        if session:
            logger.info(f"[Chat] Found session with double escaped format, migrating from '{session.session_id}' to '{session_id}'")
            await mongo_service.migrate_session_id_format(session.session_id, session_id)
            # Update the session object to use the clean ID
            session.session_id = session_id
    
    if session:
        logger.info(f"[Chat] Session has {len(session.conversation_memory.messages) if session.conversation_memory else 0} messages")
    schema_context = None
    if session and not request.refresh_schema_context:
        schema_context = session.cached_schema_context
    if not schema_context:
        # Only search schemas if we have a business_id
        if business_id:
            schema_context = await vector_search_service.search_schemas(business_id, request.message, top_k=5)
            logger.info(f"[Chat] schema_context for message '{request.message}': {len(schema_context) if schema_context else 0} schemas found")
            
            if not schema_context:
                logger.warning(f"[Chat] No schema context found for business '{business_id}' and query '{request.message}'")
                # Try to get all schemas as fallback
                all_schemas = await mongo_service.get_business_schemas(business_id)
                if all_schemas:
                    schema_context = [s.dict() for s in all_schemas[:3]]  # Take first 3 schemas
                    logger.info(f"[Chat] Using fallback: {len(schema_context)} schemas from business '{business_id}'")
        else:
            # No business_id - this is a general conversation for admin
            logger.info(f"[Chat] No business_id provided for admin user - treating as general conversation")
            schema_context = []
        
        if session:
            await mongo_service.set_cached_schema_context(session_id, schema_context)
        else:
            from backend.app.models.conversation import ConversationMemory, ConversationSessionCreate
            # Create new session with the existing session_id
            try:
                # For admin users without business_id, use "admin" as business_id
                session_business_id = business_id if business_id else "admin"
                session = await mongo_service.create_conversation_session_with_id(
                    session_id=session_id,
                    user_id=user_id,
                    business_id=session_business_id,
                    expires_at=datetime.now(timezone.utc) + timedelta(hours=24)
                )
                logger.info(f"[Chat] Created new session with existing session_id: {session_id}")
                # Update with cached schema context
                session.cached_schema_context = schema_context
                await mongo_service.update_conversation_session(session)
            except ValueError as e:
                # Session already exists, try to get it again
                logger.warning(f"[Chat] Session creation failed, trying to get existing session: {e}")
                session = await mongo_service.get_conversation_session(session_id)
                if session:
                    session.cached_schema_context = schema_context
                    await mongo_service.update_conversation_session(session)
    # Use conversation memory for context
    settings = get_settings()
    max_context_messages = settings.features.max_conversation_context_messages
    conversation_messages = []
    
    # Get messages from current session
    logger.info(f"[Chat] Checking session conversation memory: session={session is not None}, memory={session.conversation_memory is not None if session else False}")
    if session and session.conversation_memory and session.conversation_memory.messages:
        # Get last N messages from current session
        conversation_messages = session.conversation_memory.messages[-max_context_messages:]
        logger.info(f"[Chat] Using {len(conversation_messages)} messages from current session")
        for i, msg in enumerate(conversation_messages):
            logger.info(f"[Chat] Message {i+1}: {msg.role} - {msg.content[:50]}...")
    else:
        logger.info(f"[Chat] No conversation memory found in session")
        if session and session.conversation_memory:
            logger.info(f"[Chat] Session has conversation_memory but no messages: {len(session.conversation_memory.messages) if session.conversation_memory.messages else 0} messages")
        else:
            logger.info(f"[Chat] Session has no conversation_memory object")
    
            # Optionally get messages from all user sessions if enabled
        if not conversation_messages and settings.features.enable_cross_session_context:
            # Get last N messages from all user sessions
            # For admin users without business_id, use "admin" as business_id
            session_business_id = business_id if business_id else "admin"
            all_user_conversations = await mongo_service.get_user_conversations(user_id, session_business_id, limit=5)
            all_messages = []
            for conv in all_user_conversations:
                if conv.conversation_memory and conv.conversation_memory.messages:
                    all_messages.extend(conv.conversation_memory.messages)
            # Sort by timestamp and get last N
            all_messages.sort(key=lambda x: x.timestamp)
            conversation_messages = all_messages[-max_context_messages:]
            logger.info(f"[Chat] Using {len(conversation_messages)} messages from all user sessions")
    
    # Add new user message
    from backend.app.models.conversation import Message
    user_message = Message(
        role="user",
        content=request.message
    )
    conversation_messages.append(user_message)
    # Use schema_context in prompt (simple example)
    schema_text = ""
    if schema_context:
        for i, schema in enumerate(schema_context):
            schema_text += f"\nTable: {schema.get('table_name', 'Unknown')}\n"
            schema_text += f"Description: {schema.get('schema_description', 'No description')}\n"
            schema_text += "Columns:\n"
            for col in schema.get('columns', []):
                schema_text += f"  - {col.get('name', 'Unknown')}: {col.get('type', 'Unknown')} ({col.get('description', 'No description')})\n"
            if schema.get('relationships'):
                schema_text += "Relationships:\n"
                for rel in schema.get('relationships', []):
                    schema_text += f"  - {rel.get('from_table', 'Unknown')}.{rel.get('from_column', 'Unknown')} -> {rel.get('to_table', 'Unknown')}.{rel.get('to_column', 'Unknown')}\n"
            schema_text += "\n"
    else:
        schema_text = "No relevant schema context found."
    
    logger.info(f"[Chat] schema_text for message '{request.message}': {schema_text}")
    
    # Create system prompt with schema context and conversation history
    system_prompt = (
        f"You are a friendly and helpful AI assistant. You can help with general questions and also access business data when needed.\n\n"
        f"AVAILABLE DATA:\n{schema_text}\n\n"
        f"CONVERSATION HISTORY:\n"
    )
    
    # Add conversation history to system prompt
    for msg in conversation_messages[:-1]:  # Exclude the current user message
        system_prompt += f"{msg.role.upper()}: {msg.content}\n"
    
    system_prompt += (
        f"\nCURRENT USER REQUEST: {request.message}\n\n"
        "INSTRUCTIONS:\n"
        "1. Be friendly and conversational in your responses\n"
        "2. Provide helpful, accurate information based on the available data\n"
        "3. Consider the conversation history for context\n"
        "4. Keep responses concise and user-friendly\n"
        "5. If asked about specific data, provide clear and relevant information\n"
    )
    
    # Filter conversation messages to ensure proper order (no consecutive user messages)
    # Only include conversation history, not the current user message
    conversation_history = conversation_messages[:-1]  # Exclude the current user message
    
    # Ensure proper alternating pattern: user -> assistant -> user -> assistant
    # More aggressive filtering to prevent any consecutive same-role messages
    filtered_messages = []
    last_role = None
    
    logger.info(f"[Chat] DEBUG: Starting general conversation filtering with {len(conversation_history)} messages")
    for i, msg in enumerate(conversation_history):
        logger.info(f"[Chat] DEBUG: General Message {i}: role='{msg.role}', last_role='{last_role}'")
        # Only add if it has a different role than the last message we added
        if msg.role != last_role:
            filtered_messages.append(msg.model_dump(exclude={"metadata"}))
            last_role = msg.role
            logger.info(f"[Chat] DEBUG: Added general message {i} with role '{msg.role}'")
        else:
            logger.info(f"[Chat] DEBUG: Skipped general message {i} with role '{msg.role}' (same as last_role)")
    
    logger.info(f"[Chat] Original conversation history roles: {[msg.role for msg in conversation_history]}")
    logger.info(f"[Chat] Filtered conversation history roles: {[msg.get('role') for msg in filtered_messages]}")
    
    # Send system prompt with conversation history embedded, plus the current user message
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": request.message}
    ]
    # Dynamically classify message based on vector search results
    # For admin users without business_id, treat as general conversation
    if business_id:
        is_db_query = await is_database_related_query_dynamic(request.message, business_id, vector_search_service)
    else:
        # No business_id - this is a general conversation for admin
        is_db_query = False
        logger.info(f"[Chat] Admin user without business_id - treating as general conversation")
    
    if not is_db_query:
        # General conversation - no database query needed
        logger.info(f"[Chat] Treating as general conversation: '{request.message}'")
        logger.info(f"[Chat] Using {len(filtered_messages)} filtered messages for general conversation")
        logger.info(f"[Chat] Filtered message roles: {[msg.get('role') for msg in filtered_messages]}")
        response = await llm_service.chat(messages)
        # Store user and LLM messages in conversation memory
        assistant_message = Message(
            role="assistant",
            content=response
        )
        conversation_messages.append(assistant_message)
        # Persist updated session
        try:
            memory_data = {
                "messages": [msg.dict() for msg in conversation_messages],
                "context": session.conversation_memory.context.dict() if session.conversation_memory.context else {},
                "user_preferences": session.conversation_memory.user_preferences.dict() if session.conversation_memory.user_preferences else {},
                "session_variables": session.conversation_memory.session_variables or {}
            }
            await mongo_service.update_conversation_memory_upsert(cleaned_session_id, memory_data, user_id, get_storage_business_id())
            logger.info(f"[Chat] Successfully stored conversation for session: {cleaned_session_id}")
        except Exception as e:
            logger.error(f"[Chat] Failed to store conversation for session {cleaned_session_id}: {e}")
        return ChatResponse(response=response, session_id=cleaned_session_id)
    else:
        # Database query detected - proceed with SQL generation
        logger.info(f"[Chat] Treating as database query: '{request.message}'")
        
        sql_prompt = (
            "You are an expert SQL assistant for a PostgreSQL database. "
            "Your task is to convert natural language requests into SQL SELECT queries.\n\n"
            f"AVAILABLE DATABASE SCHEMAS:\n{schema_text}\n"
            f"CONVERSATION HISTORY (last {max_context_messages} messages):\n"
        )
        
        # Add conversation history to the prompt
        for msg in conversation_messages[:-1]:  # Exclude the current user message
            sql_prompt += f"{msg.role.upper()}: {msg.content}\n"
    
        # Add context from previous conversation to help with follow-up questions
        context_hint = ""
        if len(conversation_messages) > 1:
            # Look for previous mentions of names or entities
            previous_messages = [msg.content for msg in conversation_messages[:-1]]
            context_hint = "\nCONVERSATION CONTEXT: "
            context_hint += "The user has been discussing: " + "; ".join(previous_messages[-2:])  # Last 2 messages
            context_hint += "\n"
    
        sql_prompt += (
            f"\nCURRENT USER REQUEST: {request.message}\n"
            f"{context_hint}\n"
            "INSTRUCTIONS:\n"
            "1. Analyze the user's natural language request\n"
            "2. Identify relevant tables and columns from the schema above\n"
            "3. Generate a single, safe, syntactically correct SQL SELECT query\n"
            "4. Only use SELECT statements - never DROP, DELETE, TRUNCATE, ALTER, CREATE, INSERT, or UPDATE\n"
            "5. Use only the tables and columns provided in the schema context\n"
            "6. Consider the conversation history for context and follow-up questions\n"
            "7. If this is a follow-up question, use context from previous messages to understand what the user is referring to\n\n"
            "OUTPUT FORMAT: Generate ONLY the complete SQL query, no explanations, no markdown, no code blocks, no prefixes.\n"
            "Preserve all SQL clauses including WHERE, ORDER BY, GROUP BY, HAVING, etc.\n"
            "Example outputs:\n"
            "- Simple: SELECT * FROM customers WHERE active = true;\n"
            "- Complex: SELECT c.name, COUNT(o.id) as order_count\n"
            "           FROM customers c\n"
            "           LEFT JOIN orders o ON c.id = o.customer_id\n"
            "           WHERE c.active = true\n"
            "           GROUP BY c.id, c.name\n"
            "           HAVING COUNT(o.id) > 0\n"
            "           ORDER BY order_count DESC;\n"
            "If the request cannot be handled with a SELECT query, reply: 'Operation not allowed.'\n"
            "If no relevant tables are found in the schema, reply: 'No relevant tables found in schema.'\n\n"
            "SQL QUERY:"
        )
        logger.info(f"[Chat] sql_prompt for message '{request.message}': {sql_prompt}")
        
        # Filter conversation messages to ensure proper order (no consecutive user messages)
        # Only include conversation history, not the current user message
        conversation_history = conversation_messages[:-1]  # Exclude the current user message
        
        # Ensure proper alternating pattern: user -> assistant -> user -> assistant
        # More aggressive filtering to prevent any consecutive same-role messages
        filtered_messages = []
        last_role = None
        
        logger.info(f"[Chat] DEBUG: Starting filtering with {len(conversation_history)} messages")
        for i, msg in enumerate(conversation_history):
            logger.info(f"[Chat] DEBUG: Message {i}: role='{msg.role}', last_role='{last_role}'")
            # Only add if it has a different role than the last message we added
            if msg.role != last_role:
                filtered_messages.append(msg.model_dump(exclude={"metadata"}))
                last_role = msg.role
                logger.info(f"[Chat] DEBUG: Added message {i} with role '{msg.role}'")
            else:
                logger.info(f"[Chat] DEBUG: Skipped message {i} with role '{msg.role}' (same as last_role)")
        
        logger.info(f"[Chat] Original conversation history roles: {[msg.role for msg in conversation_history]}")
        logger.info(f"[Chat] Filtered conversation history roles: {[msg.get('role') for msg in filtered_messages]}")
        
        logger.info(f"[Chat] Using {len(filtered_messages)} filtered messages for SQL generation")
        logger.info(f"[Chat] Filtered message roles: {[msg.get('role') for msg in filtered_messages]}")
        
        # Create the final messages array for LLM
        # Send system prompt with conversation history embedded, plus the current user message
        final_messages = [
            {"role": "system", "content": sql_prompt},
            {"role": "user", "content": request.message}
        ]
        logger.info(f"[Chat] Final messages being sent to LLM:")
        for i, msg in enumerate(final_messages):
            logger.info(f"[Chat] Message {i}: {msg.get('role')} - {msg.get('content', '')[:50]}...")
        
        sql_response = await llm_service.chat(final_messages)
        
        # Clean and extract SQL query - preserve full multi-line SQL
        sql_query = sql_response.strip()
        
        # Remove markdown code blocks (```sql ... ```)
        if sql_query.startswith('```'):
            # Find the end of the code block
            lines = sql_query.split('\n')
            if len(lines) > 1:
                # Skip the first line (```sql) and find the closing ```
                sql_lines = []
                for line in lines[1:]:
                    if line.strip() == '```':
                        break
                    sql_lines.append(line)
                sql_query = '\n'.join(sql_lines).strip()
        
        # Remove common prefixes only from the first line
        lines = sql_query.split('\n')
        if lines:
            first_line = lines[0].strip()
            
            # Remove prefixes only from the first line
            prefixes_to_remove = ['sql query:', 'sql:', 'query:']
            for prefix in prefixes_to_remove:
                if first_line.lower().startswith(prefix):
                    first_line = first_line[len(prefix):].strip()
                    break
            
            # Update the first line
            lines[0] = first_line
            sql_query = '\n'.join(lines).strip()
        
        # Remove trailing semicolon to prevent syntax errors when LIMIT is added
        sql_query = sql_query.rstrip(';').strip()
        
        logger.info(f"[Chat] Raw LLM response:\n{sql_response}")
        logger.info(f"[Chat] Full SQL extracted:\n{sql_query}")
        if sql_query == "Operation not allowed.":
            assistant_message = Message(
                role="assistant",
                content="Operation not allowed."
            )
            conversation_messages.append(assistant_message)
            try:
                memory_data = {
                    "messages": [msg.dict() for msg in conversation_messages],
                    "context": session.conversation_memory.context.dict() if session.conversation_memory.context else {},
                    "user_preferences": session.conversation_memory.user_preferences.dict() if session.conversation_memory.user_preferences else {},
                    "session_variables": session.conversation_memory.session_variables or {}
                }
                await mongo_service.update_conversation_memory_upsert(cleaned_session_id, memory_data, user_id, get_storage_business_id())
                logger.info(f"[Chat] Successfully stored conversation for session: {cleaned_session_id}")
            except Exception as e:
                logger.error(f"[Chat] Failed to store conversation for session {cleaned_session_id}: {e}")
            return ChatResponse(response="Operation not allowed.", session_id=cleaned_session_id)
        
        if sql_query == "No relevant tables found in schema.":
            assistant_message = Message(
                role="assistant",
                content="I couldn't find any relevant database tables for your query. Please try rephrasing your request or contact support if you believe this is an error."
            )
            conversation_messages.append(assistant_message)
            try:
                memory_data = {
                    "messages": [msg.dict() for msg in conversation_messages],
                    "context": session.conversation_memory.context.dict() if session.conversation_memory.context else {},
                    "user_preferences": session.conversation_memory.user_preferences.dict() if session.conversation_memory.user_preferences else {},
                    "session_variables": session.conversation_memory.session_variables or {}
                }
                await mongo_service.update_conversation_memory_upsert(cleaned_session_id, memory_data, user_id, get_storage_business_id())
                logger.info(f"[Chat] Successfully stored conversation for session: {cleaned_session_id}")
            except Exception as e:
                logger.error(f"[Chat] Failed to store conversation for session {cleaned_session_id}: {e}")
            return ChatResponse(response="I couldn't find any relevant database tables for your query. Please try rephrasing your request or contact support if you believe this is an error.", session_id=cleaned_session_id)
        # Execute SQL query via MCP
        logger.info(f"[Chat] Executing SQL via MCP: '{sql_query}' for business '{business_id}'")
        mcp_result = await mcp_client.execute_query(sql_query, business_id)
        
        if isinstance(mcp_result, dict) and "error" in mcp_result:
            error_msg = mcp_result['error']
            logger.error(f"[Chat] MCP execution error: {error_msg}")
            
            # Provide user-friendly error message
            if "name 'time' is not defined" in error_msg:
                user_error = "Database connection error. Please try again."
            elif "connection" in error_msg.lower():
                user_error = "Database connection issue. Please check your database configuration."
            else:
                user_error = f"Database error: {error_msg}"
            
            assistant_message = Message(
                role="assistant",
                content=f"I encountered an error while executing your query: {user_error}"
            )
            conversation_messages.append(assistant_message)
            try:
                memory_data = {
                    "messages": [msg.dict() for msg in conversation_messages],
                    "context": session.conversation_memory.context.dict() if session.conversation_memory.context else {},
                    "user_preferences": session.conversation_memory.user_preferences.dict() if session.conversation_memory.user_preferences else {},
                    "session_variables": session.conversation_memory.session_variables or {}
                }
                await mongo_service.update_conversation_memory_upsert(cleaned_session_id, memory_data, user_id, get_storage_business_id())
                logger.info(f"[Chat] Successfully stored conversation for session: {cleaned_session_id}")
            except Exception as e:
                logger.error(f"[Chat] Failed to store conversation for session {cleaned_session_id}: {e}")
            return ChatResponse(response=f"I encountered an error while executing your query: {user_error}", session_id=cleaned_session_id)
        # Handle MCP result parsing
        if isinstance(mcp_result, str):
            try:
                mcp_result = json.loads(mcp_result)
            except Exception:
                pass
        
        # Handle nested MCP response structure
        if isinstance(mcp_result, dict) and "content" in mcp_result:
            # Extract the actual result from nested structure
            if isinstance(mcp_result["content"], list) and len(mcp_result["content"]) > 0:
                content_item = mcp_result["content"][0]
                if isinstance(content_item, dict) and "text" in content_item:
                    try:
                        mcp_result = json.loads(content_item["text"])
                    except Exception:
                        pass
        
        formatted_result = ""
        if isinstance(mcp_result, dict) and "results" in mcp_result:
            row_count = mcp_result.get('row_count', 0)
            formatted_result += f"Found {row_count} result{'s' if row_count != 1 else ''}:\n\n"
            
            if mcp_result["results"]:
                # Format as a nice table
                if len(mcp_result["results"]) > 0:
                    # Get column names from first row
                    columns = list(mcp_result["results"][0].keys())
                    
                    # Create header
                    header = " | ".join(columns)
                    separator = "-" * len(header)
                    formatted_result += f"{header}\n{separator}\n"
                    
                    # Add rows (limit to 10 for display)
                    for i, row in enumerate(mcp_result["results"][:10]):
                        row_values = [str(row.get(col, '')) for col in columns]
                        formatted_result += " | ".join(row_values) + "\n"
                    
                    if len(mcp_result["results"]) > 10:
                        formatted_result += f"\n... and {len(mcp_result['results']) - 10} more results\n"
            else:
                formatted_result += "No data found matching your query."
        else:
            formatted_result = f"Query executed successfully. Result: {str(mcp_result)}"
        # Create a more natural, user-friendly response
        if formatted_result.strip():
            response_content = f"Here's what I found:\n\n{formatted_result}"
        else:
            response_content = "I couldn't find any matching data for your query."
        
        # Only add debug info if explicitly requested or in development mode
        current_settings = get_settings()
        if current_settings.server.debug and current_settings.server.show_sql_debug:
            response_content += f"\n\n[Debug: SQL Query: {sql_query}]"
        
        assistant_message = Message(
            role="assistant",
            content=response_content
        )
        conversation_messages.append(assistant_message)
        logger.info(f"[Chat] About to store {len(conversation_messages)} messages in session")
        
        # Update conversation memory directly
        try:
            memory_data = {
                "messages": [msg.dict() for msg in conversation_messages],
                "context": session.conversation_memory.context.dict() if session.conversation_memory.context else {},
                "user_preferences": session.conversation_memory.user_preferences.dict() if session.conversation_memory.user_preferences else {},
                "session_variables": session.conversation_memory.session_variables or {}
            }
            # Use the cleaned session_id for storage, not the original escaped one
            result = await mongo_service.update_conversation_memory_upsert(cleaned_session_id, memory_data, user_id, get_storage_business_id())
            logger.info(f"[Chat] MongoDB memory update result: {result}")
            logger.info(f"[Chat] Successfully stored conversation for session: {cleaned_session_id}")
        except Exception as e:
            logger.error(f"[Chat] Failed to store conversation for session {cleaned_session_id}: {e}")
            logger.error(f"[Chat] Exception details: {type(e).__name__}: {str(e)}")
        return ChatResponse(response=response_content, session_id=cleaned_session_id)

@app.post("/debug/generate-sql")
async def debug_generate_sql(
    request: ChatRequest,
    current_user: dict = Depends(get_current_user),
    mongo_service: MongoDBService = Depends(get_mongodb_service)
):
    """Debug endpoint to test SQL generation without execution"""
    user_id = current_user["user_id"]
    business_id = current_user["business_id"]
    
    # Get schema context
    schema_context = await vector_search_service.search_schemas(business_id, request.message, top_k=5)
    
    # Format schema text
    schema_text = ""
    if schema_context:
        for schema in schema_context:
            schema_text += f"\nTable: {schema.get('table_name', 'Unknown')}\n"
            schema_text += f"Description: {schema.get('schema_description', 'No description')}\n"
            schema_text += "Columns:\n"
            for col in schema.get('columns', []):
                schema_text += f"  - {col.get('name', 'Unknown')}: {col.get('type', 'Unknown')} ({col.get('description', 'No description')})\n"
    
    # Generate SQL prompt
    sql_prompt = (
        "You are an expert SQL assistant for a PostgreSQL database. "
        "Your task is to convert natural language requests into SQL SELECT queries.\n\n"
        "AVAILABLE DATABASE SCHEMAS:\n{schema_text}\n"
        "CURRENT USER REQUEST: {request.message}\n\n"
        "INSTRUCTIONS:\n"
        "1. Analyze the user's natural language request\n"
        "2. Identify relevant tables and columns from the schema above\n"
        "3. Generate a single, safe, syntactically correct SQL SELECT query\n"
        "4. Only use SELECT statements - never DROP, DELETE, TRUNCATE, ALTER, CREATE, INSERT, or UPDATE\n"
        "5. Use only the tables and columns provided in the schema context\n\n"
        "OUTPUT FORMAT: Generate ONLY the complete SQL query, no explanations, no markdown, no code blocks, no prefixes.\n"
        "Preserve all SQL clauses including WHERE, ORDER BY, GROUP BY, HAVING, etc.\n"
        "If the request cannot be handled with a SELECT query, reply: 'Operation not allowed.'\n"
        "If no relevant tables are found in the schema, reply: 'No relevant tables found in schema.'\n\n"
        "SQL QUERY:"
    )
    
    # Generate SQL
    sql_response = await llm_service.chat([{"role": "system", "content": sql_prompt}])
    
    # Clean SQL (same logic as main endpoint)
    sql_query = sql_response.strip()
    
    # Remove markdown code blocks
    if sql_query.startswith('```'):
        lines = sql_query.split('\n')
        if len(lines) > 1:
            sql_lines = []
            for line in lines[1:]:
                if line.strip() == '```':
                    break
                sql_lines.append(line)
            sql_query = '\n'.join(sql_lines).strip()
    
    # Remove prefixes from first line
    lines = sql_query.split('\n')
    if lines:
        first_line = lines[0].strip()
        prefixes_to_remove = ['sql query:', 'sql:', 'query:']
        for prefix in prefixes_to_remove:
            if first_line.lower().startswith(prefix):
                first_line = first_line[len(prefix):].strip()
                break
        lines[0] = first_line
        sql_query = '\n'.join(lines).strip()
    
    # Remove trailing semicolon to prevent syntax errors when LIMIT is added
    sql_query = sql_query.rstrip(';').strip()
    
    return {
        "business_id": business_id,
        "user_message": request.message,
        "raw_llm_response": sql_response,
        "cleaned_sql": sql_query,
        "schema_context": [s.get('table_name') for s in schema_context]
    }

if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    
    uvicorn.run(
        "main:app",
        host=settings.server.host,
        port=settings.server.port,
        reload=settings.server.debug,
        log_level=settings.server.log_level.lower()
    ) 