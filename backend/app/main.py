"""
Main FastAPI application for the Multi-Business Conversational Chatbot.
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from contextlib import asynccontextmanager
import logging
import sys
from typing import Dict, Any, List, Optional
import re
import os
import json
from datetime import datetime

from backend.app.config import get_settings, Settings
from backend.app.auth.routes import router as auth_router
from backend.app.services.business_service import router as business_admin_router
from backend.app.services.mistral_llm_service import MistralLLMService
from backend.app.services.vector_search import FaissVectorSearchService
from backend.app.services.mongodb_service import MongoDBService
from backend.app.models.conversation import ConversationSession
from backend.app.mcp.mcp_client import MCPClient
from backend.app.auth.jwt_handler import get_current_user, require_business_access, require_admin
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
import aioredis
from backend.app.models.business import BusinessConfig, BusinessConfigCreate, BusinessConfigUpdate
from backend.app.models.user import User, UserCreate, UserUpdate
from backend.app.services.mongodb_service import mongodb_service
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting Multi-Business Conversational Chatbot...")
    
    # Initialize services here
    # await initialize_database()
    # await initialize_redis()
    # await initialize_mcp_server()
    
    # Initialize MCP client (singleton)
    # MCP_SERVER_PATH = os.path.join(os.path.dirname(__file__), "mcp", "server_enhanced.py")
    # mcp_client = MCPClient(MCP_SERVER_PATH)
    
    # Initialize FastAPI-Limiter with Redis
    settings = get_settings()
    redis_url = f"redis://{settings.redis.host}:{settings.redis.port}/0"
    redis = await aioredis.from_url(redis_url, encoding="utf8", decode_responses=True)
    await FastAPILimiter.init(redis)
    
    logger.info("Application startup complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Multi-Business Conversational Chatbot...")
    
    # Cleanup services here
    # await cleanup_database()
    # await cleanup_redis()
    # await cleanup_mcp_server()
    
    await FastAPILimiter.close()
    
    logger.info("Application shutdown complete")

def create_app() -> FastAPI:
    """Create and configure the FastAPI application"""
    settings = get_settings()
    
    app = FastAPI(
        title="Multi-Business Conversational Chatbot",
        description="A production-ready, dynamic multi-business conversational chatbot with PostgreSQL integration, vector-based schema discovery, and LangChain conversational AI.",
        version="1.0.0",
        docs_url="/docs" if settings.server.debug else None,
        redoc_url="/redoc" if settings.server.debug else None,
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
async def list_businesses(settings: Settings = Depends(get_settings)):
    """List all configured businesses"""
    return {
        "businesses": settings.business_ids,
        "count": len(settings.business_ids)
    }

@app.get("/businesses/{business_id}/config")
async def get_business_config(business_id: str, settings: Settings = Depends(get_settings)):
    """Get configuration for a specific business"""
    if business_id not in settings.business_configs:
        raise HTTPException(status_code=404, detail=f"Business '{business_id}' not found")
    
    config = settings.business_configs[business_id]
    return {
        "business_id": business_id,
        "host": config.host,
        "database": config.database,
        "port": config.port,
        "user": config.user,
        # Don't expose password in response
    }

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

@app.post("/admin/businesses", dependencies=[Depends(require_admin)])
async def admin_add_business(business_id: str, config: Dict[str, Any], current_user: dict = Depends(get_current_user)):
    """Admin: Add a new business"""
    mongo_service = MongoDBService()
    success = await mongo_service.add_business(business_id, config)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to add business")
    return {"message": f"Business '{business_id}' added", "status": "success"}

@app.delete("/admin/businesses/{business_id}", dependencies=[Depends(require_admin)])
async def admin_remove_business(business_id: str, current_user: dict = Depends(get_current_user)):
    """Admin: Remove a business"""
    mongo_service = MongoDBService()
    success = await mongo_service.remove_business(business_id)
    if not success:
        raise HTTPException(status_code=404, detail="Business not found or could not be removed")
    return {"message": f"Business '{business_id}' removed", "status": "success"}

@app.post("/admin/users", dependencies=[Depends(require_admin)])
async def admin_add_user(user: Dict[str, Any], current_user: dict = Depends(get_current_user)):
    """Admin: Add a new user and assign to business(es)"""
    mongo_service = MongoDBService()
    success = await mongo_service.add_user(user)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to add user")
    return {"message": f"User '{user.get('username')}' added", "status": "success"}

@app.delete("/admin/users/{user_id}", dependencies=[Depends(require_admin)])
async def admin_remove_user(user_id: str, current_user: dict = Depends(get_current_user)):
    """Admin: Remove a user"""
    mongo_service = MongoDBService()
    success = await mongo_service.remove_user(user_id)
    if not success:
        raise HTTPException(status_code=404, detail="User not found or could not be removed")
    return {"message": f"User '{user_id}' removed", "status": "success"}

@app.post("/admin/users/{user_id}/assign", dependencies=[Depends(require_admin)])
async def admin_assign_user_to_business(user_id: str, business_ids: List[str], current_user: dict = Depends(get_current_user)):
    """Admin: Assign user to one or more businesses"""
    mongo_service = MongoDBService()
    success = await mongo_service.assign_user_to_businesses(user_id, business_ids)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to assign user to businesses")
    return {"message": f"User '{user_id}' assigned to businesses {business_ids}", "status": "success"}

@app.post("/admin/businesses/{business_id}/schemas", dependencies=[Depends(require_admin)])
async def admin_add_or_update_business_schema(business_id: str, schema: Dict[str, Any], current_user: dict = Depends(get_current_user)):
    """Admin: Add or update a business schema"""
    mongo_service = MongoDBService()
    success = await mongo_service.add_or_update_business_schema(business_id, schema)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to add/update schema")
    # AUTOMATION: Re-index vector embeddings after schema change
    await FaissVectorSearchService().index_business_schemas(business_id)
    return {"message": f"Schema for business '{business_id}' added/updated and vector index refreshed", "status": "success"}

# =============================================================================
# ERROR HANDLERS
# =============================================================================

@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors"""
    return {
        "error": "Not found",
        "message": "The requested resource was not found",
        "status_code": 404
    }

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {exc}")
    return {
        "error": "Internal server error",
        "message": "An unexpected error occurred",
        "status_code": 500
    }

# =============================================================================
# DEVELOPMENT ENDPOINTS (only in debug mode)
# =============================================================================

@app.get("/debug/config")
async def debug_config(settings: Settings = Depends(get_settings)):
    """Debug endpoint to view configuration (only in debug mode)"""
    if not settings.server.debug:
        raise HTTPException(status_code=404, detail="Not found")
    
    return {
        "server": {
            "host": settings.server.host,
            "port": settings.server.port,
            "debug": settings.server.debug,
            "log_level": settings.server.log_level
        },
        "database": {
            "host": settings.database.host,
            "database": settings.database.database,
            "port": settings.database.port,
            "user": settings.database.user
        },
        "businesses": list(settings.business_configs.keys()),
        "features": {
            "multi_business": settings.features.enable_multi_business,
            "vector_search": settings.features.enable_vector_search,
            "schema_discovery": settings.features.enable_schema_discovery,
            "conversation_memory": settings.features.enable_conversation_memory,
            "analytics": settings.features.enable_analytics
        }
    }

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    session_id: str
    conversation_history: Optional[List[ChatMessage]] = []
    refresh_schema_context: Optional[bool] = False

class ChatResponse(BaseModel):
    response: str

llm_service = MistralLLMService()
vector_search_service = FaissVectorSearchService()
mongo_service = MongoDBService()

# Helper: Keyword-based DB query classifier
def is_database_related_query(message: str) -> bool:
    database_keywords = [
        'select', 'query', 'database', 'table', 'sql', 'data', 'show', 'find', 'get', 'fetch',
        'customer', 'order', 'product', 'restaurant', 'menu', 'count', 'list', 'search',
        'where', 'from', 'join', 'group', 'having', 'order by', 'limit','give','what'
    ]
    message_lower = message.lower()
    return any(re.search(rf'\\b{re.escape(keyword)}\\b', message_lower) for keyword in database_keywords)

@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(RateLimiter(times=10, seconds=60, identifier="user_id"))])
async def chat_endpoint(
    request: ChatRequest,
    current_user: dict = Depends(get_current_user)
):
    """AI-powered chat endpoint using Mistral LLM, session-based schema context caching, and MCP SQL execution. JWT-protected."""
    user_id = current_user["user_id"]
    business_id = current_user["business_id"]
    role = current_user["role"]
    await require_business_access(user_business_id=business_id, requested_business_id=business_id)
    # Try to get session
    session: Optional[ConversationSession] = await mongo_service.get_conversation_session(request.session_id)
    schema_context = None
    if session and not request.refresh_schema_context:
        schema_context = session.cached_schema_context
    if not schema_context:
        schema_context = await vector_search_service.search_schemas(business_id, request.message, top_k=3)
        if session:
            await mongo_service.set_cached_schema_context(request.session_id, schema_context)
        else:
            new_session = ConversationSession(
                session_id=request.session_id,
                user_id=user_id,
                business_id=business_id,
                conversation_memory=None,
                cached_schema_context=schema_context,
                messages=[]
            )
            await mongo_service.create_conversation_session(new_session)
    # Persistent conversation memory: fetch last 10 messages
    conversation_messages = await mongo_service.get_last_n_messages(request.session_id, n=10)
    # Add new user message
    user_message = {
        "role": "user",
        "content": request.message,
        "timestamp": datetime.utcnow().isoformat()
    }
    conversation_messages.append(user_message)
    # Use schema_context in prompt (simple example)
    schema_text = "\n".join([str(s) for s in schema_context])
    system_prompt = f"Relevant schema context for your business:\n{schema_text}"
    messages = [{"role": "system", "content": system_prompt}] + conversation_messages
    # Classify message
    if not is_database_related_query(request.message):
        response = await llm_service.chat(messages)
        # Store user and LLM messages in MongoDB
        await mongo_service.add_message_to_conversation(request.session_id, user_message)
        assistant_message = {
            "role": "assistant",
            "content": response,
            "timestamp": datetime.utcnow().isoformat()
        }
        await mongo_service.add_message_to_conversation(request.session_id, assistant_message)
        return ChatResponse(response=response)
    else:
        sql_prompt = (
            "You are an expert SQL assistant for a PostgreSQL database. "
            "Convert the user request into a single, safe, syntactically correct SQL SELECT query.\n"
            "Only generate SELECT statements. Never use DROP, DELETE, TRUNCATE, ALTER, CREATE, INSERT, or UPDATE.\n"
            f"Use the following table schemas:\n{schema_text}\n"
            "Do NOT use Markdown formatting or code blocks. Only output the SQL statement, nothing else.\n"
            "If the user asks for something not possible with a SELECT, reply: 'Operation not allowed.'\n"
            "Use only the columns and tables provided.\n"
            f"User request: {request.message}"
        )
        sql_response = await llm_service.chat([
            {"role": "system", "content": sql_prompt}
        ] + conversation_messages)
        sql_query = sql_response.strip()
        if sql_query.startswith('```'):
            sql_query = sql_query.strip('`').strip()
        if sql_query == "Operation not allowed.":
            await mongo_service.add_message_to_conversation(request.session_id, user_message)
            assistant_message = {
                "role": "assistant",
                "content": "Operation not allowed.",
                "timestamp": datetime.utcnow().isoformat()
            }
            await mongo_service.add_message_to_conversation(request.session_id, assistant_message)
            return ChatResponse(response="Operation not allowed.")
        mcp_result = await mcp_client.execute_query(sql_query, business_id)
        if isinstance(mcp_result, dict) and "error" in mcp_result:
            await mongo_service.add_message_to_conversation(request.session_id, user_message)
            assistant_message = {
                "role": "assistant",
                "content": f"SQL Query: {sql_query}\n\nError: {mcp_result['error']}",
                "timestamp": datetime.utcnow().isoformat()
            }
            await mongo_service.add_message_to_conversation(request.session_id, assistant_message)
            return ChatResponse(response=f"SQL Query: {sql_query}\n\nError: {mcp_result['error']}")
        if isinstance(mcp_result, str):
            try:
                mcp_result = json.loads(mcp_result)
            except Exception:
                pass
        formatted_result = ""
        if isinstance(mcp_result, dict) and "results" in mcp_result:
            formatted_result += f"Query Results:\nRows returned: {mcp_result.get('row_count', 0)}\n"
            if mcp_result["results"]:
                formatted_result += "Data:\n"
                for i, row in enumerate(mcp_result["results"][:10]):
                    formatted_result += f"  {i+1}. {row}\n"
                if len(mcp_result["results"]) > 10:
                    formatted_result += f"  ... and {len(mcp_result['results']) - 10} more rows\n"
        else:
            formatted_result = str(mcp_result)
        # Store user and LLM messages in MongoDB
        await mongo_service.add_message_to_conversation(request.session_id, user_message)
        assistant_message = {
            "role": "assistant",
            "content": f"SQL Query: {sql_query}\n\n{formatted_result}",
            "timestamp": datetime.utcnow().isoformat()
        }
        await mongo_service.add_message_to_conversation(request.session_id, assistant_message)
        return ChatResponse(response=f"SQL Query: {sql_query}\n\n{formatted_result}")

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