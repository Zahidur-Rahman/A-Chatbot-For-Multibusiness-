"""
Main FastAPI application for the Multi-Business Conversational Chatbot.
"""

from fastapi import FastAPI, HTTPException, Depends, Request
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
from pydantic import BaseModel
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
        raise

    logger.info("Application startup complete")
    yield  # Don't forget this!

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

# Instantiate the vector search service globally
vector_search_service = FaissVectorSearchService()

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
# mongo_service = mongodb_service # This line is no longer needed

# Add this function to extract user_id from the request state (assuming authentication middleware sets it)
async def get_user_id(request: Request):
    # If you use request.state.user, adjust as needed
    user = getattr(request.state, 'user', None)
    if user and hasattr(user, 'user_id'):
        return str(user.user_id)
    # Fallback: use IP if user_id is not available
    return request.client.host

# Helper: Keyword-based DB query classifier
def is_database_related_query(message: str) -> bool:
    database_keywords = [
        'select', 'query', 'database', 'table', 'sql', 'data', 'show', 'find', 'get', 'fetch',
        'customer', 'order', 'product', 'restaurant', 'menu', 'count', 'list', 'search',
        'where', 'from', 'join', 'group', 'having', 'order by', 'limit','give','what'
    ]
    message_lower = message.lower()
    return any(re.search(rf'\\b{re.escape(keyword)}\\b', message_lower) for keyword in database_keywords)

@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(RateLimiter(times=10, seconds=60, identifier=get_user_id))])
async def chat_endpoint(
    request: ChatRequest,
    current_user: dict = Depends(get_current_user),
    mongo_service: MongoDBService = Depends(get_mongodb_service)
):
    user_id = current_user["user_id"]
    business_id = current_user["business_id"]
    role = current_user["role"]
    await require_business_access(user_id, business_id)
    # Try to get session
    session: Optional[ConversationSession] = await mongo_service.get_conversation_session(request.session_id)
    schema_context = None
    if session and not request.refresh_schema_context:
        schema_context = session.cached_schema_context
    if not schema_context:
        schema_context = await vector_search_service.search_schemas(business_id, request.message, top_k=3)
        logger.info(f"[Chat] schema_context for message '{request.message}': {schema_context}")
        if session:
            await mongo_service.set_cached_schema_context(request.session_id, schema_context)
        else:
            from backend.app.models.conversation import ConversationMemory
            session = ConversationSession(
                session_id=request.session_id,
                user_id=user_id,
                business_id=business_id,
                conversation_memory=ConversationMemory(),
                cached_schema_context=schema_context,
                messages=[]
            )
            await mongo_service.create_conversation_session(session)
    # Use conversation memory for context
    conversation_messages = session.conversation_memory.messages if session else []
    # Add new user message
    from backend.app.models.conversation import Message
    user_message = Message(
        role="user",
        content=request.message
    )
    conversation_messages.append(user_message)
    # Use schema_context in prompt (simple example)
    schema_text = "\n".join([str(s) for s in schema_context])
    logger.info(f"[Chat] schema_text for message '{request.message}': {schema_text}")
    system_prompt = f"Relevant schema context for your business:\n{schema_text}"
    messages = [{"role": "system", "content": system_prompt}] + [m.model_dump(exclude={"metadata"}) for m in conversation_messages]
    # Classify message
    if not is_database_related_query(request.message):
        response = await llm_service.chat(messages)
        # Store user and LLM messages in conversation memory
        assistant_message = Message(
            role="assistant",
            content=response
        )
        conversation_messages.append(assistant_message)
        # Persist updated session
        session.conversation_memory.messages = conversation_messages
        await mongo_service.update_conversation_session(session)
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
        logger.info(f"[Chat] sql_prompt for message '{request.message}': {sql_prompt}")
        sql_response = await llm_service.chat([
            {"role": "system", "content": sql_prompt}
        ] + [m.model_dump(exclude={"metadata"}) for m in conversation_messages])
        sql_query = sql_response.strip()
        if sql_query.startswith('```'):
            sql_query = sql_query.strip('`').strip()
        logger.info(f"[Chat] SQL generated for message '{request.message}': {sql_query}")
        if sql_query == "Operation not allowed.":
            assistant_message = Message(
                role="assistant",
                content="Operation not allowed."
            )
            conversation_messages.append(assistant_message)
            session.conversation_memory.messages = conversation_messages
            await mongo_service.update_conversation_session(session)
            return ChatResponse(response="Operation not allowed.")
        mcp_result = await mcp_client.execute_query(sql_query, business_id)
        if isinstance(mcp_result, dict) and "error" in mcp_result:
            assistant_message = Message(
                role="assistant",
                content=f"SQL Query: {sql_query}\n\nError: {mcp_result['error']}"
            )
            conversation_messages.append(assistant_message)
            session.conversation_memory.messages = conversation_messages
            await mongo_service.update_conversation_session(session)
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
        assistant_message = Message(
            role="assistant",
            content=f"SQL Query: {sql_query}\n\n{formatted_result}"
        )
        conversation_messages.append(assistant_message)
        session.conversation_memory.messages = conversation_messages
        await mongo_service.update_conversation_session(session)
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