from langgraph.graph import StateGraph
from backend.app.services.mistral_llm_service import MistralLLMService
from backend.app.services.vector_search import FaissVectorSearchService
from backend.app.mcp.mcp_client import MCPClient
from backend.app.utils.query_classifier import is_database_related_query_dynamic
from backend.app.utils.chat_helpers import build_sql_prompt, clean_sql_from_llm, format_db_result, build_system_prompt
from backend.app.models.conversation import ChatMessage
from backend.app.models.chat_graph_state import ChatGraphState
import logging
logger = logging.getLogger(__name__)

# Instantiate your services (adjust script_path as needed)
llm_service = MistralLLMService()
vector_search_service = FaissVectorSearchService()
mcp_client = MCPClient(script_path="backend/app/mcp/server_enhanced.py")

def ensure_chat_messages(messages):
    return [msg if isinstance(msg, ChatMessage) else ChatMessage.model_validate(msg) for msg in messages]

# --- Node Functions ---

async def classify_message(context: ChatGraphState) -> ChatGraphState:
    context.conversation_history = ensure_chat_messages(context.conversation_history or [])
    context.is_db_query = await is_database_related_query_dynamic(
        context.message,
        context.business_id,
        vector_search_service,
        [msg.model_dump() for msg in context.conversation_history]
    )
    return context

async def router_node(context: ChatGraphState) -> ChatGraphState:
    if context.is_db_query:
        context.next = "VectorSearch"
    else:
        context.next = "LLMChat"
    return context

async def vector_search_node(context: ChatGraphState) -> ChatGraphState:
    context.schema_context = await vector_search_service.search_schemas(
        context.business_id, context.message, top_k=5
    )
    return context

async def llm_chat_node(context: ChatGraphState) -> ChatGraphState:
    context.conversation_history = ensure_chat_messages(context.conversation_history or [])
    # FINAL CHECK: Remove any trailing 'user' messages before building LLM input
    while context.conversation_history and (
        getattr(context.conversation_history[-1], 'role', None) == 'user' or
        (isinstance(context.conversation_history[-1], dict) and context.conversation_history[-1].get('role') == 'user')
    ):
        context.conversation_history.pop()
    context.system_prompt = build_system_prompt(
        context, context.conversation_history, context.schema_context or []
    )
    messages = [
        {"role": "system", "content": context.system_prompt},
    ] + [msg.model_dump() for msg in context.conversation_history]
    # Always add the new user message as the last message
    messages.append({"role": "user", "content": context.message})
    # ABSOLUTE GUARANTEE: If the last two messages are both 'user', remove the second-to-last one
    if len(messages) > 2 and messages[-1]["role"] == "user" and messages[-2]["role"] == "user":
        messages.pop(-2)
    logger.error(f"[LLM INPUT - BEFORE CALL] messages: {messages}")
    # Assertion: no two consecutive messages have the same role
    for i in range(1, len(messages)):
        if messages[i]['role'] == messages[i-1]['role']:
            logger.error(f"[ASSERTION FAILED] Two consecutive roles: {messages[i-1]['role']} and {messages[i]['role']} at positions {i-1}, {i}: {messages}")
            raise ValueError(f"Two consecutive roles: {messages[i-1]['role']} and {messages[i]['role']} at positions {i-1}, {i}")
    # Truncate to last valid system/user/assistant turn if error persists
    if len(messages) > 3:
        messages = [messages[0]] + messages[-2:]
        logger.error(f"[LLM INPUT - TRUNCATED] messages: {messages}")
    response = await llm_service.chat(messages)
    context.response = response
    return context

async def db_tool_node(context: ChatGraphState) -> ChatGraphState:
    context.conversation_history = ensure_chat_messages(context.conversation_history or [])
    # FINAL CHECK: Remove any trailing 'user' messages before building LLM input
    while context.conversation_history and (
        getattr(context.conversation_history[-1], 'role', None) == 'user' or
        (isinstance(context.conversation_history[-1], dict) and context.conversation_history[-1].get('role') == 'user')
    ):
        context.conversation_history.pop()
    context.sql_prompt = build_sql_prompt(
        context, context.conversation_history, context.schema_context or []
    )
    messages = [
        {"role": "system", "content": context.sql_prompt},
    ] + [msg.model_dump() for msg in context.conversation_history]
    # Always add the new user message as the last message
    messages.append({"role": "user", "content": context.message})
    # ABSOLUTE GUARANTEE: If the last two messages are both 'user', remove the second-to-last one
    if len(messages) > 2 and messages[-1]["role"] == "user" and messages[-2]["role"] == "user":
        messages.pop(-2)
    logger.error(f"[LLM INPUT - BEFORE CALL] messages: {messages}")
    # Assertion: no two consecutive messages have the same role
    for i in range(1, len(messages)):
        if messages[i]['role'] == messages[i-1]['role']:
            logger.error(f"[ASSERTION FAILED] Two consecutive roles: {messages[i-1]['role']} and {messages[i]['role']} at positions {i-1}, {i}: {messages}")
            raise ValueError(f"Two consecutive roles: {messages[i-1]['role']} and {messages[i]['role']} at positions {i-1}, {i}")
    # Truncate to last valid system/user/assistant turn if error persists
    if len(messages) > 3:
        messages = [messages[0]] + messages[-2:]
        logger.error(f"[LLM INPUT - TRUNCATED] messages: {messages}")
    sql_response = await llm_service.chat(messages)
    sql_query = clean_sql_from_llm(sql_response)
    logger.error(f"[DBTool] Generated SQL: {sql_query}")
    context.sql = sql_query
    mcp_result = await mcp_client.execute_query(sql_query, context.business_id)
    context.db_result = mcp_result
    context.response = format_db_result(mcp_result)
    return context

async def response_node(context: ChatGraphState) -> ChatGraphState:
    return context

# --- Build the LangGraph Workflow ---
builder = StateGraph(ChatGraphState)
builder.add_node("Classify", classify_message)
builder.add_node("Router", router_node)
builder.add_node("VectorSearch", vector_search_node)
builder.add_node("LLMChat", llm_chat_node)
builder.add_node("DBTool", db_tool_node)
builder.add_node("Respond", response_node)

# Edges
builder.add_edge("Classify", "Router")
builder.add_conditional_edges(
    "Router",
    lambda x: x.next,  # Use the 'next' attribute from router_node's return value
    {"LLMChat": "LLMChat", "VectorSearch": "VectorSearch"}
)
builder.add_edge("VectorSearch", "DBTool")
builder.add_edge("DBTool", "Respond")
builder.add_edge("LLMChat", "Respond")

builder.set_entry_point("Classify")

chat_graph = builder.compile()

# --- Usage Example (in FastAPI endpoint) ---
# from .chat_graph import chat_graph
# result = await chat_graph.arun(context)
# return {"response": result['response'], ...}

# --- TODOs ---
# - Implement or import is_database_related_query_dynamic
# - Implement prompt-building and SQL-cleaning logic
# - Format DB results for user
# - Integrate conversation memory/history as needed
