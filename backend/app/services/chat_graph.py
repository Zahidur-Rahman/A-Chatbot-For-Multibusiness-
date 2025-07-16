from langgraph.graph import StateGraph
from backend.app.services.mistral_llm_service import MistralLLMService
from backend.app.services.vector_search import FaissVectorSearchService
from backend.app.mcp.mcp_client import MCPClient
from backend.app.utils.query_classifier import is_database_related_query_dynamic
from backend.app.utils.chat_helpers import build_sql_prompt, clean_sql_from_llm, format_db_result, build_system_prompt
from backend.app.models.conversation import ChatMessage
from backend.app.models.chat_graph_state import ChatGraphState
import logging
import time
import string
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
    # Decide next node based on classification
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
    # Assertion: no two consecutive messages have the same role
    for i in range(1, len(messages)):
        if messages[i]['role'] == messages[i-1]['role']:
            raise ValueError(f"Two consecutive roles: {messages[i-1]['role']} and {messages[i]['role']} at positions {i-1}, {i}")
    # Truncate to last valid system/user/assistant turn if error persists
    if len(messages) > 3:
        messages = [messages[0]] + messages[-2:]
    response = await llm_service.chat(messages)
    context.response = response
    return context

# --- New Node: Generate SQL Only (no execution) ---
async def generate_sql_node(context: ChatGraphState) -> ChatGraphState:
    context.conversation_history = ensure_chat_messages(context.conversation_history or [])
    # Remove any trailing 'user' messages before building LLM input
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
    messages.append({"role": "user", "content": context.message})
    if len(messages) > 2 and messages[-1]["role"] == "user" and messages[-2]["role"] == "user":
        messages.pop(-2)
    # Assertion: no two consecutive messages have the same role
    for i in range(1, len(messages)):
        if messages[i]['role'] == messages[i-1]['role']:
            raise ValueError(f"Two consecutive roles: {messages[i-1]['role']} and {messages[i]['role']} at positions {i-1}, {i}")
    if len(messages) > 3:
        messages = [messages[0]] + messages[-2:]
    sql_response = await llm_service.chat(messages)
    sql_query = clean_sql_from_llm(sql_response)
    context.sql = sql_query
    return context

# --- New Node: Execute SQL via MCP ---
async def execute_sql_node(context: ChatGraphState) -> ChatGraphState:
    import logging
    logger = logging.getLogger(__name__)
    sql = getattr(context, 'sql', None)
    business_id = getattr(context, 'business_id', None)
    user_id = getattr(context, 'user_id', None)
    logger.info(f"[SQL_EXECUTION] Sending to MCP: user_id={user_id}, business_id={business_id}, sql={sql}")
    start_time = time.time()
    try:
        mcp_result = await mcp_client.execute_query(sql, business_id)
        elapsed = time.time() - start_time
        if isinstance(mcp_result, dict) and mcp_result.get('error'):
            logger.error(f"[SQL_EXECUTION] MCP ERROR for business_id={business_id}: {mcp_result.get('error')}")
        else:
            logger.info(f"[SQL_EXECUTION] MCP response for business_id={business_id} (elapsed={elapsed:.2f}s): {mcp_result}")
        context.db_result = mcp_result
        context.response = format_db_result(mcp_result)
        # --- FIX: Clear pause/confirmation fields after successful execution ---
        context.pause_reason = None
        context.pause_message = None
        context.confirm = None
        context.resume_from_pause = None
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"[SQL_EXECUTION] Exception calling MCP for business_id={business_id}, sql={sql} (elapsed={elapsed:.2f}s): {e}", exc_info=True)
        context.db_result = None
        context.response = f"Error executing query: {e}"
    return context

# --- Dependency Resolver logic after SQL generation (for delete confirmation) ---
async def db_tool_with_dependency_check(context: ChatGraphState) -> ChatGraphState:
    sql = context.sql or ""
    import logging
    logger = logging.getLogger(__name__)
    # Check for DELETE and require confirmation BEFORE executing
    if "delete" in sql.lower() and not getattr(context, "confirm", False):
        logger.error("[DBToolWithDepCheck] Pausing for delete confirmation")
        context.response = "You are about to delete data. Please confirm by replying 'confirm delete'."
        context.pause_reason = "confirm_delete"
        context.pause_message = context.response
        context.next = "PauseNode"
        return context
    # Check for UPDATE and require confirmation BEFORE executing
    if "update" in sql.lower() and not getattr(context, "confirm", False):
        logger.error("[DBToolWithDepCheck] Pausing for update confirmation")
        context.response = "You are about to update data. Please confirm by replying 'confirm update'."
        context.pause_reason = "confirm_update"
        context.pause_message = context.response
        context.next = "PauseNode"
        return context
    # Only execute the SQL if not a DELETE/UPDATE or if already confirmed
    logger.error("[DBToolWithDepCheck] No confirmation needed, proceeding to ExecuteSQL")
    context.next = "ExecuteSQL"
    return context

async def response_node(context: ChatGraphState) -> ChatGraphState:
    # --- FIX: Safety net to clear any lingering pause/confirmation fields ---
    context.pause_reason = None
    context.pause_message = None
    context.confirm = None
    context.resume_from_pause = None
    return context

async def pause_node(context: ChatGraphState) -> ChatGraphState:
    # Set the correct pause message based on the reason
    if getattr(context, "pause_reason", None) == "confirm_update":
        context.response = "You are about to update data. Please confirm by replying 'confirm update'."
        context.pause_message = context.response
    elif getattr(context, "pause_reason", None) == "confirm_delete":
        context.response = "You are about to delete data. Please confirm by replying 'confirm delete'."
        context.pause_message = context.response
    else:
        # Fallback in case pause_reason is missing
        context.response = "Confirmation required for this action. Please confirm to proceed."
        context.pause_message = context.response
    return context

# --- New Node: Resume or Classify ---
async def resume_or_classify_node(context: ChatGraphState) -> ChatGraphState:
    import logging
    logger = logging.getLogger(__name__)
    confirm_triggers = {"confirm", "yes", "confirm update", "confirm delete", "yes, update", "yes, delete", "update confirmed", "delete confirmed"}
    user_message = getattr(context, 'message', '').strip().lower()
    # Remove punctuation for more robust matching
    user_message_clean = user_message.translate(str.maketrans('', '', string.punctuation))
    logger.debug(f"[ResumeOrClassify] user_message: '{user_message}', cleaned: '{user_message_clean}', pause_reason: '{getattr(context, 'pause_reason', None)}', resume_from_pause: '{getattr(context, 'resume_from_pause', False)}'")
    # If we are in a pause state and the user confirms, treat as confirmation
    if not getattr(context, 'resume_from_pause', False) and getattr(context, 'pause_reason', None) and any(trigger in user_message_clean for trigger in confirm_triggers):
        logger.error(f"[ResumeOrClassify] Confirmation '{user_message}' received for pause_reason '{context.pause_reason}'. Resuming to ExecuteSQL.")
        context.resume_from_pause = True
        context.confirm = True
        context.next = "ExecuteSQL"
        return context
    if getattr(context, "resume_from_pause", False):
        context.next = "ExecuteSQL"
    else:
        context.next = "Classify"
    return context

# --- Build the LangGraph Workflow ---
builder = StateGraph(ChatGraphState)
builder.add_node("ResumeOrClassify", resume_or_classify_node)
builder.add_node("Classify", classify_message)
builder.add_node("Router", router_node)
builder.add_node("VectorSearch", vector_search_node)
builder.add_node("LLMChat", llm_chat_node)
builder.add_node("GenerateSQL", generate_sql_node)
builder.add_node("DBToolWithDepCheck", db_tool_with_dependency_check)
builder.add_node("ExecuteSQL", execute_sql_node)
builder.add_node("Respond", response_node)
builder.add_node("PauseNode", pause_node)

# Edges
builder.add_conditional_edges(
    "ResumeOrClassify",
    lambda x: x.next,
    {"Classify": "Classify", "ExecuteSQL": "ExecuteSQL"}
)
builder.add_edge("Classify", "Router")
builder.add_conditional_edges(
    "Router",
    lambda x: x.next,  # Use the 'next' attribute from router_node's return value
    {"LLMChat": "LLMChat", "VectorSearch": "VectorSearch"}
)
builder.add_edge("VectorSearch", "GenerateSQL")
builder.add_edge("GenerateSQL", "DBToolWithDepCheck")
builder.add_conditional_edges(
    "DBToolWithDepCheck",
    lambda x: getattr(x, "next", None),
    {"PauseNode": "PauseNode", "ExecuteSQL": "ExecuteSQL"}
)
builder.add_edge("ExecuteSQL", "Respond")
builder.add_edge("PauseNode", "Respond")

builder.set_entry_point("ResumeOrClassify")

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
