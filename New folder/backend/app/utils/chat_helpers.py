def build_system_prompt(context, conversation_history, schema_context):
    # Compose a system prompt similar to your main.py logic
    schema_text = ""
    if schema_context:
        for schema in schema_context:
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

    system_prompt = (
        f"You are a friendly and helpful AI assistant. You can help with general questions and also access business data when needed.\n\n"
        f"AVAILABLE DATA:\n{schema_text}\n\n"
        f"CONVERSATION HISTORY:\n"
    )
    for msg in conversation_history:
        role = msg.get('role', 'unknown') if isinstance(msg, dict) else getattr(msg, 'role', 'unknown')
        content = msg.get('content', '') if isinstance(msg, dict) else getattr(msg, 'content', '')
        system_prompt += f"{role.upper()}: {content}\n"
    # Fix: Use attribute access for context.message
    system_prompt += (
        f"\nCURRENT USER REQUEST: {context.message}\n\n"
        "INSTRUCTIONS:\n"
        "1. Be friendly and conversational in your responses\n"
        "2. Provide helpful, accurate information based on the available data\n"
        "3. Consider the conversation history for context\n"
        "4. Keep responses concise and user-friendly\n"
        "5. If asked about specific data, provide clear and relevant information\n"
    )
    return system_prompt

def build_sql_prompt(context, conversation_history, schema_context):
    # Compose a SQL prompt similar to your main.py logic
    schema_text = ""
    if schema_context:
        for schema in schema_context:
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

    sql_prompt = (
        "You are an expert SQL assistant for a PostgreSQL database. "
        "Your task is to convert natural language requests into SQL SELECT queries.\n\n"
        f"AVAILABLE DATABASE SCHEMAS:\n{schema_text}\n"
        f"CONVERSATION HISTORY:\n"
    )
    for msg in conversation_history:
        role = msg.get('role', 'unknown') if isinstance(msg, dict) else getattr(msg, 'role', 'unknown')
        content = msg.get('content', '') if isinstance(msg, dict) else getattr(msg, 'content', '')
        sql_prompt += f"{role.upper()}: {content}\n"
    # Fix: Use attribute access for context.message
    sql_prompt += (
        f"\nCURRENT USER REQUEST: {context.message}\n"
        "INSTRUCTIONS:\n"
        "1. Analyze the user's natural language request\n"
        "2. Identify relevant tables and columns from the schema above\n"
        "3. Generate a single, safe, syntactically correct SQL query\n"
        "4. Use SELECT, INSERT, UPDATE, or DELETE statements as appropriate\n"
        "5. For INSERT: Generate valid INSERT statements with proper values\n"
        "6. For UPDATE: Generate UPDATE statements with WHERE clauses to target specific records\n"
        "7. For DELETE: Generate DELETE statements with WHERE clauses to target specific records\n"
        "8. Never use DROP, TRUNCATE, or ALTER statements\n"
        "9. Use only the tables and columns provided in the schema context\n"
        "10. Consider the conversation history for context and follow-up questions\n"
        "11. If this is a follow-up question, use context from previous messages to understand what the user is referring to\n"
        "12. For name or text fields, use ILIKE and wildcards for partial, case-insensitive matches (e.g., WHERE full_name ILIKE '%zahid%').\n\n"
        "OUTPUT FORMAT: Generate ONLY the complete SQL query, no explanations, no markdown, no code blocks, no prefixes.\n"
        "Preserve all SQL clauses including WHERE, ORDER BY, GROUP BY, HAVING, etc.\n"
        "Example outputs:\n"
        "- SELECT: SELECT * FROM customers WHERE active = true;\n"
        "- INSERT: INSERT INTO customers (name, email, phone) VALUES ('John Doe', 'john@example.com', '1234567890');\n"
        "- UPDATE: UPDATE customers SET phone = '0987654321' WHERE id = 1;\n"
        "- DELETE: DELETE FROM customers WHERE id = 1;\n"
        "If the request cannot be handled with a valid SQL query, reply: 'Operation not allowed.'\n"
        "If no relevant tables are found in the schema, reply: 'No relevant tables found in schema.'\n\n"
        "SQL QUERY:"
    )
    return sql_prompt

def clean_sql_from_llm(sql_response):
    # Remove markdown/code block and common prefixes, as in your main.py
    sql_query = sql_response.strip()
    if sql_query.startswith('```'):
        lines = sql_query.split('\n')
        if len(lines) > 1:
            sql_lines = []
            for line in lines[1:]:
                if line.strip() == '```':
                    break
                sql_lines.append(line)
            sql_query = '\n'.join(sql_lines).strip()
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
    sql_query = sql_query.rstrip(';').strip()
    return sql_query

def format_db_result(mcp_result):
    # User-friendly formatting for DB results
    if isinstance(mcp_result, dict):
        if mcp_result.get('success') and mcp_result.get('results'):
            rows = mcp_result['results']
            if not rows:
                return "No results found."
            headers = list(rows[0].keys())
            if len(rows) == 1:
                # Single row: return as a summary sentence
                row = rows[0]
                summary = ", ".join(f"{h}: {row.get(h, '')}" for h in headers)
                return f"Result: {summary}"
            # Multi-row: Build a table
            lines = [" | ".join(headers)]
            lines.append("-|-".join(["---"] * len(headers)))
            for row in rows:
                lines.append(" | ".join(str(row.get(h, '')) for h in headers))
            return "\n".join(lines)
        elif mcp_result.get('success'):
            return "Query executed successfully, but no results found."
        elif mcp_result.get('error'):
            return f"Error: {mcp_result['error']}"
        else:
            # Fallback: pretty-print JSON
            import json
            return json.dumps(mcp_result, indent=2)
    # Try to parse as JSON string
    try:
        import json
        parsed = json.loads(mcp_result)
        return format_db_result(parsed)
    except Exception:
        pass
    # Fallback: return as-is
    return str(mcp_result) 