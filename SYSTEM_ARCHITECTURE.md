# Multi-Business Conversational Chatbot System - Architecture & Flow

## System Overview

A sophisticated multi-business conversational chatbot system with FastAPI backend, supporting dynamic query classification, LLM-powered SQL generation, secure database execution, and intelligent conversation management.

## Core Components

### 1. Backend Architecture
- **FastAPI**: Main web framework with async support
- **PostgreSQL**: Business-specific databases with role-based access
- **MongoDB**: Configuration storage, conversation memory, and user management
- **Redis**: Session conversation caching with 1-hour TTL
- **Mistral LLM**: Natural language processing and SQL generation
- **FAISS**: Vector embeddings for semantic schema search

### 2. Authentication & Authorization
- **JWT Tokens**: 2-hour expiration with refresh capability
- **Role-Based Access**: Admin users with "all" business access
- **Business Isolation**: Automatic business_id detection and validation
- **Rate Limiting**: Redis-based rate limiting for chat and login endpoints

### 3. Database Management
- **MCP Server**: Secure SQL execution with validation
- **Read-Only Default**: SELECT queries only for security
- **Write Operations**: INSERT/UPDATE/DELETE with audit logging
- **Transaction Safety**: Automatic rollback on errors

## Complete System Flow

### 1. User Authentication Flow
```
User Login Request → JWT Token Generation → Token Validation → Business Access Check
```

**Steps:**
1. User provides credentials via `/auth/login`
2. System validates against MongoDB user collection
3. JWT token generated with user_id, business_id, and role
4. Token stored in Redis for rate limiting
5. Response includes access_token and user info

### 2. Chat Conversation Flow
```
User Query → Authentication → Business Context → Query Classification → LLM Processing → Response Generation → Dual Storage
```

**Detailed Steps:**

#### Step 1: Request Processing
- User sends query to `/chat` endpoint
- JWT token validated and user context extracted
- Business_id determined (admin users can omit for general conversations)

#### Step 2: Conversation Context Retrieval
- **Redis-First Strategy**: Check Redis for session conversation (1-hour TTL)
- **MongoDB Fallback**: If Redis miss, retrieve from MongoDB
- **Context Assembly**: Last 20 messages (10 user + 10 assistant) for LLM context

#### Step 3: Query Classification
- **Dynamic Classification**: LLM determines if query is general conversation or database query
- **Context-Aware**: Uses last 4 conversation messages for better classification
- **Primary Factor**: Current user query is the main classification driver
- **Secondary Factor**: Conversation context for clarification of ambiguous queries

#### Step 4: Schema Context Retrieval
- **Vector Search**: FAISS retrieves top 5 relevant database schemas
- **Semantic Matching**: Embeddings match user intent to schema structure
- **Context Assembly**: Schema metadata formatted for LLM consumption

#### Step 5: LLM Processing
- **System Prompt Construction**: Combines conversation history + schema context
- **Query Generation**: Mistral LLM generates appropriate response
- **SQL Generation**: For database queries, generates executable SQL
- **Response Formatting**: Structured response with confidence scores

#### Step 6: Database Execution (if applicable)
- **MCP Server**: Secure SQL execution with validation
- **Forbidden Keywords**: Regex-based validation (CREATE, DROP, etc.)
- **Result Processing**: Format database results for user consumption
- **Error Handling**: Graceful error messages and rollback

#### Step 7: Response Delivery & Storage
- **Dual Storage**: Conversation saved to both Redis and MongoDB
- **Redis Cache**: Session conversation with 1-hour TTL
- **MongoDB Persistence**: Long-term conversation storage
- **Response Delivery**: Formatted response sent to user

### 3. Database Query Flow
```
Natural Language → SQL Generation → MCP Validation → Execution → Result Processing → Audit Logging
```

**Steps:**
1. LLM generates SQL from natural language query
2. MCP server validates SQL syntax and security
3. Forbidden keywords checked with regex word boundaries
4. SQL executed against business-specific PostgreSQL database
5. Results formatted and returned to user
6. Audit log created for write operations

### 4. Vector Search Flow
```
Schema Changes → Embedding Generation → FAISS Index Update → Semantic Search → Context Retrieval
```

**Steps:**
1. Database schema changes trigger embedding regeneration
2. FAISS index updated with new embeddings
3. User queries matched against schema embeddings
4. Top 5 relevant schemas retrieved
5. Schema metadata formatted for LLM context

## Storage Architecture

### 1. Redis Usage
- **Session Conversations**: 1-hour TTL for fast retrieval
- **Rate Limiting**: Request counting and throttling
- **Cache Strategy**: Redis-first with MongoDB fallback

### 2. MongoDB Collections
- **users**: User accounts, roles, and business associations
- **conversations**: Long-term conversation storage
- **business_configs**: Business-specific configurations
- **audit_logs**: Write operation tracking

### 3. PostgreSQL Databases
- **Business-Specific**: Separate database per business
- **Schema Management**: Automatic schema discovery and caching
- **Data Isolation**: Role-based access control

## Security Features

### 1. Authentication Security
- JWT tokens with 2-hour expiration
- Redis-based rate limiting
- Role-based access control
- Business isolation

### 2. Database Security
- Read-only by default
- SQL injection prevention
- Forbidden keyword validation
- Transaction safety
- Audit logging for write operations

### 3. API Security
- Input validation
- Error handling without data leakage
- Secure headers and CORS configuration

## Performance Optimizations

### 1. Caching Strategy
- **Redis Conversation Cache**: 1-hour TTL for session data
- **Vector Search**: FAISS for fast semantic search
- **Schema Caching**: Automatic refresh on schema changes

### 2. Database Optimization
- **Connection Pooling**: Efficient database connections
- **Query Optimization**: LLM-generated optimized SQL
- **Index Management**: Automatic index creation

### 3. Response Optimization
- **Async Processing**: Non-blocking operations
- **Context Limiting**: Last 20 messages for manageable context
- **Efficient Storage**: Dual storage with smart fallback

## Configuration Management

### 1. Environment Variables
- Database connections (PostgreSQL, MongoDB, Redis)
- LLM API keys and endpoints
- JWT secrets and expiration times
- Rate limiting parameters
- Debug and logging settings

### 2. Business Configuration
- Per-business database schemas
- Custom prompts and instructions
- Access control policies
- Rate limiting rules

## Monitoring & Debugging

### 1. Debug Endpoints
- `/debug/cache-stats`: Redis cache statistics
- `/debug/redis-conversation`: Session conversation data
- `/debug/search-stats`: Vector search performance
- `/debug/sql`: SQL generation testing

### 2. Logging
- Request/response logging
- Error tracking and debugging
- Performance metrics
- Audit trail for write operations

## Error Handling

### 1. Graceful Degradation
- Redis fallback to MongoDB
- Vector search fallback mechanisms
- Database connection retry logic
- LLM API error handling

### 2. User-Friendly Errors
- Clear error messages
- Actionable feedback
- Security-conscious error responses

## Future Enhancements

### 1. Multi-Query Support
- LangGraph integration for complex workflows
- Sequential query execution
- State management for multi-step operations

### 2. Advanced Features
- Cross-session context sharing
- Real-time collaboration
- Advanced analytics and reporting
- Custom business logic integration

## Deployment Considerations

### 1. Docker Support
- Containerized application
- Environment-specific configurations
- Easy scaling and deployment

### 2. Production Readiness
- Health checks and monitoring
- Backup and recovery procedures
- Security hardening
- Performance tuning

This system provides a robust, scalable, and secure foundation for multi-business conversational AI with intelligent database interaction capabilities. 