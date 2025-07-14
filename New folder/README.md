# Multi-Business Conversational Chatbot

A production-ready, dynamic multi-business conversational chatbot with PostgreSQL integration, vector-based schema discovery, and LangChain conversational AI.

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    CLIENT APPLICATIONS                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐           │
│  │   Web App   │  │ Mobile App  │  │   API Client │           │
│  └─────────────┘  └─────────────┘  └─────────────┘           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     API GATEWAY                                 │
│              ┌─────────────────────────────────┐               │
│              │     FastAPI Application         │               │
│              │   • Authentication Middleware   │               │
│              │   • Rate Limiting              │               │
│              │   • Request Validation         │               │
│              └─────────────────────────────────┘               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 CONVERSATION ORCHESTRATOR                       │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              LangChain Conversation Engine                   ││
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐    ││
│  │  │Conversation │  │   Memory    │  │  Output Format  │    ││
│  │  │   Chain     │  │  Manager    │  │   Controller    │    ││
│  │  └─────────────┘  └─────────────┘  └─────────────────┘    ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
                              │
                ┌─────────────┼─────────────┐
                ▼             ▼             ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  VECTOR SEARCH  │  │  SCHEMA MANAGER │  │  QUERY ENGINE   │
│                 │  │                 │  │                 │
│ ┌─────────────┐ │  │ ┌─────────────┐ │  │ ┌─────────────┐ │
│ │ Embeddings  │ │  │ │  MongoDB    │ │  │ │ SQL Builder │ │
│ │   Engine    │ │  │ │  Schemas    │ │  │ │   LLM       │ │
│ └─────────────┘ │  │ └─────────────┘ │  │ └─────────────┘ │
│ ┌─────────────┐ │  │ ┌─────────────┐ │  │ ┌─────────────┐ │
│ │  Pinecone/  │ │  │ │ Auth Cache  │ │  │ │ Validation  │ │
│ │   Weaviate  │ │  │ │   Redis     │ │  │ │   Engine    │ │
│ └─────────────┘ │  │ └─────────────┘ │  │ └─────────────┘ │
└─────────────────┘  └─────────────────┘  └─────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      MCP SERVER                                 │
│              ┌─────────────────────────────────┐               │
│              │   Dynamic Connection Manager    │               │
│              │                                 │               │
│              │  ┌─────────────────────────────┐│               │
│              │  │    PostgreSQL Pool Manager  ││               │
│              │  │  • Business-specific pools  ││               │
│              │  │  • Connection health check  ││               │
│              │  │  • Auto-scaling pools      ││               │
│              │  └─────────────────────────────┘│               │
│              └─────────────────────────────────┘               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  BUSINESS DATABASES                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐           │
│  │Business A   │  │Business B   │  │Business C   │           │
│  │PostgreSQL   │  │PostgreSQL   │  │PostgreSQL   │           │
│  │   Database  │  │   Database  │  │   Database  │           │
│  └─────────────┘  └─────────────┘  └─────────────┘           │
└─────────────────────────────────────────────────────────────────┘
```

## 🎯 Core Features

- **Dynamic Multi-Business Support**: On-board/off-board businesses dynamically
- **Vector-Based Schema Discovery**: Intelligent query generation using embeddings
- **LangChain Conversational AI**: Persistent memory and context-aware conversations
- **Production-Grade Security**: Multi-tenant authentication and data isolation
- **Scalable Architecture**: Connection pooling and horizontal scaling support

## 📁 Project Structure

```
M_Chatbot/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                 # FastAPI application
│   │   ├── config.py               # Configuration management
│   │   ├── auth/
│   │   │   ├── __init__.py
│   │   │   ├── jwt_handler.py      # JWT authentication
│   │   │   └── permissions.py      # Role-based access control
│   │   ├── mcp/
│   │   │   ├── __init__.py
│   │   │   ├── server.py           # MCP server (enhanced)
│   │   │   └── connection_manager.py # Connection pooling
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── business.py         # Business models
│   │   │   ├── user.py             # User models
│   │   │   └── conversation.py     # Conversation models
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── conversation.py     # LangChain integration
│   │   │   ├── vector_search.py    # Schema discovery
│   │   │   └── query_engine.py     # SQL generation
│   │   └── utils/
│   │       ├── __init__.py
│   │       └── helpers.py          # Utility functions
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/                       # (Future: React/Vue dashboard)
├── docs/
│   ├── api.md                      # API documentation
│   └── deployment.md               # Deployment guide
├── tests/
│   ├── test_mcp_server.py
│   └── test_conversation.py
├── .env.example                    # Environment variables template
├── docker-compose.yml              # Development environment
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- PostgreSQL
- MongoDB (for business configurations)
- Redis (for caching)

### 1. Clone and Setup

```bash
git clone <your-repo>
cd M_Chatbot
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 3. Environment Configuration

```bash
cp .env.example .env
# Edit .env with your database configurations
```

### 4. Run Development Server

```bash
# Start the MCP server
python app/mcp/server.py

# Start the FastAPI application (in another terminal)
uvicorn app.main:app --reload
```

## 🔧 Configuration

### Environment Variables

```bash
# Database Configuration
POSTGRES_HOST=localhost
POSTGRES_DB=resturent
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_PORT=5432

# Multi-Business Configuration
BUSINESS_IDS=default,business_a,business_b

# Business A Configuration
BUSINESS_BUSINESS_A_POSTGRES_HOST=localhost
BUSINESS_BUSINESS_A_POSTGRES_DB=business_a_db
BUSINESS_BUSINESS_A_POSTGRES_USER=business_a_user
BUSINESS_BUSINESS_A_POSTGRES_PASSWORD=business_a_pass

# MongoDB Configuration
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB=chatbot_config

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379

# JWT Configuration
JWT_SECRET_KEY=your-secret-key
JWT_ALGORITHM=HS256
JWT_EXPIRATION=3600

# OpenAI Configuration
OPENAI_API_KEY=your-openai-key
```

## 📊 Implementation Phases

### Phase 1: Core Infrastructure ✅
- [x] Dynamic MCP server with PostgreSQL pools
- [ ] Authentication and authorization system
- [ ] Basic schema storage in MongoDB
- [ ] Simple query execution pipeline

### Phase 2: Intelligence Layer
- [ ] Vector embeddings for schema search
- [ ] LangChain conversation engine
- [ ] Context-aware SQL generation
- [ ] Memory management system

### Phase 3: Production Features
- [ ] Advanced security and monitoring
- [ ] Performance optimization
- [ ] Scalability enhancements
- [ ] User experience improvements

### Phase 4: Enterprise Features
- [ ] Advanced analytics and reporting
- [ ] Custom business logic integration
- [ ] Multi-region deployment
- [ ] Advanced compliance features

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue in the repository
- Check the documentation in `/docs`
- Review the API documentation

---

**Built with ❤️ for scalable, multi-business conversational AI** 