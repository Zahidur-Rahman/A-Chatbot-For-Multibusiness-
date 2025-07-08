import asyncio
import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional
from collections import defaultdict
import threading
import time

import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import SimpleConnectionPool
from dotenv import load_dotenv
import traceback

# MCP imports
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolRequest,
    CallToolResult,
    ListToolsRequest,
    ListToolsResult,
    TextContent,
    Tool,
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("multi-business-mcp-server")

class DummyNotificationOptions:
    tools_changed = None

class ConnectionPoolManager:
    """Manages database connection pools for multiple businesses"""
    
    def __init__(self):
        self.pools = {}
        self.business_configs = {}
        self.lock = threading.Lock()
        self.health_check_interval = 300  # 5 minutes
        self.last_health_check = defaultdict(int)
        self._load_business_configs()
    
    def _load_business_configs(self):
        """Load business configurations from environment variables"""
        # Load business IDs from environment
        business_ids = os.getenv("BUSINESS_IDS", "").split(",")
        
        for business_id in business_ids:
            business_id = business_id.strip()
            if not business_id:
                continue
                
            # Load business-specific config from environment
            prefix = f"BUSINESS_{business_id.upper()}_"
            config = {
                "host": os.getenv(f"{prefix}POSTGRES_HOST"),
                "database": os.getenv(f"{prefix}POSTGRES_DB"),
                "user": os.getenv(f"{prefix}POSTGRES_USER"),
                "password": os.getenv(f"{prefix}POSTGRES_PASSWORD"),
                "port": int(os.getenv(f"{prefix}POSTGRES_PORT", "5432")),
                # Add connection pool settings
                "minconn": int(os.getenv(f"{prefix}MIN_CONNECTIONS", "2")),
                "maxconn": int(os.getenv(f"{prefix}MAX_CONNECTIONS", "10")),
                # Add connection timeout settings
                "connect_timeout": int(os.getenv(f"{prefix}CONNECT_TIMEOUT", "30")),
                "keepalives": 1,
                "keepalives_idle": 30,
                "keepalives_interval": 10,
                "keepalives_count": 5,
            }
            
            # Validate that all required config is present
            required_fields = ["host", "database", "user", "password"]
            if all(config.get(field) for field in required_fields):
                self.business_configs[business_id] = config
                logger.info(f"Loaded configuration for business: {business_id}")
            else:
                missing_fields = [field for field in required_fields if not config.get(field)]
                logger.warning(f"Incomplete configuration for business: {business_id}, missing: {missing_fields}")
    
    def _create_connection_pool(self, business_id: str) -> SimpleConnectionPool:
        """Create a new connection pool for a business"""
        config = self.business_configs[business_id]
        
        # Extract pool-specific settings
        minconn = config.pop("minconn", 2)
        maxconn = config.pop("maxconn", 10)
        
        try:
            pool = SimpleConnectionPool(
                minconn=minconn,
                maxconn=maxconn,
                **config
            )
            logger.info(f"Created connection pool for business: {business_id} (min={minconn}, max={maxconn})")
            return pool
        except Exception as e:
            logger.error(f"Failed to create pool for business {business_id}: {e}")
            raise
    
    def get_pool(self, business_id: str) -> SimpleConnectionPool:
        """Get or create a connection pool for a business"""
        with self.lock:
            if business_id not in self.pools:
                if business_id not in self.business_configs:
                    raise ValueError(f"Business '{business_id}' not configured. Available businesses: {list(self.business_configs.keys())}")
                
                self.pools[business_id] = self._create_connection_pool(business_id)
            
            return self.pools[business_id]
    
    def get_connection(self, business_id: str):
        """Get a connection from the pool with health check"""
        current_time = time.time()
        
        # Perform health check if needed
        if current_time - self.last_health_check[business_id] > self.health_check_interval:
            self._health_check(business_id)
            self.last_health_check[business_id] = current_time
        
        pool = self.get_pool(business_id)
        try:
            connection = pool.getconn()
            if connection.closed:
                logger.warning(f"Got closed connection for business {business_id}, attempting to recreate")
                self._recreate_pool(business_id)
                connection = pool.getconn()
            return connection
        except Exception as e:
            logger.error(f"Failed to get connection for business {business_id}: {e}")
            raise
    
    def return_connection(self, business_id: str, connection, error=False):
        """Return a connection to the pool"""
        if business_id in self.pools:
            try:
                if error or connection.closed:
                    # Don't return bad connections to the pool
                    connection.close()
                    logger.warning(f"Discarded bad connection for business {business_id}")
                else:
                    self.pools[business_id].putconn(connection)
            except Exception as e:
                logger.error(f"Error returning connection for business {business_id}: {e}")
    
    def _health_check(self, business_id: str):
        """Perform health check on a business connection"""
        try:
            connection = self.get_connection(business_id)
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
                cursor.fetchone()
            self.return_connection(business_id, connection)
            logger.debug(f"Health check passed for business: {business_id}")
        except Exception as e:
            logger.warning(f"Health check failed for business {business_id}: {e}")
            self._recreate_pool(business_id)
    
    def _recreate_pool(self, business_id: str):
        """Recreate connection pool for a business"""
        with self.lock:
            if business_id in self.pools:
                try:
                    self.pools[business_id].closeall()
                except Exception as e:
                    logger.error(f"Error closing old pool for business {business_id}: {e}")
                
                self.pools[business_id] = self._create_connection_pool(business_id)
                logger.info(f"Recreated connection pool for business: {business_id}")
    
    def close_all_pools(self):
        """Close all connection pools"""
        with self.lock:
            for business_id, pool in self.pools.items():
                try:
                    pool.closeall()
                    logger.info(f"Closed connection pool for business: {business_id}")
                except Exception as e:
                    logger.error(f"Error closing pool for business {business_id}: {e}")
            self.pools.clear()
    
    def list_businesses(self) -> List[str]:
        """List all configured businesses"""
        return list(self.business_configs.keys())
    
    def get_business_info(self, business_id: str) -> Dict[str, Any]:
        """Get business configuration info (without sensitive data)"""
        if business_id not in self.business_configs:
            return {}
        
        config = self.business_configs[business_id]
        return {
            "business_id": business_id,
            "host": config["host"],
            "database": config["database"],
            "port": config["port"],
            "user": config["user"],
            "pool_status": "active" if business_id in self.pools else "inactive"
        }

class MultiBusinessPostgreSQLServer:
    """Multi-Business PostgreSQL MCP Server implementation"""
    
    def __init__(self):
        self.server = Server("multi-business-postgres-server")
        self.pool_manager = ConnectionPoolManager()
        self.query_timeout = int(os.getenv("QUERY_TIMEOUT", "30"))  # seconds
        self._setup_handlers()
    
    def get_db_connection(self, business_id: str):
        """Get database connection for a specific business"""
        try:
            return self.pool_manager.get_connection(business_id)
        except Exception as e:
            logger.error(f"Database connection error for business {business_id}: {e}")
            raise
    
    def _setup_handlers(self):
        """Setup MCP request handlers"""
        
        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            """List available tools"""
            return [
                Tool(
                    name="execute_query",
                    description="Execute a SQL query and return results. Only SELECT queries are allowed for security.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The SQL query to execute (SELECT statements only)"
                            },
                            "business_id": {
                                "type": "string",
                                "description": "Business ID to execute query against (REQUIRED)"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Optional limit for result rows (default: 1000, max: 10000)",
                                "minimum": 1,
                                "maximum": 10000
                            }
                        },
                        "required": ["query", "business_id"]
                    }
                ),
                Tool(
                    name="get_table_schema",
                    description="Get detailed schema information for a specific table",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "table_name": {
                                "type": "string",
                                "description": "Name of the table to get schema for"
                            },
                            "business_id": {
                                "type": "string",
                                "description": "Business ID (REQUIRED)"
                            }
                        },
                        "required": ["table_name", "business_id"]
                    }
                ),
                Tool(
                    name="list_tables",
                    description="List all tables in the public schema",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "business_id": {
                                "type": "string",
                                "description": "Business ID (REQUIRED)"
                            }
                        },
                        "required": ["business_id"]
                    }
                ),
                Tool(
                    name="list_businesses",
                    description="List all available businesses with their connection status",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                ),
                Tool(
                    name="get_business_info",
                    description="Get detailed information about a specific business",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "business_id": {
                                "type": "string",
                                "description": "Business ID (REQUIRED)"
                            }
                        },
                        "required": ["business_id"]
                    }
                )
            ]
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            """Handle tool calls"""
            try:
                if name == "list_businesses":
                    result = await self._list_businesses()
                elif name == "get_business_info":
                    business_id = arguments.get("business_id")
                    if not business_id:
                        return [TextContent(type="text", text=json.dumps({
                            "error": "business_id is required for this operation"
                        }))]
                    result = await self._get_business_info(business_id)
                else:
                    # All other tools require business_id
                    business_id = arguments.get("business_id")
                    if not business_id:
                        return [TextContent(type="text", text=json.dumps({
                            "error": "business_id is required for this operation",
                            "tool": name,
                            "available_businesses": self.pool_manager.list_businesses()
                        }))]
                    
                    if name == "execute_query":
                        query = arguments.get("query", "")
                        limit = arguments.get("limit", 1000)
                        result = await self._execute_query(query, business_id, limit)
                    elif name == "get_table_schema":
                        result = await self._get_table_schema(arguments.get("table_name", ""), business_id)
                    elif name == "list_tables":
                        result = await self._list_tables(business_id)
                    else:
                        return [TextContent(type="text", text=json.dumps({
                            "error": f"Unknown tool: {name}"
                        }))]
                
                return [TextContent(type="text", text=result)]
                
            except Exception as e:
                logger.error(f"Tool call error for {name}: {e}")
                return [TextContent(type="text", text=json.dumps({
                    "error": str(e),
                    "tool": name,
                    "traceback": traceback.format_exc() if logger.level <= logging.DEBUG else None
                }))]
    
    async def _execute_query(self, query: str, business_id: str, limit: int = 1000) -> str:
        """Execute a SQL query with security checks"""
        conn = None
        error_occurred = False
        
        try:
            # Input validation
            query = query.strip()
            if not query:
                return json.dumps({"error": "Empty query provided"})
            
            # Validate limit
            limit = max(1, min(limit, 10000))
            
            # Security check - only allow SELECT statements
            if not query.upper().startswith('SELECT'):
                return json.dumps({
                    "error": "Only SELECT queries are allowed for security reasons",
                    "provided_query": query[:100] + "..." if len(query) > 100 else query
                })
            
            # Check for dangerous patterns
            dangerous_patterns = [
                'DROP', 'DELETE', 'TRUNCATE', 'ALTER', 'CREATE', 'INSERT', 'UPDATE',
                'GRANT', 'REVOKE', 'COPY', 'CALL', 'EXECUTE', 'IMPORT', 'EXPORT'
            ]
            
            query_upper = query.upper()
            for pattern in dangerous_patterns:
                if pattern in query_upper:
                    return json.dumps({
                        "error": f"Query contains forbidden keyword: {pattern}",
                        "provided_query": query[:100] + "..." if len(query) > 100 else query
                    })
            
            # Get business-specific connection
            conn = self.get_db_connection(business_id)
            
            # Add automatic LIMIT if not present
            if 'LIMIT' not in query_upper:
                query += f" LIMIT {limit}"
            
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # Set query timeout
                cursor.execute(f"SET statement_timeout = {self.query_timeout * 1000}")
                
                start_time = time.time()
                cursor.execute(query)
                execution_time = time.time() - start_time
                
                results = cursor.fetchall()
                
                # Convert to JSON-serializable format
                json_results = []
                for row in results:
                    json_row = {}
                    for key, value in row.items():
                        # Handle special types that aren't JSON serializable
                        if hasattr(value, 'isoformat'):  # datetime objects
                            json_row[key] = value.isoformat()
                        elif isinstance(value, (bytes, memoryview)):
                            json_row[key] = str(value)
                        elif hasattr(value, '__class__') and value.__class__.__name__ == 'Decimal':
                            # Handle Decimal objects from PostgreSQL NUMERIC/DECIMAL columns
                            json_row[key] = float(value)
                        elif isinstance(value, (int, float, str, bool, type(None))):
                            # Basic JSON-serializable types
                            json_row[key] = value
                        else:
                            # Fallback for any other non-serializable types
                            json_row[key] = str(value)
                    json_results.append(json_row)
                
                return json.dumps({
                    "success": True,
                    "business_id": business_id,
                    "query": query,
                    "row_count": len(json_results),
                    "execution_time_seconds": round(execution_time, 3),
                    "results": json_results
                }, indent=2)
                    
        except psycopg2.Error as e:
            error_occurred = True
            logger.error(f"Database error for business {business_id}: {e}")
            return json.dumps({
                "error": f"Database error: {str(e)}",
                "business_id": business_id,
                "query": query[:100] + "..." if len(query) > 100 else query
            })
        except Exception as e:
            error_occurred = True
            logger.error(f"Unexpected error for business {business_id}: {e}")
            return json.dumps({
                "error": f"Unexpected error: {str(e)}",
                "business_id": business_id,
                "query": query[:100] + "..." if len(query) > 100 else query
            })
        finally:
            if conn:
                self.pool_manager.return_connection(business_id, conn, error=error_occurred)
    
    async def _get_table_schema(self, table_name: str, business_id: str) -> str:
        """Get detailed schema information for a table"""
        conn = None
        error_occurred = False
        
        try:
            if not table_name:
                return json.dumps({"error": "Table name is required"})
            
            conn = self.get_db_connection(business_id)
            with conn.cursor() as cursor:
                # Get column information
                cursor.execute("""
                    SELECT 
                        column_name,
                        data_type,
                        character_maximum_length,
                        is_nullable,
                        column_default,
                        ordinal_position
                    FROM information_schema.columns
                    WHERE table_name = %s AND table_schema = 'public'
                    ORDER BY ordinal_position;
                """, (table_name,))
                
                columns = cursor.fetchall()
                if not columns:
                    return json.dumps({
                        "error": f"Table '{table_name}' not found in business '{business_id}'",
                        "business_id": business_id
                    })
                
                # Get primary key information
                cursor.execute("""
                    SELECT column_name
                    FROM information_schema.key_column_usage
                    WHERE table_name = %s AND table_schema = 'public'
                    AND constraint_name IN (
                        SELECT constraint_name
                        FROM information_schema.table_constraints
                        WHERE table_name = %s AND constraint_type = 'PRIMARY KEY'
                    );
                """, (table_name, table_name))
                
                primary_keys = [row[0] for row in cursor.fetchall()]
                
                schema_info = []
                for col in columns:
                    col_info = {
                        "name": col[0],
                        "type": col[1],
                        "max_length": col[2],
                        "nullable": col[3] == "YES",
                        "default": col[4],
                        "position": col[5],
                        "is_primary_key": col[0] in primary_keys
                    }
                    schema_info.append(col_info)
                
                return json.dumps({
                    "business_id": business_id,
                    "table_name": table_name,
                    "columns": schema_info,
                    "column_count": len(schema_info),
                    "primary_keys": primary_keys
                }, indent=2)
                    
        except Exception as e:
            error_occurred = True
            logger.error(f"Schema error for business {business_id}: {e}")
            return json.dumps({
                "error": f"Error getting schema: {str(e)}",
                "business_id": business_id,
                "table_name": table_name
            })
        finally:
            if conn:
                self.pool_manager.return_connection(business_id, conn, error=error_occurred)
    
    async def _list_tables(self, business_id: str) -> str:
        """List all tables in the public schema"""
        conn = None
        error_occurred = False
        
        try:
            conn = self.get_db_connection(business_id)
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT 
                        table_name,
                        table_type
                    FROM information_schema.tables 
                    WHERE table_schema = 'public'
                    ORDER BY table_name;
                """)
                
                tables = cursor.fetchall()
                table_list = [{"name": table[0], "type": table[1]} for table in tables]
                
                return json.dumps({
                    "business_id": business_id,
                    "tables": table_list,
                    "count": len(table_list)
                }, indent=2)
                    
        except Exception as e:
            error_occurred = True
            logger.error(f"List tables error for business {business_id}: {e}")
            return json.dumps({
                "error": f"Error listing tables: {str(e)}",
                "business_id": business_id
            })
        finally:
            if conn:
                self.pool_manager.return_connection(business_id, conn, error=error_occurred)
    
    async def _list_businesses(self) -> str:
        """List all available businesses with their status"""
        try:
            businesses = self.pool_manager.list_businesses()
            business_info = []
            
            for business_id in businesses:
                info = self.pool_manager.get_business_info(business_id)
                business_info.append(info)
            
            return json.dumps({
                "businesses": business_info,
                "count": len(business_info)
            }, indent=2)
        except Exception as e:
            logger.error(f"List businesses error: {e}")
            return json.dumps({"error": f"Error listing businesses: {str(e)}"})
    
    async def _get_business_info(self, business_id: str) -> str:
        """Get detailed information about a specific business"""
        try:
            info = self.pool_manager.get_business_info(business_id)
            if not info:
                return json.dumps({
                    "error": f"Business '{business_id}' not found",
                    "available_businesses": self.pool_manager.list_businesses()
                })
            
            return json.dumps(info, indent=2)
        except Exception as e:
            logger.error(f"Get business info error: {e}")
            return json.dumps({"error": f"Error getting business info: {str(e)}"})

async def main():
    """Main function to run the MCP server"""
    postgres_server = None
    try:
        # Create server
        postgres_server = MultiBusinessPostgreSQLServer()
        
        # Check if any businesses are configured
        businesses = postgres_server.pool_manager.list_businesses()
        if not businesses:
            logger.error("No businesses configured. Please set BUSINESS_IDS and business-specific environment variables.")
            logger.error("Example: BUSINESS_IDS=biz1,biz2 BUSINESS_BIZ1_POSTGRES_HOST=localhost ...")
            sys.exit(1)
        
        logger.info(f"Configured businesses: {businesses}")
        
        # Test connections to all businesses
        for business_id in businesses:
            try:
                conn = postgres_server.get_db_connection(business_id)
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    cursor.fetchone()
                postgres_server.pool_manager.return_connection(business_id, conn)
                logger.info(f"Database connection test successful for business: {business_id}")
            except Exception as e:
                logger.error(f"Database connection test failed for business {business_id}: {e}")
                # Continue anyway - connection might be temporary issue
        
        # Run the MCP server with stdio transport
        async with stdio_server() as (read_stream, write_stream):
            logger.info("Starting Multi-Business PostgreSQL MCP server...")
            await postgres_server.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="multi-business-postgres-server",
                    server_version="1.0.0",
                    capabilities=postgres_server.server.get_capabilities(DummyNotificationOptions(), {}),
                ),
            )
            
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.error(f"Server error: {e}")
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Clean up connection pools
        if postgres_server:
            postgres_server.pool_manager.close_all_pools()
            logger.info("Connection pools closed")

if __name__ == "__main__":
    # Set event loop policy for Windows compatibility
    logger.info("Checking OS compatibility...")
    if sys.platform.startswith('win'):
        logger.info("Windows detected, setting ProactorEventLoopPolicy")
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    asyncio.run(main())