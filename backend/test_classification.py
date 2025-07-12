import asyncio
import sys
import os

# Add the current directory to the path
sys.path.append(os.path.dirname(__file__))

from app.main import is_database_related_query_dynamic
from app.services.vector_search import FaissVectorSearchService

async def test_classification():
    print("Testing query classification...")
    
    # Initialize vector search service
    vs = FaissVectorSearchService()
    
    # Test the specific query
    query = "give me all information about Zahid"
    business_id = "resturent"
    
    print(f"Testing query: '{query}' for business: {business_id}")
    
    try:
        result = await is_database_related_query_dynamic(query, business_id, vs)
        print(f"Classification result: {result}")
    except Exception as e:
        print(f"Error during classification: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_classification()) 