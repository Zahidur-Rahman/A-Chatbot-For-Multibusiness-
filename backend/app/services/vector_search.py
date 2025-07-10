import os
import faiss
import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from backend.app.services.mongodb_service import mongodb_service
from backend.app.config import get_settings
from backend.app.models.business import BusinessSchema
import logging

settings = get_settings()

logger = logging.getLogger("vector_search_service")

class FaissVectorSearchService:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index_dir = settings.vector_search.faiss_index_path or './faiss_indices'
        os.makedirs(self.index_dir, exist_ok=True)
        self.mongo_service = mongodb_service
        self.indices: Dict[str, faiss.IndexFlatL2] = {}  # business_id -> index
        self.schema_id_map: Dict[str, List[str]] = {}    # business_id -> [schema_id]

    def _get_index_path(self, business_id: str) -> str:
        return os.path.join(self.index_dir, f'{business_id}_schema.index')

    def _load_index(self, business_id: str):
        path = self._get_index_path(business_id)
        if os.path.exists(path):
            index = faiss.read_index(path)
        else:
            index = faiss.IndexFlatL2(self.model.get_sentence_embedding_dimension())
        self.indices[business_id] = index
        return index

    def _save_index(self, business_id: str):
        path = self._get_index_path(business_id)
        faiss.write_index(self.indices[business_id], path)

    async def index_business_schemas(self, business_id: str):
        """Index all schemas (including relationships) for a business."""
        logger.info(f"[Indexing] Starting schema indexing for business: {business_id}")
        schemas: List[BusinessSchema] = await self.mongo_service.get_business_schemas(business_id)
        texts = []
        schema_ids = []
        for schema in schemas:
            # Combine schema description, columns, and relationships
            col_desc = ", ".join([f"{col.name}: {col.description or col.business_meaning or col.type}" for col in schema.columns])
            rel_desc = "; ".join([f"{rel.from_table}({rel.from_column}) -> {rel.to_table}({rel.to_column})" for rel in schema.relationships])
            embedding_text = f"Table: {schema.table_name}. Description: {schema.schema_description}. Columns: {col_desc}. Relationships: {rel_desc}"
            texts.append(embedding_text)
            schema_ids.append(str(schema.id))
        if not texts:
            logger.info(f"[Indexing] No schemas found for business: {business_id}")
            return
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        self.indices[business_id] = index
        self.schema_id_map[business_id] = schema_ids
        self._save_index(business_id)
        logger.info(f"[Indexing] Finished indexing for business: {business_id}. Indexed {len(schemas)} schemas.")

    async def search_schemas(self, business_id: str, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant schemas (including relationships) for a business given a query."""
        if business_id not in self.indices:
            self._load_index(business_id)
        if business_id not in self.indices:
            return []

        # Vector search
        query_emb = self.model.encode([query], convert_to_numpy=True)
        D, I = self.indices[business_id].search(query_emb, top_k)
        results = []
        for idx in I[0]:
            if idx < 0 or idx >= len(self.schema_id_map[business_id]):
                continue
            schema_id = self.schema_id_map[business_id][idx]
            schema = await self.mongo_service.get_business_schema(business_id, schema_id)
            if schema:
                results.append(schema.dict())

        # Fallback: If no results, try keyword match on table name and schema description
        if not results:
            logger.info(f"[VectorSearch] No vector results for '{query}'. Trying keyword fallback.")
            schemas = await self.mongo_service.get_business_schemas(business_id)
            query_lower = query.lower()
            for schema in schemas:
                if (query_lower in schema.table_name.lower() or
                    (schema.schema_description and query_lower in schema.schema_description.lower())):
                    results.append(schema.dict())
            # If still no results and query is very generic, return all schemas
            if not results and query_lower.strip() in ["menu", "data", "table", "tables"]:
                results = [s.dict() for s in schemas]

        return results 