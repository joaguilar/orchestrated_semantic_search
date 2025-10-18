import json
import logging
from typing import List, Dict, Any, Optional, Annotated
from elasticsearch import Elasticsearch
from openai import AzureOpenAI
from semantic_kernel.functions import kernel_function


class ElasticsearchSearch:
    """
    Semantic Kernel plugin for performing semantic search on documents stored in Elasticsearch.
    Uses Azure OpenAI to generate embeddings and Elasticsearch vector search capabilities.
    """
    
    def __init__(self, es_host: str = "localhost", es_port: int = 9200, 
                 es_username: Optional[str] = None, es_password: Optional[str] = None,
                 es_api_key: Optional[str] = None, use_ssl: bool = False):
        """
        Initialize the Elasticsearch search plugin.
        
        Args:
            es_host: Elasticsearch host
            es_port: Elasticsearch port
            es_username: Username for basic auth
            es_password: Password for basic auth
            es_api_key: API key for authentication
            use_ssl: Whether to use SSL
        """
        self.logger = logging.getLogger(__name__)
        
        # Configure Elasticsearch client
        es_config = {
            'hosts': [{'host': es_host, 'port': es_port, 'scheme': 'https' if use_ssl else 'http'}],
            'verify_certs': use_ssl,
        }
        
        if es_api_key:
            es_config['api_key'] = es_api_key
        elif es_username and es_password:
            es_config['basic_auth'] = (es_username, es_password)
            
        self.es_client = Elasticsearch(**es_config)
        
        # Test connection
        try:
            if self.es_client.ping():
                self.logger.info("Successfully connected to Elasticsearch")
            else:
                self.logger.warning("Failed to connect to Elasticsearch")
        except Exception as e:
            self.logger.error(f"Error connecting to Elasticsearch: {e}")

    def _generate_embeddings(self, text: str, openai_client: AzureOpenAI, 
                           embedding_model: str = "text-embedding-3-small") -> List[float]:
        """
        Generate embeddings using Azure OpenAI.
        
        Args:
            text: Text to generate embeddings for
            openai_client: Azure OpenAI client
            embedding_model: Name of the embedding model
            
        Returns:
            List of float values representing the embedding
        """
        try:
            response = openai_client.embeddings.create(
                input=text,
                model=embedding_model
            )
            return response.data[0].embedding
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {e}")
            raise

    @kernel_function(
        description="Search for documents semantically using vector similarity in Elasticsearch. Takes a query text and returns relevant documents.",
        name="semantic_search",
    )
    async def semantic_search(
        self,
        query: Annotated[str, "The search query text to find semantically similar documents"],
        index_name: Annotated[str, "The name of the Elasticsearch index to search in"] = "documents",
        vector_field: Annotated[str, "The name of the field containing the document vectors/embeddings"] = "vector",
        text_fields: Annotated[str, "Comma-separated list of text fields to return from matching documents"] = "content",
        size: Annotated[int, "Maximum number of results to return"] = 5,
        min_score: Annotated[float, "Minimum similarity score threshold"] = 0.0,
        openai_api_key: Annotated[str, "Azure OpenAI API key for generating embeddings"] = "",
        openai_endpoint: Annotated[str, "Azure OpenAI endpoint URL"] = "",
        embedding_model: Annotated[str, "Name of the Azure OpenAI embedding model"] = "text-embedding-3-small",
        emmbedding: Annotated[List[float], "Precomputed embedding vector for the query text"] = None
    ) -> str:
        """
        Perform semantic search on Elasticsearch index using vector similarity.
        
        Returns:
            JSON string containing search results with document content and scores
        """
        try:
            text_fields_list = [field.strip() for field in text_fields.split(",")]
            
            if not query:
                return json.dumps({"error": "Query text is required"})
            
            if not index_name:
                return json.dumps({"error": "Index name is required"})
                
            if not openai_api_key or not openai_endpoint:
                return json.dumps({"error": "Azure OpenAI credentials are required"})
            
            print("Preparing to Query Elasticsearch...")
            # Initialize Azure OpenAI client
            openai_client = AzureOpenAI(
                api_key=openai_api_key,
                api_version="2024-02-01",
                azure_endpoint=openai_endpoint
            )
            
            # Generate embeddings for the query
            if not emmbedding:
                print(f"Generating embeddings for query: {query}")
                self.logger.info(f"Generating embeddings for query: {query}")
                query_vector = self._generate_embeddings(query, openai_client, embedding_model)
                print(f"Generated query vector, size: {len(query_vector) if query_vector else 'N/A'}")
            else:
                print(f"Using precomputed embedding vector, size: {len(emmbedding) if emmbedding else 'N/A'}")
                query_vector = emmbedding

            # Prepare Elasticsearch query
            es_query = {
                "size": size,
                "min_score": min_score,
                "_source": text_fields_list,
                "query": {
                    "script_score": {
                        "query": {"match_all": {}},
                        "script": {
                            "source": f"cosineSimilarity(params.query_vector, '{vector_field}') + 1.0",
                            "params": {"query_vector": query_vector}
                        }
                    }
                }
            }
            
            # Execute search
            self.logger.info(f"Executing semantic search on index: {index_name}")
            response = self.es_client.search(index=index_name, body=es_query)
            print("Elasticsearch query executed.")
            
            # Format results
            results = []
            for hit in response["hits"]["hits"]:
                result = {
                    "score": hit["_score"],
                    "document": hit["_source"][text_fields_list[0]] if len(text_fields_list) == 1 else {field: hit["_source"][field] for field in text_fields_list}
                }
                results.append(result)
            print(f"Elasticsearch search returned {len(results)} results.")
            
            search_results = {
                "query": query,
                "total_hits": response["hits"]["total"]["value"],
                "max_score": response["hits"]["max_score"],
                "results": results
            }
            
            self.logger.info(f"Found {len(results)} results for query: {query}")
            return json.dumps(search_results, indent=2)
            
        except Exception as e:
            error_msg = f"Error performing semantic search: {str(e)}"
            self.logger.error(error_msg)
            return json.dumps({"error": error_msg})

    @kernel_function(
        description="Search for documents using hybrid search (combining semantic vector search with keyword search) in Elasticsearch.",
        name="hybrid_search",
    )
    async def hybrid_search(
        self,
        query: Annotated[str, "The search query text"],
        index_name: Annotated[str, "The name of the Elasticsearch index to search in"] = "documents",
        vector_field: Annotated[str, "The name of the field containing document vectors/embeddings"] = "vector",
        text_fields: Annotated[str, "Comma-separated list of text fields to search and return"] = "content",
        keyword_fields: Annotated[str, "Comma-separated list of fields to perform keyword search on"] = "content",
        vector_weight: Annotated[float, "Weight for vector search component (0.0-1.0)"] = 0.7,
        keyword_weight: Annotated[float, "Weight for keyword search component (0.0-1.0)"] = 0.3,
        size: Annotated[int, "Maximum number of results to return"] = 5,
        openai_api_key: Annotated[str, "Azure OpenAI API key for generating embeddings"] = "",
        openai_endpoint: Annotated[str, "Azure OpenAI endpoint URL"] = "",
        embedding_model: Annotated[str, "Name of the Azure OpenAI embedding model"] = "text-embedding-3-small"
    ) -> str:
        """
        Perform hybrid search combining vector similarity and keyword matching.
        
        Returns:
            JSON string containing search results with combined scoring
        """
        try:
            text_fields_list = [field.strip() for field in text_fields.split(",")]
            keyword_fields_list = [field.strip() for field in keyword_fields.split(",")]
            
            if not query or not index_name:
                return json.dumps({"error": "Query and index name are required"})
                
            if not openai_api_key or not openai_endpoint:
                return json.dumps({"error": "Azure OpenAI credentials are required"})
            
            # Initialize Azure OpenAI client
            openai_client = AzureOpenAI(
                api_key=openai_api_key,
                api_version="2024-02-01",
                azure_endpoint=openai_endpoint
            )
            
            # Generate embeddings for the query
            query_vector = self._generate_embeddings(query, openai_client, embedding_model)
            
            # Prepare hybrid search query
            es_query = {
                "size": size,
                "_source": text_fields_list,
                "query": {
                    "bool": {
                        "should": [
                            {
                                "script_score": {
                                    "query": {"match_all": {}},
                                    "script": {
                                        "source": f"({vector_weight} * (cosineSimilarity(params.query_vector, '{vector_field}') + 1.0))",
                                        "params": {"query_vector": query_vector}
                                    }
                                }
                            },
                            {
                                "constant_score": {
                                    "filter": {
                                        "multi_match": {
                                            "query": query,
                                            "fields": keyword_fields_list,
                                            "type": "best_fields"
                                        }
                                    },
                                    "boost": keyword_weight
                                }
                            }
                        ]
                    }
                }
            }
            
            # Execute search
            self.logger.info(f"Executing hybrid search on index: {index_name}")
            response = self.es_client.search(index=index_name, body=es_query)
            
            # Format results
            results = []
            for hit in response["hits"]["hits"]:
                result = {
                    "score": hit["_score"],
                    "document": hit["_source"]
                }
                results.append(result)
            
            search_results = {
                "query": query,
                "search_type": "hybrid",
                "vector_weight": vector_weight,
                "keyword_weight": keyword_weight,
                "total_hits": response["hits"]["total"]["value"],
                "max_score": response["hits"]["max_score"],
                "results": results
            }
            
            self.logger.info(f"Found {len(results)} results for hybrid search: {query}")
            return json.dumps(search_results, indent=2)
            
        except Exception as e:
            error_msg = f"Error performing hybrid search: {str(e)}"
            self.logger.error(error_msg)
            return json.dumps({"error": error_msg})

    @kernel_function(
        description="Get information about available indices and their mappings in Elasticsearch.",
        name="get_index_info",
    )
    async def get_index_info(
        self,
        index_pattern: Annotated[str, "Index pattern to match (e.g., 'documents*' or specific index name)"] = "*"
    ) -> str:
        """
        Get information about Elasticsearch indices and their field mappings.
        
        Returns:
            JSON string containing index information and field mappings
        """
        try:
            # Get index information
            indices = self.es_client.indices.get(index=index_pattern)
            
            result = {
                "indices": {}
            }
            
            for index_name, index_info in indices.items():
                mappings = index_info.get("mappings", {})
                properties = mappings.get("properties", {})
                
                # Extract field information
                fields = {}
                for field_name, field_info in properties.items():
                    fields[field_name] = {
                        "type": field_info.get("type", "unknown"),
                        "properties": field_info.get("properties", {}) if field_info.get("type") == "object" else None
                    }
                
                result["indices"][index_name] = {
                    "fields": fields,
                    "field_count": len(fields)
                }
            
            self.logger.info(f"Retrieved information for {len(result['indices'])} indices")
            return json.dumps(result, indent=2)
            
        except Exception as e:
            error_msg = f"Error getting index information: {str(e)}"
            self.logger.error(error_msg)
            return json.dumps({"error": error_msg})