# Elasticsearch Search Plugin Configuration Example
# Copy this file and update with your actual configuration values

# Elasticsearch Configuration
ELASTICSEARCH_CONFIG = {
    "host": "localhost",
    "port": 9200,
    "username": None,  # Set if using basic auth
    "password": None,  # Set if using basic auth
    "api_key": None,   # Set if using API key auth
    "use_ssl": False   # Set to True for HTTPS
}

# Azure OpenAI Configuration
AZURE_OPENAI_CONFIG = {
    "api_key": "your-azure-openai-api-key",
    "endpoint": "https://your-resource.openai.azure.com/",
    "deployment_name": "your-embedding-deployment",
    "embedding_model": "text-embedding-ada-002",
    "api_version": "2024-02-01"
}

# Search Configuration
SEARCH_CONFIG = {
    "default_index": "documents",
    "default_vector_field": "vector",
    "default_text_fields": "title,content,summary",
    "default_max_results": 5,
    "default_min_score": 0.0,
    "default_search_type": "semantic"  # Options: semantic, hybrid, index_info
}

# Hybrid Search Weights
HYBRID_SEARCH_CONFIG = {
    "vector_weight": 0.7,  # Weight for semantic similarity
    "keyword_weight": 0.3  # Weight for keyword matching
}

# Example usage in PromptFlow
PROMPTFLOW_EXAMPLE = {
    "tool_name": "semantic_search",
    "inputs": {
        "user_query": "${inputs.question}",
        "search_intent": "SearchDocuments", 
        "index_name": "documents",
        "deployment_name": "text-embedding-ada-002",
        "connection": "${inputs.azure_openai_connection}",
        "elasticsearch_host": "localhost",
        "elasticsearch_port": 9200,
        "text_fields": "title,content",
        "max_results": 5,
        "search_type": "semantic"
    }
}