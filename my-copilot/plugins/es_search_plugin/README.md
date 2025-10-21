# Elasticsearch Search Plugin

This plugin provides semantic search capabilities for documents stored in Elasticsearch using Azure OpenAI embeddings and Semantic Kernel.

## Features

- **Semantic Search**: Find documents using vector similarity search with Azure OpenAI embeddings
- **Hybrid Search**: Combine semantic search with traditional keyword search for better results
- **Index Information**: Get information about available Elasticsearch indices and their field mappings
- **Flexible Configuration**: Support for various Elasticsearch authentication methods

## Setup

### Prerequisites

1. **Elasticsearch**: Running Elasticsearch instance (local or cloud)
2. **Azure OpenAI**: Azure OpenAI service with embedding model deployment
3. **Python Dependencies**: Install required packages from `requirements.txt`

### Installation

```bash
pip install -r requirements.txt
```

### Configuration

The plugin requires the following configuration parameters:

#### Elasticsearch Configuration
- `elasticsearch_host`: Elasticsearch host (default: "localhost")
- `elasticsearch_port`: Elasticsearch port (default: 9200)
- `elasticsearch_username`: Username for basic auth (optional)
- `elasticsearch_password`: Password for basic auth (optional)
- `elasticsearch_api_key`: API key for authentication (optional)
- `use_ssl`: Use SSL for connection (default: False)

#### Azure OpenAI Configuration
- `connection`: Azure OpenAI connection object (from PromptFlow)
- `deployment_name`: Name of the Azure OpenAI deployment
- `embedding_model`: Embedding model name (default: "text-embedding-ada-002")

#### Search Configuration
- `index_name`: Name of the Elasticsearch index to search
- `vector_field`: Name of the field containing document vectors (default: "vector")
- `text_fields`: Comma-separated list of text fields to return (default: "content")
- `max_results`: Maximum number of results (default: 5)
- `min_score`: Minimum similarity score threshold (default: 0.0)

## Usage

### Basic Semantic Search

```python
from plugins.es_search_plugin import semantic_search

# Configure your connection and parameters
result = semantic_search(
    user_query="artificial intelligence in healthcare",
    search_intent="SearchDocuments",
    index_name="documents",
    deployment_name="your-embedding-deployment",
    connection=azure_openai_connection,
    elasticsearch_host="your-es-host",
    text_fields="title,content,summary"
)
```

### Hybrid Search

```python
result = semantic_search(
    user_query="machine learning algorithms",
    search_intent="SearchDocuments",
    index_name="research_papers",
    deployment_name="your-embedding-deployment",
    connection=azure_openai_connection,
    search_type="hybrid",
    text_fields="title,abstract,content"
)
```

### Get Index Information

```python
result = semantic_search(
    user_query="",
    search_intent="GetIndexInfo",
    index_name="documents",
    deployment_name="your-embedding-deployment",
    connection=azure_openai_connection
)
```

## Elasticsearch Document Structure

For optimal performance, your Elasticsearch documents should have the following structure:

```json
{
  "title": "Document Title",
  "content": "Document content text...",
  "summary": "Brief summary...",
  "vector": [0.1, 0.2, 0.3, ...],  // Embedding vector
  "metadata": {
    "author": "Author Name",
    "date": "2024-01-01",
    "category": "Category"
  }
}
```

### Index Mapping Example

```json
{
  "mappings": {
    "properties": {
      "title": {"type": "text"},
      "content": {"type": "text"},
      "summary": {"type": "text"},
      "vector": {
        "type": "dense_vector",
        "dims": 1536,
        "index": true,
        "similarity": "cosine"
      },
      "metadata": {
        "properties": {
          "author": {"type": "keyword"},
          "date": {"type": "date"},
          "category": {"type": "keyword"}
        }
      }
    }
  }
}
```

## Integration with PromptFlow

This plugin is designed to work with PromptFlow and can be integrated into chat flows for semantic document search capabilities.

### Example Flow Configuration

1. Create a PromptFlow that uses the `semantic_search` tool
2. Configure Azure OpenAI connection
3. Set up Elasticsearch parameters
4. Use intent classification to determine when to perform searches

## Troubleshooting

### Common Issues

1. **Connection Errors**: Verify Elasticsearch host, port, and authentication credentials
2. **Embedding Errors**: Check Azure OpenAI API key, endpoint, and model deployment
3. **Index Not Found**: Ensure the specified index exists in Elasticsearch
4. **Vector Field Missing**: Verify that documents have the expected vector field

### Logging

The plugin includes comprehensive logging. Set the logging level to DEBUG to see detailed operation logs:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## License

This plugin is part of the orchestrated semantic search project.