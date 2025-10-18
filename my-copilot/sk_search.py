
import os
import json
import asyncio
from typing import Optional
from promptflow import tool
from promptflow.connections import AzureOpenAIConnection

import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.functions import KernelArguments
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior

from plugins.es_search_plugin.ElasticsearchSearch import ElasticsearchSearch


def run_async_safely(coro):
    """
    Safely run async code in PromptFlow context
    """
    try:
        # Try to get the current loop
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No loop is running, we can use asyncio.run
        return asyncio.run(coro)
    
    # If we're here, there's already a loop running
    # We need to run in a new thread with its own loop
    import threading
    import concurrent.futures
    
    def run_in_thread():
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        try:
            return new_loop.run_until_complete(coro)
        finally:
            new_loop.close()
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(run_in_thread)
        return future.result()


@tool
def semantic_search(
    question: str,
    connection: AzureOpenAIConnection,
    elasticsearch_host: str = "localhost",
    elasticsearch_port: int = 9200,
    elasticsearch_username: Optional[str] = None,
    elasticsearch_password: Optional[str] = None,
    elasticsearch_api_key: Optional[str] = None,
    use_ssl: bool = False,
    index_name: str = "documents",
    vector_field: str = "vector",
    text_fields: str = "content",
    search_type: str = "semantic", 
    emmbedding: list = None
) -> str:
    """
    Perform semantic search using Elasticsearch and Azure OpenAI.
    
    Args:
        question: The user's question/query
        connection: Azure OpenAI connection from PromptFlow
        elasticsearch_host: Elasticsearch server host
        elasticsearch_port: Elasticsearch server port  
        elasticsearch_username: Username for basic auth (optional)
        elasticsearch_password: Password for basic auth (optional)
        elasticsearch_api_key: API key for authentication (optional)
        use_ssl: Whether to use SSL connection
        index_name: Name of the Elasticsearch index
        vector_field: Name of the vector field in documents
        text_fields: Comma-separated list of text fields to return
        search_type: Type of search ('semantic', 'hybrid', or 'index_info')
    
    Returns:
        JSON string with search results or response
    """
    
    async def run_search():
        try:
            
            print("Initializing Semantic Kernel and Elasticsearch Plugin...")
            
            # Initialize Semantic Kernel
            kernel = sk.Kernel()
            
            # Add Azure OpenAI chat completion service
            kernel.add_service(AzureChatCompletion(
                deployment_name="gpt-5-nano",
                endpoint=connection.api_base,
                api_key=connection.api_key,
                api_version=connection.api_version
            ))
            
            # Initialize and add Elasticsearch plugin
            es_plugin = ElasticsearchSearch(
                es_host=elasticsearch_host,
                es_port=elasticsearch_port,
                es_username=elasticsearch_username,
                es_password=elasticsearch_password,
                es_api_key=elasticsearch_api_key,
                use_ssl=use_ssl
            )
            
            print("Elasticsearch Plugin initialized.")
            print(f"sk_search: embedding size = {len(emmbedding) if emmbedding else 'N/A'}")
            
            # Add the plugin to the kernel
            kernel.add_plugin(es_plugin, plugin_name="elasticsearch")
            
            # Prepare arguments based on search type
            if search_type == "index_info":
                # Get index information
                function_name = "elasticsearch-get_index_info"
                arguments = KernelArguments(
                    index_pattern=index_name
                )
            elif search_type == "hybrid":
                # Perform hybrid search
                function_name = "elasticsearch-hybrid_search"
                arguments = KernelArguments(
                    query=question,
                    index_name=index_name,
                    vector_field=vector_field,
                    text_fields=text_fields,
                    openai_api_key=connection.api_key,
                    openai_endpoint=connection.api_base
                )
            else:
                # Default to semantic search
                function_name = "elasticsearch-semantic_search"
                arguments = KernelArguments(
                    query=question,
                    emmbedding=emmbedding if emmbedding and len(emmbedding) > 0 else None,
                    index_name=index_name,
                    vector_field=vector_field,
                    text_fields=text_fields,
                    openai_api_key=connection.api_key,
                    openai_endpoint=connection.api_base
                )
            
            print(f"Invoking function: {function_name}")
            
            # Execute the function directly
            # Split the function_name to get plugin and function names
            plugin_name, func_name = function_name.split("-", 1)
            function = kernel.get_function(plugin_name, func_name)
            result = await function.invoke(kernel, arguments)
            print("Search completed successfully.")
            print(f"Search results: {result}")
            return str(result)
            
        except Exception as e:
            error_response = {
                "error": f"Failed to perform search: {str(e)}",
                "question": question,
                "search_type": search_type
            }
            return json.dumps(error_response, indent=2)
    
    # Run the async function safely
    try:
        return run_async_safely(run_search())
    except Exception as e:
        error_response = {
            "error": f"Failed to execute search: {str(e)}",
            "question": question
        }
        return json.dumps(error_response, indent=2)
