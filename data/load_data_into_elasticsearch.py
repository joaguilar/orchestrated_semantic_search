import os
import csv
from typing import List, Dict

from elasticsearch import Elasticsearch, helpers
from openai import AzureOpenAI
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()
CSV_PATH = os.environ.get("CSV_PATH", "documents.csv")
ES_URL = os.environ.get("ES_URL", "http://localhost:9200")
ES_INDEX = os.environ.get("ES_INDEX", "docs-1")

# --- Azure OpenAI config ---
# Required env vars:
#   AZURE_OPENAI_API_KEY
#   AZURE_OPENAI_ENDPOINT           e.g. https://my-aoai.openai.azure.com/
# Optional (has default below):
#   AZURE_OPENAI_API_VERSION        e.g. 2024-05-01-preview
#   AZURE_OPENAI_EMBEDDING_DEPLOYMENT (your deployment name in Azure; must map to text-embedding-small-3)
AOAI_API_KEY = os.environ["AZURE_OPENAI_API_KEY"]
AOAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
AOAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-05-01-preview")
EMBEDDING_DEPLOYMENT = os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-small-3")

client = AzureOpenAI(
    api_key=AOAI_API_KEY,
    api_version=AOAI_API_VERSION,
    azure_endpoint=AOAI_ENDPOINT,
)

def embed(texts: List[str]) -> List[List[float]]:
    """
    Creates embeddings for a list of strings using the specified Azure OpenAI deployment.
    """
    # Azure supports batching; for simplicity we send the whole list if small.
    # For large corpora, batch to ~64–512 inputs depending on your rate limits.
    resp = client.embeddings.create(
        model=EMBEDDING_DEPLOYMENT,  # deployment name in Azure
        input=texts
    )
    return [d.embedding for d in resp.data]

def load_rows(csv_path: str) -> List[Dict[str, str]]:
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            title = (row.get("title") or "").strip()
            content = (row.get("content") or "").strip()
            if title and content:
                rows.append({"title": title, "content": content})
    return rows

def main():
    # Connect to Elasticsearch
    es = Elasticsearch(ES_URL)

    # Load CSV
    docs = load_rows(CSV_PATH)
    if not docs:
        raise RuntimeError(f"No valid rows found in {CSV_PATH}")

    # Create embeddings in batches
    BATCH = 64
    actions = []
    for i in range(0, len(docs), BATCH):
        batch = docs[i:i+BATCH]
        embeddings = embed([d["content"] for d in batch])

        for doc, vec in zip(batch, embeddings):
            actions.append({
                "_index": ES_INDEX,
                "_source": {
                    "title": doc["title"],
                    "content": doc["content"],
                    "vector": vec
                }
            })

    # Bulk index
    success, err = helpers.bulk(es, actions, raise_on_error=False)
    print(f"Indexed OK: {success}")
    if err:
        print(f"Errors: {len(err)}")
        # Print a couple of examples for debugging
        for e in err[:3]:
            print(e)

    # Quick sanity check: count
    count = es.count(index=ES_INDEX)["count"]
    print(f"Total docs now in {ES_INDEX}: {count}")

if __name__ == "__main__":
    main()
