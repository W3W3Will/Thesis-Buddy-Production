import logging

import dashscope
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# DashScope API setup
dashscope.base_http_api_url = "https://dashscope-intl.aliyuncs.com/api/v1"
dashscope.api_key = "sk-447f61df00ca4dcd84feeed099b3e630"

# Initialize the Elasticsearch client
ELASTICSEARCH_HOST = (
    "http://es-sg-ju33w74pb0002nt56.public.elasticsearch.aliyuncs.com:9200"
)
ELASTICSEARCH_USERNAME = "elastic"
ELASTICSEARCH_PASSWORD = "100%Winrate"

es = Elasticsearch(
    [ELASTICSEARCH_HOST], basic_auth=(ELASTICSEARCH_USERNAME, ELASTICSEARCH_PASSWORD)
)

# Initialize the model
model_name = "sentence-transformers/msmarco-MiniLM-L-12-v3"
logger.info(f"Loading model {model_name}...")
model = SentenceTransformer(model_name)
logger.info("Model loaded.")


def retrieve(query, size=5, num_candidates=10):
    # Generate embedding for the query text
    logger.info(f"Generating embedding for query: '{query}'")
    embedding = model.encode(query).tolist()

    # Perform k-NN search in Elasticsearch using the embedding
    logger.info("Performing k-NN search in Elasticsearch...")
    response = es.search(
        index="pdf-files",
        body={
            "_source": ["filename", "content"],
            "size": size,  # Number of results to return
            "query": {
                "knn": {
                    "field": "embedding",
                    "query_vector": embedding,
                    "num_candidates": num_candidates,
                }
            },
        },
    )

    # Extract relevant information from the search results
    hits = response.get("hits", {}).get("hits", [])
    results = [
        {
            "filename": hit["_source"]["filename"],
            "content_snippet": hit["_source"]["content"][
                :200
            ],  # Show only the first 200 characters
            "full_content": hit["_source"]["content"],  # Include the full content
            "_score": hit["_score"],
        }
        for hit in hits
    ]

    return results


if __name__ == "__main__":
    # Test the retrieval functionality
    query = "Based on data you have, how does stemming compare to lemmatization?"
    results = retrieve(query)

    # Print the results
    if results:
        for idx, result in enumerate(results):
            print(f"Result {idx + 1}:")
            print(f"Filename: {result['filename']}")
            print(f"Content Snippet: {result['content_snippet']}")
            print(
                f"Full Content:\n{result['full_content'][:1000]}"
            )  # Print first 1000 characters of full content
            print(f"Score: {result['_score']}\n")
    else:
        print("No results found.")
