# import logging

# import dashscope
# from elasticsearch import Elasticsearch
# from flask import Flask
# from flask_cors import CORS

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# app = Flask(__name__)
# CORS(app)

# dashscope.base_http_api_url = "https://dashscope-intl.aliyuncs.com/api/v1"
# dashscope.api_key = "sk-447f61df00ca4dcd84feeed099b3e630"

# ELASTICSEARCH_HOST = (
#     "http://es-sg-ju33w74pb0002nt56.public.elasticsearch.aliyuncs.com:9200"
# )
# ELASTICSEARCH_USERNAME = "elastic"
# ELASTICSEARCH_PASSWORD = "100%Winrate"

# es = Elasticsearch(
#     [ELASTICSEARCH_HOST], basic_auth=(ELASTICSEARCH_USERNAME, ELASTICSEARCH_PASSWORD)
# )

# # Define the search query with a standard retriever
# search_query = {
#     "retriever": {
#         "standard": {
#             "query": {
#                 "bool": {
#                     "should": [{"match": {"region": "Austria"}}],
#                     "filter": [{"term": {"year": "2019"}}],
#                 }
#             }
#         }
#     }
# }

# # Execute the search
# response = es.search(index="restaurants", body=search_query)

# # Print the search results
# for hit in response["hits"]["hits"]:
#     print(hit["_source"])

# import time

# from sentence_transformers import SentenceTransformer

# # Measure the time to load the model
# start_time = time.time()

# # Load the model
# model = SentenceTransformer("sentence-transformers/msmarco-MiniLM-L-12-v3")

# load_time = time.time() - start_time
# print(f"Model loading time: {load_time:.2f} seconds")

# # Prepare the sentence
# sentence = "This is an example sentence."

# # Run the embedding three times and measure the time for each
# for i in range(3):
#     start_time = time.time()

#     # Compute the embedding
#     embedding = model.encode(sentence)

#     embedding_time = time.time() - start_time
#     print(f"Embedding {i+1} time: {embedding_time:.2f} seconds")

# import time

# import fitz  # PyMuPDF
# from sentence_transformers import SentenceTransformer

# # Step 1: Load the model and measure the time
# start_time = time.time()
# model = SentenceTransformer("sentence-transformers/msmarco-MiniLM-L-12-v3")
# load_time = time.time() - start_time
# print(f"Model loading time: {load_time:.2f} seconds")

# # Step 2: Upload the PDF file
# pdf_file_path = "_Procedia__Skripsi_Rio_Marcel_Habel___ICCSCI(1).pdf"  # Replace with your actual PDF file path

# # Measure the time to read and extract text from the PDF
# start_time = time.time()

# # Step 3: Extract text from the PDF
# doc = fitz.open(pdf_file_path)
# text = ""
# for page in doc:
#     text += page.get_text()

# text_extraction_time = time.time() - start_time
# print(f"Text extraction time: {text_extraction_time:.2f} seconds")

# import sys

# # Ensure the output encoding is UTF-8
# sys.stdout.reconfigure(encoding="utf-8")

# # print("Full extracted text from PDF:\n")
# # print(text)

# # Step 4: Embed the extracted text and measure the time
# start_time = time.time()

# # Split text into sentences if needed (optional step depending on your use case)
# # sentences = text.split('.')

# # Compute embeddings (you can adjust this depending on how you want to handle the text)
# embedding = model.encode(text)

# embedding_time = time.time() - start_time
# print(f"Embedding time: {embedding_time:.2f} seconds")

# # Print the total time for the entire process
# total_time = load_time + text_extraction_time + embedding_time
# print(f"Total process time: {total_time:.2f} seconds")

# # Optionally print the embedding (for verification)
# print("Embedding:", embedding)

import logging

import dashscope
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
            print(f"Score: {result['_score']}\n")
    else:
        print("No results found.")
