import os
import json
import requests
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.core.prompts.prompts import SimpleInputPrompt
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding

# Function to query the Mixtral API
def send(query):
    try:
        url = "https://mixtral.k8s-gosha.atlas.illinois.edu/completion"

        myobj = {
            "prompt": f"<s>[INST]{query}[/INST]",
            "n_predict": -1  # -1 for no limit of tokens for output
        }

        headers = {
            "Content-Type": "application/json",
            # "Authorization": "Basic YXRsYXNhaXRlYW06anhAVTJXUzhCR1Nxd3U="
        }

        response = requests.post(url, data=json.dumps(myobj), headers=headers,
                                 auth=('atlasaiteam', 'jx@U2WS8BGSqwu'), timeout=1000)

        # Check HTTP status code for success or failure
        if response.status_code == 403:
            print("Access to Mixtral API forbidden. Check your credentials.")
            return None

        # Check if response is empty
        if not response.content:
            print("Empty response received from server.")
            return None

        # Try to parse JSON response
        try:
            json_response = response.json()
            return json_response
        except json.JSONDecodeError as json_err:
            print(f"Error decoding JSON response: {json_err}")
            print(f"Response content: {response.content}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None

# Load documents from the 'data' directory
data_dir = "./data"
documents = SimpleDirectoryReader(data_dir).load_data()

# Set up embeddings and service context
embed_model = LangchainEmbedding(
    HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
)

service_context = ServiceContext.from_defaults(
    chunk_size=256,
    llm=None,
    embed_model=embed_model,
)

# Create an index of the documents
index = VectorStoreIndex.from_documents(documents, service_context=service_context)

# Convert the index into a query engine
query_engine = index.as_query_engine()

# Define a function to query the index
def query_index(query):
    wrapped_query = f"<s>[INST]{query}[/INST]"  # Format query as per Mixtral requirements
    response = send(wrapped_query)
    return response

# Example queries
query1 = "Who is Dhruv?"
query2 = "Give me a summary of Dhruv's technical skills and background"

response1 = query_index(query1)
response2 = query_index(query2)

print("Response 1:")
print(response1)

print("\nResponse 2:")
print(response2)