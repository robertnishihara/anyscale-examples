import os
import requests
import ray
import datasets

ray.init(runtime_env={"py_modules": [datasets]})

ds_name = "HuggingFaceFV/finevideo"

token = os.environ["HF_TOKEN"]
headers = {
    "Authorization": f"Bearer {token}"
}

file_urls = requests.get(
    f"https://huggingface.co/api/datasets/HuggingFaceFV/finevideo/parquet/default/train",
    headers=headers
).json()

import fsspec.implementations.http
http = fsspec.implementations.http.HTTPFileSystem(
    headers=headers, use_ssl=True
)
from aiohttp.client_exceptions import ClientResponseError

ds = ray.data.read_parquet(
    file_urls[:1],
    filesystem=http,
    # ray_remote_args={
    #     "retry_exceptions": [FileNotFoundError, ClientResponseError]
    # },
)




import ray
import datasets
import pandas as pd
import requests
import os
from io import BytesIO

from ray.data._internal.datasource.huggingface_datasource import HuggingFaceDatasource

ray.init(runtime_env={"py_modules": [datasets]})

# ds_name = "HuggingFaceFV/finevideo"
ds_name = "nvidia/OpenMathInstruct-2"

# dataset = datasets.load_dataset(ds_name, split="train")  # Temp for debugging

import fsspec

dataset = datasets.load_dataset(ds_name, split="train", streaming=True)

# file_urls = HuggingFaceDatasource.list_parquet_urls_from_dataset(dataset)

token = os.environ["HF_TOKEN"]
headers = {
    "Authorization": f"Bearer {token}"
}
API_URL = f"https://datasets-server.huggingface.co/parquet?dataset={ds_name}"
def query():
    response = requests.get(API_URL, headers=headers)
    return response.json()
file_urls = [file_info["url"] for file_info in query()["parquet_files"]]

import fsspec.implementations.http
http = fsspec.implementations.http.HTTPFileSystem()
from aiohttp.client_exceptions import ClientResponseError

ray.data.read_parquet(
    file_urls[:1],
    # parallelism=parallelism,
    filesystem=http,
    # concurrency=concurrency,
    # override_num_blocks=override_num_blocks,
    # ray_remote_args={
    #     "retry_exceptions": [FileNotFoundError, ClientResponseError]
    # },
)


# hf_dataset_stream = datasets.load_dataset("HuggingFaceFV/finevideo", streaming=True)
# parquet_url = "https://huggingface.co/api/datasets/HuggingFaceFV/finevideo/parquet/default/train"

url = "https://huggingface.co/api/datasets/HuggingFaceFV/finevideo/parquet/default/train/0.parquet"

urls = ray.data.from_items([
    {"url": url},
])



def load_parquet(url):
    with requests.get(url, headers=headers, stream=True) as response:
        response.raise_for_status()  # Raise an error for bad status codes
            
        # Initialize a BytesIO object to store the data
        parquet_bytes = BytesIO()
            
        # Write the content in chunks to avoid loading the entire file into memory at once
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:  # Filter out keep-alive chunks
                parquet_bytes.write(chunk)
    
        # Reset the pointer to the beginning of the BytesIO object
        parquet_bytes.seek(0)
    return pd.read_parquet(parquet_bytes)
    
df = urls.map(load_parquet)

df.count()