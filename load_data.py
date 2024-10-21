import os
import requests
import ray
import datasets

ray.init(runtime_env={"py_modules": [datasets]})

ds_name = "nvidia/OpenMathInstruct-2"

token = os.environ["HF_TOKEN"]
headers = {
    "Authorization": f"Bearer {token}"
}

file_urls = requests.get(f"https://huggingface.co/api/datasets/nvidia/OpenMathInstruct-2/parquet/default/train", headers=headers).json()

import fsspec.implementations.http
http = fsspec.implementations.http.HTTPFileSystem()
# http = fsspec.implementations.http.HTTPFileSystem(
#     headers=headers, use_ssl=True
# )

from aiohttp.client_exceptions import ClientResponseError

ds = ray.data.read_parquet(
    file_urls,
    filesystem=http,
)

# hf_ds = load_dataset("nvidia/OpenMathInstruct-2", streaming=True)
# ds = ray.data.from_huggingface(hf_ds["train"])
# ds.count()

# ds = ray.data.read_parquet("s3://ray-benchmark-data/parquet/128MiB-file/10TiB", override_num_blocks=512)
# %time ds.count()