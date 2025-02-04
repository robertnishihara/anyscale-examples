from typing import Any, Dict, List
import ray
import datasets
import numpy as np
import os
from huggingface_hub import HfFileSystem, HfApi, DatasetCardData, errors


dataset_infos = {
    # Parquet
    "HuggingFaceFV/finevideo": ("data", "train", "parquet"),
    "HuggingFaceFW/fineweb-2": ("data", "train", "parquet"),
    "wikimedia/wikipedia": (None, "train", "parquet"),
    "nvidia/OpenMathInstruct-2": (None, "train", "parquet"),
    "HuggingFaceFW/fineweb": ("data", "train", "parquet"),
    "HuggingFaceTB/cosmopedia": ("data", "train", "parquet"),
    "BAAI/IndustryCorpus2": (None, "train", "parquet"),
    "HuggingFaceTB/smollm-corpus": (None, "train", "parquet"),
    "ylecun/mnist": (None, "train", "parquet"),
    "beans": (None, "train", "parquet"),
    "glue": (None, "train", "parquet"),
    "Zyphra/Zyda-2": ("data", "train", "parquet"),
    "Marqo/marqo-GS-10M": ("data", "in_domain", "parquet"),
    "HuggingFaceM4/OBELICS": ("data", "train", "parquet"),
    "rotten_tomatoes": (None, "train", "parquet"),

    # Json
    "allenai/c4": (None, "train", "json"),  # .json.gz
    "teknium/OpenHermes-2.5": (None, "train", "json"),  # .json
    "rombodawg/Everything_Instruct_Multilingual": (None, "train", "json"),  # .json
    "databricks/databricks-dolly-15k": (None, "train", "json"),  # .jsonl

    # Binary
    "mozilla-foundation/common_voice_17_0": ("audio", "train", "binary"),  # .tar
    "PolyAI/minds14": ("data", "train", "binary"),  # .zip
    "facebook/multilingual_librispeech": ("data", "train", "binary"),  # .tar.gz
}

dataset_name = os.environ.get("DATASET_NAME")
data_dir, split, file_format = dataset_infos[dataset_name]
print(f"dataset_name is {dataset_name}")
print(f"data_dir is {data_dir}")
print(f"split is {split}")
print(f"file_format is {file_format}")
token = os.environ["HF_TOKEN"]

# Limit scale (just for development to run the script faster)
FILE_LIMIT = None  # None for unlimited, e.g., 2 for a small value

ray.data.DataContext.get_current().retried_io_errors.append("429 Client Error: Too Many Requests for url")


def retrieve_hf_data_files(dataset_name, data_dir=None, token=None):
    # If you are unsure what the data_dir should be, run something like
    #     HfFileSystem(token=token).ls(base_path)
    base_path = f"hf://datasets/{dataset_name}/{data_dir or ''}".rstrip("/")
    patterns = datasets.data_files.get_data_patterns(base_path)
    data_files = datasets.data_files.DataFilesDict.from_patterns(
        patterns,
        base_path=base_path,
        allowed_extensions=datasets.load.ALL_ALLOWED_EXTENSIONS,
    )
    return data_files


data_files = retrieve_hf_data_files(dataset_name, data_dir=data_dir)[split]

if file_format == "parquet":
    ds = ray.data.read_parquet(
        data_files[:FILE_LIMIT],
        filesystem=HfFileSystem(token=token),
    )
elif file_format == "json":
    ds = ray.data.read_json(
        data_files[:FILE_LIMIT],
        filesystem=HfFileSystem(token=token),
    )   
elif file_format == "binary":
    ds = ray.data.read_binary_files(
        data_files[:FILE_LIMIT],
        filesystem=HfFileSystem(token=token),
    )      
else:
    raise Exception("Unsupported file format.")

ds = ds.materialize()

print(f"Success! Count is {ds.count()}.")




