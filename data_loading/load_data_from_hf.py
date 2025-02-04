from typing import Any, Dict, List
import ray
import datasets
import numpy as np
import os
from huggingface_hub import HfFileSystem, HfApi, DatasetCardData, errors


dataset_infos = {
    "HuggingFaceFV/finevideo": (None, "train", "parquet"),
    "HuggingFaceFW/fineweb-2": (None, "train", "parquet"),

    # BELOW ARE ALL UNTESTED

    "beans": ("default", "train", "parquet"),
    "glue": ("mrpc", "train", "parquet"),
    "nvidia/HelpSteer2": ("default", "train", "parquet"),
    "fka/awesome-chatgpt-prompts": ("default", "train", "parquet"),
    "LLM360/TxT360": ("default", "train", None),  # Doesn't support Parquet
    "KingNish/reasoning-base-20k": ("default", "train", "parquet"),
    "Zyphra/Zyda-2": ("default", "train", "parquet"),  # Fails with job manager crashed???
    "Marqo/marqo-GS-10M": ("default", "in_domain", "parquet"),
    "BAAI/IndustryCorpus2": ("default", "in_domain", "parquet"),  # Timeout?
    "mozilla-foundation/common_voice_17_0": ("ab", "train", "parquet"),  # Permissions
    "rombodawg/Everything_Instruct_Multilingual": ("default", "train", "parquet"),
    "facebook/multilingual_librispeech": ("dutch", "train", "parquet"),
    "databricks/databricks-dolly-15k": ("default", "train", "parquet"),
    "HuggingFaceTB/cosmopedia": ("auto_math_text", "train", "parquet"),
    "HuggingFaceTB/smollm-corpus": ("cosmopedia-v2", "train", "parquet"),  # aiohttp.client_exceptions.ClientResponseError: 400, message='Bad Request', url=URL('https://huggingface.co/api/datasets/HuggingFaceTB/smollm-corpus/parquet/cosmopedia-v2/train/20.parquet')
    "HuggingFaceFW/fineweb": ("default", "train", "parquet"),  # Error: {'error': 'The computed response content exceeds the supported size in bytes (10000000).'}
    "ylecun/mnist": ("default", "train", "parquet"),  # Exception: Dataset generation failed: {'error': 'Not found.'}
    "wikimedia/wikipedia": ("20231101.ab", "train", "parquet"),
    "HuggingFaceM4/OBELICS": ("default", "train", "parquet"),  # aiohttp.client_exceptions.ClientResponseError: 429, message='Too Many Requests', url=URL('https://huggingface.co/api/datasets/HuggingFaceM4/OBELICS/parquet/default/train/763.parquet')
    "nvidia/OpenMathInstruct-2": (None, "train", "parquet"),
    "allenai/c4": ("en", "train", "parquet"),
    "PolyAI/minds14": ("en-US", "train", "parquet"),  # Doesn't support Parquet, weird directory structure
    "rotten_tomatoes": (None, "train", "parquet"),
    "teknium/OpenHermes-2.5": (None, "train", "parquet"),  # Some issue with reading json
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

# ray.data.DataContext.get_current().retried_io_errors.append("429 Client Error: Too Many Requests for url")


def retrieve_hf_data_files(dataset_name, split=None, revision=None, data_dir=None, data_files=None, token=None):
    hfh_dataset_info = HfApi("https://huggingface.co").dataset_info(
        dataset_name,
        revision=revision,
        token=token,
        # timeout=100.0,
    )

    revision = hfh_dataset_info.sha  # fix the revision in case there are new commits in the meantime
    base_path = f"hf://datasets/{dataset_name}@{revision}/{data_dir or ''}".rstrip("/")

    dataset_card_data = DatasetCardData()
    metadata_configs = datasets.utils.metadata.MetadataConfigs.from_dataset_card_data(dataset_card_data)

    # we need a set of data files to find which dataset builder to use
    # because we need to infer module name by files extensions
    if data_files is not None:
        patterns = datasets.data_files.sanitize_patterns(data_files)
    elif metadata_configs and not data_dir and "data_files" in next(iter(metadata_configs.values())):
        patterns = datasets.data_files.sanitize_patterns(next(iter(metadata_configs.values()))["data_files"])
    else:
        patterns = datasets.data_files.get_data_patterns(base_path) #, download_config=self.download_config)
    data_files = datasets.data_files.DataFilesDict.from_patterns(
        patterns,
        base_path=base_path,
        allowed_extensions=datasets.load.ALL_ALLOWED_EXTENSIONS,
        # download_config=self.download_config,
    )
    return data_files[split]


data_files = retrieve_hf_data_files(dataset_name, data_dir=data_dir, split=split)


ds = ray.data.read_parquet(
    data_files[:FILE_LIMIT],
    filesystem=HfFileSystem(token=token),
)
ds = ds.materialize()

print(f"Success! Count is {ds.count()}.")




