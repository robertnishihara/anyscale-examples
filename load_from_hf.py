import fsspec.implementations.http
import datasets
from huggingface_hub import HfFileSystem, HfApi, DatasetCardData
import os
import requests
import ray


def validate_arguments(dataset_name, subset, split, headers):
    # Check if the dataset repository exists.
    response = requests.get(
        f"https://huggingface.co/api/datasets/{dataset_name}",
        headers=headers
    ).json()
    if "error" in response:
        raise Exception(f"Dataset repository doesn't exist. Error: {response}")

    # Check if the dataset supports parquet.
    response = requests.get(
        f"https://huggingface.co/api/datasets/{dataset_name}/parquet",
        headers=headers
    ).json()
    if "error" in response:
        raise Exception(f"Dataset repository doesn't support parquet. Error: {response}")
    if len(response) == 0:
        raise Exception(f"Dataset repository doesn't appear to support parquet. Error: {response}")

    # Check overall dataset generation.
    response = requests.get(
        f"https://huggingface.co/api/datasets/{dataset_name}/parquet/{subset}/{split}",
        headers=headers
    ).json()
    if "error" in response:
        raise Exception(f"Dataset generation failed: {response}")


def retrieve_data_files(dataset_name, revision=None, data_dir=None, data_files=None, token=None):
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
    return data_files


def load_from_hf(dataset_name, subset, split):
    token = os.environ["HF_TOKEN"]
    fs = HfFileSystem(token=token)
    headers = {
        "Authorization": f"Bearer {token}"
    }

    validate_arguments(dataset_name, subset, split, headers)

    file_urls = retrieve_urls(dataset_name, subset, split)

    print("FILE URLS")
    print(file_urls)

    ds = ray.data.read_parquet(
        file_urls,
        filesystem=fs,
    )

    return ds


def load_from_hf_old_version(dataset_name, subset, split):
    token = os.environ["HF_TOKEN"]
    headers = {
        "Authorization": f"Bearer {token}"
    }    

    validate_arguments(dataset_name, subset, split, headers)

    file_urls = requests.get(
        f"https://huggingface.co/api/datasets/{dataset_name}/parquet/{subset}/{split}",
        headers=headers
    ).json()

    fs = fsspec.implementations.http.HTTPFileSystem()

    ds = ray.data.read_parquet(
        file_urls,
        filesystem=fs,
    )
    return ds


token = os.environ["HF_TOKEN"]

dataset_name, data_dir, split = "beans", "default", "train"
dataset_name, data_dir, split = "glue", "mrpc", "train"
dataset_name, data_dir, split = "nvidia/HelpSteer2", "default", "train"
dataset_name, data_dir, split = "fka/awesome-chatgpt-prompts", "default", "train"
dataset_name, data_dir, split = "LLM360/TxT360", "default", "train"  # Doesn't support Parquet
dataset_name, data_dir, split = "KingNish/reasoning-base-20k", "default", "train"
dataset_name, data_dir, split = "Zyphra/Zyda-2", "default", "train"  # Fails with job manager crashed???
dataset_name, data_dir, split = "Marqo/marqo-GS-10M", "default", "in_domain"
dataset_name, data_dir, split = "BAAI/IndustryCorpus2", "default", "in_domain"  # Timeout?
dataset_name, data_dir, split = "mozilla-foundation/common_voice_17_0", "ab", "train"  # Permissions
dataset_name, data_dir, split = "rombodawg/Everything_Instruct_Multilingual", "default", "train"
dataset_name, data_dir, split = "facebook/multilingual_librispeech", "dutch", "train"
dataset_name, data_dir, split = "databricks/databricks-dolly-15k", "default", "train"
dataset_name, data_dir, split = "HuggingFaceTB/cosmopedia", "auto_math_text", "train"
dataset_name, data_dir, split = "HuggingFaceTB/smollm-corpus", "cosmopedia-v2", "train"  # aiohttp.client_exceptions.ClientResponseError: 400, message='Bad Request', url=URL('https://huggingface.co/api/datasets/HuggingFaceTB/smollm-corpus/parquet/cosmopedia-v2/train/20.parquet')
dataset_name, data_dir, split = "HuggingFaceFW/fineweb", "default", "train"  # Error: {'error': 'The computed response content exceeds the supported size in bytes (10000000).'}
dataset_name, data_dir, split = "ylecun/mnist", "default", "train"  # Exception: Dataset generation failed: {'error': 'Not found.'}
dataset_name, data_dir, split = "wikimedia/wikipedia", "20231101.ab", "train"
dataset_name, data_dir, split = "HuggingFaceM4/OBELICS", "default", "train"  # aiohttp.client_exceptions.ClientResponseError: 429, message='Too Many Requests', url=URL('https://huggingface.co/api/datasets/HuggingFaceM4/OBELICS/parquet/default/train/763.parquet')
dataset_name, data_dir, split = "nvidia/OpenMathInstruct-2", None, "train"
dataset_name, data_dir, split = "allenai/c4", "en", "train"
dataset_name, data_dir, split = "PolyAI/minds14", "en-US", "train"  # Doesn't support Parquet, weird directory structure
dataset_name, data_dir, split = "rotten_tomatoes", None, "train"
dataset_name, data_dir, split = "teknium/OpenHermes-2.5", None, "train"  # Some issue with reading json
dataset_name, data_dir, split = "HuggingFaceFV/finevideo", None, "train"

data_files = retrieve_data_files(dataset_name, revision=None, data_dir=data_dir, data_files=None, token=token)[split]
print(data_files)
print(len(data_files))

if data_files[0].endswith("parquet"):
    format = "parquet"
elif data_files[0].endswith("json"):
    format = "json"
else:
    raise Exception("Unsupported data format")


import time
start = time.time()

ds = ray.data.read_parquet(
    data_files[:500],
    # concurrency=100,
    filesystem=HfFileSystem(token=token),
)

print(ds.count())

print("elapsed: ", time.time() - start)
