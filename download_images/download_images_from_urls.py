from typing import Dict
import ray
import datasets
import os
from huggingface_hub import HfFileSystem, HfApi, DatasetCardData, errors
import requests
import io


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


def download_images(batch):
    success = []
    images = []
    for url in batch["url"]:
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
        except Exception as e:
            print(f"Failed to download {url}, issue was {e}.")
            success.append(False)
            images.append(b"")
            continue

        if "image" not in response.headers.get("Content-Type", ""):
            print(f"No image in response for {url}.")
            images.append(b"")
            success.append(False)
            continue

        images.append(response.content)
        success.append(True)

    batch["images"] = images
    batch["success"] = success
    return batch


# token = os.environ["HF_TOKEN"]
dataset_name, data_dir, split = "dalle-mini/open-images", None, "train"
data_files = retrieve_hf_data_files(dataset_name, data_dir=data_dir, split=split)
data_files = [f for f in data_files if f.endswith("parquet")]  # Just parquet files.

ds = ray.data.read_parquet(
    data_files,
    filesystem=HfFileSystem(), # (token=token),
)
ds = ds.map_batches(download_images)
ds = ds.filter(lambda row: row["success"])
ds.write_parquet(os.path.join(os.environ["ANYSCALE_ARTIFACT_STORAGE"], "open-images-downloaded"))
