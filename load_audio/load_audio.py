import ray
import datasets
import os
from huggingface_hub import HfFileSystem, HfApi, DatasetCardData, errors


# Limit scale (just for development to run the script faster)
AUDIO_FILE_LIMIT = None  # None for unlimited, e.g., 2 for a small value

ray.data.DataContext.get_current().retried_io_errors.append("OSError 429 Client Error: Too Many Requests for url")
ray.data.DataContext.get_current().retried_io_errors.append("HfHubHTTPError: 429 Client Error: Too Many Requests for url")
ray.data.DataContext.get_current().retried_io_errors.append("HTTPError: 429 Client Error: Too Many Requests for url")


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


token = os.environ["HF_TOKEN"]
dataset_name, data_dir, split = "MLCommons/unsupervised_peoples_speech", None, "train"
data_files = retrieve_hf_data_files(dataset_name, data_dir=data_dir, split=split)

print(data_files)

ds = ray.data.read_binary_files(
    data_files[:AUDIO_FILE_LIMIT],
    filesystem=HfFileSystem(token=token),
    ray_remote_args={
        "memory": 8 * 10**9,  # Reserve roughly 8GB per reader.
        "retry_exceptions": [errors.HfHubHTTPError],
        "max_retries": -1,  # Retry infinitely.
    },
)

ds = ds.materialize()
print("COUNT", ds.count())
