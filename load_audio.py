import ray
import datasets
import os
from huggingface_hub import HfFileSystem, HfApi, DatasetCardData
import requests


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

ds = ray.data.read_binary_files(
    data_files[:1],
    filesystem=HfFileSystem(token=token),
)

def parse_filename(row: Dict[str, Any]) -> Dict[str, Any]:
    row["filename"] = os.path.basename(row["path"])
    return row

def file_path_to_audio_files(file_path):
    # This function does the following:
    # - Take a file path and retrieve the file (it's a .tar file)
    # - Untar the file (it contains a bunch of directories filled with .mp3 files and some other miscellaneous files)
    # - Read through the directories and return all of the .mp3 files along with the directory that the files came from
    result = requests.get(file_path, headers=headers)


materialized_ds = ds.materialize()

print(ds.count())

######
info = ds.take_all()
raw = info[0]["bytes"]

import tarfile
import io
import pydub


def extract_audio_objects_from_tar(tarfile_bytes):
    audio_objects = []

    # Open the byte string as a tar file
    with tarfile.open(fileobj=io.BytesIO(tarfile_bytes)) as tar:
        # Iterate through each file in the tar archive
        for member in tar.getmembers():
            # Check if the file is an MP3
            if member.isfile() and member.name.endswith(".mp3"):
                try:
                    # Extract the file in-memory and load it with pydub
                    file_data = tar.extractfile(member).read()
                    audio_obj = pydub.AudioSegment.from_file(io.BytesIO(file_data), format="mp3")
                    audio_objects.append(audio_obj)
                except pydub.exceptions.CouldntDecodeError:
                    print(f"Ignoring: {member.name} (Could not decode)")
            else:
                print(f"Ignoring: {member.name} (Not mp3)")

    return audio_objects

audio_objects = extract_audio_objects_from_tar(raw)