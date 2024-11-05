import ray
import datasets
import os
from huggingface_hub import HfFileSystem, HfApi, DatasetCardData
import requests
import tarfile
import io
import pydub

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


def extract_audio_objects_from_tar(tarfile_bytes):
    # This function does the following:
    # - Take a file path and retrieve the file (it's a .tar file)
    # - Untar the file (it contains a bunch of directories filled with .mp3 files and some other miscellaneous files)
    # - Read through the directories and return all of the .mp3 files along with the directory that the files came from
    audio_objects = []
    filenames = []

    counter = 0

    # Open the byte string as a tar file
    with tarfile.open(fileobj=io.BytesIO(tarfile_bytes)) as tar:
        # Iterate through each file in the tar archive
        for member in tar.getmembers():

            if counter == 10:
                break
            counter += 1

            # Check if the file is an MP3
            if member.isfile() and member.name.endswith(".mp3"):
                try:
                    # Extract the file in-memory and load it with pydub
                    file_data = tar.extractfile(member).read()
                    audio_obj = pydub.AudioSegment.from_file(io.BytesIO(file_data), format="mp3")
                    audio_objects.append(audio_obj)
                    filenames.append(member.name)
                except pydub.exceptions.CouldntDecodeError:
                    print(f"Ignoring: {member.name} (Could not decode)")
            else:
                print(f"Ignoring: {member.name} (Not mp3)")

    return [{"audio": audio, "filename": filename} for audio, filename in zip(audio_objects, filenames)]


def process_tarfile(row):
    return extract_audio_objects_from_tar(row["bytes"])


def set_sampling_rate(row):
    row["audio"] = row["audio"].set_frame_rate(16000)
    return row


def chunk_audio(row):
    audio = row["audio"]
    filename = row["filename"]
    chunk_duration_ms = 30 * 1000  # 30 seconds in milliseconds
    chunks = []

    # Split audio into chunks of 30 seconds or less.
    for i in range(0, len(audio), chunk_duration_ms):
        chunk = audio[i:i + chunk_duration_ms]
        chunks.append({
            "audio_chunk": chunk,
            "filename": f"{filename}_chunk_{i // chunk_duration_ms}"
        })
    
    return chunks


token = os.environ["HF_TOKEN"]
dataset_name, data_dir, split = "MLCommons/unsupervised_peoples_speech", None, "train"
data_files = retrieve_hf_data_files(dataset_name, data_dir=data_dir, split=split)

ds = ray.data.read_binary_files(
    data_files[:2],
    filesystem=HfFileSystem(token=token),
)

materialized_ds = ds.materialize()
# info = ds.take_all()
# raw = info[0]["bytes"]
# audio_objects = extract_audio_objects_from_tar(raw)

new_ds = materialized_ds.flat_map(process_tarfile)
new_ds = new_ds.map(set_sampling_rate)
new_ds = new_ds.materialize()

result_ds = new_ds.flat_map(chunk_audio)
result_ds = result_ds.materialize()