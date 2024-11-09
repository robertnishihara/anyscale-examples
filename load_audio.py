from typing import Dict
import ray
import datasets
import os
from huggingface_hub import HfFileSystem, HfApi, DatasetCardData, errors
import requests
import tarfile
import io
import pydub
import numpy as np
import torch
import whisper

SAMPLING_RATE = 16000
CHUNK_SIZE_SECONDS = 30
NUM_CHANNELS = 1

# Limit scale (just for development to run the script faster)
AUDIO_FILE_LIMIT = None  # None for unlimited, e.g., 2 for a small value
MP3S_PER_TAR_LIMIT = None  # None for unlimited, e.g., 10 for a small value

ray.data.DataContext.get_current().retried_io_errors.append("OSError 429 Client Error: Too Many Requests for url")
ray.data.DataContext.get_current().retried_io_errors.append("HfHubHTTPError: 429 Client Error: Too Many Requests for url")


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
                    filenames.append(member.name)
                except pydub.exceptions.CouldntDecodeError:
                    print(f"Ignoring: {member.name} (Could not decode)")
            else:
                print(f"Ignoring: {member.name} (Not mp3)")

            if MP3S_PER_TAR_LIMIT and len(audio_objects) == MP3S_PER_TAR_LIMIT:
                break

    return [{"audio": audio, "filename": filename} for audio, filename in zip(audio_objects, filenames)]


def process_tarfile(row):
    return extract_audio_objects_from_tar(row["bytes"])


def set_sampling_rate_and_channels(row):
    row["audio"] = row["audio"].set_frame_rate(SAMPLING_RATE).set_channels(NUM_CHANNELS)
    return row


def chunk_audio(row):
    audio = row["audio"]
    filename = row["filename"]
    chunk_duration_ms = CHUNK_SIZE_SECONDS * 1000  # CHUNK_SIZE_SECONDS in milliseconds
    chunks = []

    # Split audio into chunks of CHUNK_SIZE_SECONDS seconds or less.
    for i in range(0, len(audio), chunk_duration_ms):
        chunk = audio[i:i + chunk_duration_ms]
        chunks.append({
            "audio_chunk": chunk,
            "filename": f"{filename}_chunk_{i // chunk_duration_ms}"
        })
    
    return chunks


def audio_to_numpy(row):
    # Do we need the audio_chunk anymore or can we drop that here?
    audio_chunk = row["audio_chunk"]
    sample = np.array(audio_chunk.get_array_of_samples()).astype(np.float32)
    max_value = float(2 ** (audio_chunk.sample_width * 8 - 1))
    sample = sample / max_value

    # Pad the numpy array so they all have the same length. This might be a bad idea.
    target_length = CHUNK_SIZE_SECONDS * SAMPLING_RATE * NUM_CHANNELS
    if len(sample) < target_length:
        sample = np.pad(sample, (0, target_length - len(sample)), mode="constant", constant_values=0)

    row["numpy"] = sample
    return row


class WhisperPredictor:
    def __init__(self):
        self.model = whisper.load_model("base").to("cuda")  # Can be "small", "medium", etc.

    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        inputs = torch.as_tensor(
            batch["numpy"],
            dtype=torch.float32,
            device="cuda"
        )
        audio_tensor = torch.tensor(inputs)

        with torch.no_grad():
            mel = whisper.log_mel_spectrogram(audio_tensor)  # Get the mel spectrogram.
            embeddings = self.model.encoder(mel).cpu().numpy()  # Compute embeddings.
            batch["whisper_embeddings"] = embeddings

        return batch


token = os.environ["HF_TOKEN"]
dataset_name, data_dir, split = "MLCommons/unsupervised_peoples_speech", None, "train"
data_files = retrieve_hf_data_files(dataset_name, data_dir=data_dir, split=split)


ds = ray.data.read_binary_files(
    data_files[:AUDIO_FILE_LIMIT],
    filesystem=HfFileSystem(token=token),
    ray_remote_args={
        "memory": 8 * 10**9,  # Reserve roughly 8GB per reader.
        "retry_exceptions": [errors.HfHubHTTPError],
        "max_retries": -1,  # Retry infinitely.
    },
)


ds = ds.flat_map(process_tarfile)
ds = ds.map(set_sampling_rate_and_channels)

ds = ds.flat_map(chunk_audio)
ds = ds.map(audio_to_numpy)

ds = ds.map_batches(
    WhisperPredictor,
    concurrency=4,
    batch_size=16,
    num_gpus=1
)

print("COUNT", ds.count())

# TESTING EMBEDDINGS
"""
name = 'betv-16557frankmooreup-e576-collection24/betv-16557frankmooreup-e576-collection24.mp3_chunk_150'
row = ds_with_embeddings.filter(lambda row: row["filename"] == name).take(1)[0]
e = row["whisper_embeddings"]
def dot_product(row):
    row["dot_product"] = np.sum(e * row["whisper_embeddings"]) / (np.linalg.norm(row["whisper_embeddings"]) * np.linalg.norm(e))
    return row
dot_ds = ds_with_embeddings.map(dot_product)
max_dot_product = dot_ds.filter(lambda row: row["filename"] != name).max("dot_product")
max_dot_product_row = dot_ds.filter(lambda row: row["dot_product"] == max_dot_product).take(1)[0]
max_dot_product_row["filename"]  # Should sound similar to `name`
"""