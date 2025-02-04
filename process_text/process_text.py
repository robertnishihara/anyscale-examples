from typing import Any, Dict, List
import ray
import datasets
import numpy as np
import os
from huggingface_hub import HfFileSystem, HfApi, DatasetCardData, errors
import vllm


dataset_name = "HuggingFaceFW/fineweb-2"
data_dir = "data"
split = "train"
print(f"dataset_name is {dataset_name}")
print(f"data_dir is {data_dir}")
print(f"split is {split}")

# Limit scale (just for development to run the script faster)
FILE_LIMIT = None  # None for unlimited, e.g., 2 for a small value

ray.data.DataContext.get_current().retried_io_errors.append("429 Client Error: Too Many Requests for url")


# If you are unsure what the data_dir should be, run: HfFileSystem(token=token).ls(base_path)
base_path = f"hf://datasets/{dataset_name}/{data_dir or ''}".rstrip("/")
patterns = datasets.data_files.get_data_patterns(base_path)
data_files_with_splits = datasets.data_files.DataFilesDict.from_patterns(
    patterns,
    base_path=base_path,
    allowed_extensions=datasets.load.ALL_ALLOWED_EXTENSIONS,
)
data_files = data_files_with_splits[split]


tensor_parallel_size = 2


class VLLMPredictor:
    def __init__(self):
        # Create an LLM.
        self.llm = vllm.LLM(
            model="meta-llama/Llama-3.1-8B-Instruct",
            tensor_parallel_size=tensor_parallel_size
        )
        self.sampling_params = vllm.SamplingParams(temperature=0.8, top_p=0.95)

    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, list]:
        # Generate texts from the prompts.
        # The output is a list of RequestOutput objects that contain the prompt,
        # generated text, and other information.
        outputs = self.llm.generate(batch["text"], self.sampling_params)
        prompt: List[str] = []
        generated_text: List[str] = []
        for output in outputs:
            prompt.append(output.prompt)
            generated_text.append(' '.join([o.text for o in output.outputs]))
        return {
            "prompt": prompt,
            "generated_text": generated_text,
        }


# Run into issues with dtypes...
def add_prefix(row):
    prefix = "Please translate the following into English:\n--------------------------------------------\n\n\n\n\n "
    row["text"] = prefix + row["text"]
    return row


def add_prefix_to_batch(batch):
    prefix = "Please translate the following into English:\n--------------------------------------------\n\n\n\n\n "
    batch["text"] = prefix + batch["text"]
    return batch


def generate_remote_args_fn():
    # One bundle per tensor parallel worker
    pg = ray.util.placement_group(
        [{
            "GPU": 1,
            "CPU": 1,
            "accelerator_type:A10G": 0.0001
        }] * tensor_parallel_size,
        strategy="PACK",
    )
    return {
        "scheduling_strategy": ray.util.scheduling_strategies.PlacementGroupSchedulingStrategy(
            pg, placement_group_capture_child_tasks=True
        )
    }


ds = ray.data.read_parquet(
    data_files[:FILE_LIMIT],
    filesystem=HfFileSystem(),
)

ds = ds.map_batches(add_prefix_to_batch)

ds = ds.map_batches(
    VLLMPredictor,
    concurrency=(1, 64),
    batch_size=32,
    num_gpus=2,
    ray_remote_args_fn=generate_remote_args_fn,
)

ds = ds.materialize()

print(f"Success! Count is {ds.count()}.")