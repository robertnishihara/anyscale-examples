import ray
import numpy as np
import os
import time
from typing import Dict, Any

# Configuration.
# Each image is about 1MB.
# So 1 billion images would be 1 PB.
NUM_IMAGES = 10**8
IMAGE_WIDTH = 580
IMAGE_HEIGHT = 580
CHANNELS = 3

# EXTENSIONS to add
# Read data from URLs instead of generating in Python.
# Use an actor map stage.
# Add GPU step.
# Instead of storing images, store metadata and output of inference.


def generate_synthetic_image(image_id: int, width: int = 580, height: int = 580, channels: int = 3) -> Dict[str, Any]:
    image_array = np.random.randint(0, 256, size=(height, width, channels), dtype=np.uint8)
    return {
        "image_id": image_id,
        "image_array": image_array,
        "metadata": {
            "dtype": str(image_array.dtype),
            "shape": image_array.shape,
            "generated_by": "ray_data_synthetic"
        }
    }


print(f"Ray Data Synthetic Image Generator")
print(f"==================================")
print(f"Number of images: {NUM_IMAGES:,}")
print(f"Image dimensions: {IMAGE_WIDTH}x{IMAGE_HEIGHT}x{CHANNELS}")
print()

image_ids = list(range(NUM_IMAGES))
output_path = os.path.join(os.environ["ANYSCALE_ARTIFACT_STORAGE"], "rkn/synthetic_image_output")

ds = ray.data.from_items(image_ids)
ds = ds.repartition(target_num_rows_per_block=1000)
ds = ds.map(lambda x: generate_synthetic_image(x["item"], IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS))
ds.write_parquet(output_path)

