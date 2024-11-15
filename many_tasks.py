import ray
import time


@ray.remote
def f():
    time.sleep(15)


ray.get([f.remote() for _ in range(1000000)])