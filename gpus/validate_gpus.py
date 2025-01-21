import argparse
import os
import ray
import torch
import torch.distributed as dist
import time


def parse_args():
    parser = argparse.ArgumentParser(description="GPU benchmark script")
    parser.add_argument("--num-nodes",
                        type=int, 
                        required=True,
                        help="Number of nodes in the Ray cluster")
    parser.add_argument("--num-gpus-per-machine",
                        type=int, 
                        required=True,
                        help="Number of GPUs per machine")
    parser.add_argument("--gpu-type",
                        type=str,
                        required=True,
                        help="Type of GPU (e.g., \"H100\", \"A100\")")
    return parser.parse_args()


@ray.remote(num_gpus=1)
def benchmark_single_gpu(gpu_type):
    results = {
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "node_ip": ray._private.services.get_node_ip_address(),
        "errors": []
    }

    try:
        device = torch.device("cuda:0")
        results.update({"gpu_name": torch.cuda.get_device_name(device)})
        torch.cuda.set_device(device)

        # Basic matrix multiplication benchmark
        size = 8192 * 3
        warmup_rounds = 3
        test_rounds = 25
        
        # Warmup
        for _ in range(warmup_rounds):
            x = torch.randn(size, size, device=device)
            y = torch.randn(size, size, device=device)
            z = torch.matmul(x, y)
            torch.cuda.synchronize()
        
        # Benchmark
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        for _ in range(test_rounds):
            x = torch.randn(size, size, device=device)
            y = torch.randn(size, size, device=device)
            z = torch.matmul(x, y)
            
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        # Calculate metrics
        average_time = (end - start) / test_rounds
        tflops = (2 * size * size * size) / (average_time * 1e12)  # Theoretical FLOPs for matmul
        
        # Memory info
        memory_allocated = torch.cuda.memory_allocated(device) / 1e9  # GB
        memory_cached = torch.cuda.memory_reserved(device) / 1e9  # GB
        total_memory = torch.cuda.get_device_properties(device).total_memory / 1e9  # GB
        
        results.update({
            "matmul_time": average_time,
            "matmul_tflops": tflops,
            "memory_allocated_gb": memory_allocated,
            "memory_cached_gb": memory_cached,
            "total_memory_gb": total_memory,
        })
        
        if gpu_type == "H100":
            if tflops < 45:
                results["errors"].append("Flops appear too low.")
            if total_memory < 80:
                results["errors"].append("Total memory appears too low.")
        else:
            print(f"Warning: flop and memory benchmarks not implemented for GPU type {gpu_type}.")

    except Exception as e:
        results["errors"].append(str(e))

    return results


def interpret_single_gpu_results(results):
    if all([len(result["errors"]) == 0 for result in results]):
        print("All single GPU tests passed.")
    else:
        for result in results:
            if len(result["errors"]) > 0:
                print(result)

    return {
        "min_tflops": min([result["matmul_tflops"] for result in results if len(result["errors"]) == 0]),
        "max_tflops": max([result["matmul_tflops"] for result in results if len(result["errors"]) == 0]),
        "min_total_memory_gb": min([result["total_memory_gb"] for result in results if len(result["errors"]) == 0]),
        "max_total_memory_gb": max([result["total_memory_gb"] for result in results if len(result["errors"]) == 0]),
    }


@ray.remote(num_gpus=1)
class GPUActor:
    def __init__(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        self.size = 1024 * 1024 * 32  # Size for bandwidth tests
        
        # Initialize process group
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        
        # Get the GPU ID from Ray
        self.gpu_id = ray.get_gpu_ids()[0]
        self.device = torch.device("cuda:0")  # Use cuda:0 since Ray sets CUDA_VISIBLE_DEVICES
        
        # Initialize process group
        dist.init_process_group(
            backend="nccl",
            world_size=world_size,
            rank=rank
        )
        
    def run_p2p_test(self):
        if self.world_size % 2 != 0:
            raise Exception("The P2P tests require an even number of GPUs per machine.")

        x = torch.randn(self.size, device=self.device)
        next_rank = (self.rank + 1) % self.world_size
        prev_rank = (self.rank - 1) % self.world_size

        for i in range(1, self.world_size)

        # Warmup
        for _ in range(5):
            if self.rank % 2 == 0:
                dist.send(x, next_rank)
                dist.recv(x, prev_rank)
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        iterations = 10
        
        for _ in range(iterations):
            if self.rank % 2 == 0:
                dist.send(x, next_rank)
                dist.recv(x, prev_rank)
        
        torch.cuda.synchronize()
        end = time.perf_counter()
        p2p_bandwidth = (self.size * 4 * iterations) / (end - start) / 1e9  # GB/s
        
        return {"p2p_bandwidth": p2p_bandwidth}
    
    def run_allreduce_test(self):
        x = torch.randn(self.size, device=self.device)
        
        # Warmup
        for _ in range(5):
            dist.all_reduce(x)
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        iterations = 10
        
        for _ in range(iterations):
            dist.all_reduce(x)
        
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        allreduce_bandwidth = (self.size * 4 * 2 * (self.world_size - 1) * iterations) / (end - start) / self.world_size / 1e9
        return {"allreduce_bandwidth": allreduce_bandwidth}
    
    def run_matmul_test(self):
        matrix_size = 8192
        x = torch.randn(matrix_size, matrix_size, device=self.device)
        y = torch.randn(matrix_size, matrix_size, device=self.device)
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        for _ in range(5):
            z = torch.matmul(x, y)
            dist.all_reduce(z)
        
        torch.cuda.synchronize()
        end = time.perf_counter()
        
        compute_time = (end - start) / 5
        tflops = (2 * matrix_size * matrix_size * matrix_size) / (compute_time * 1e12)
        return {"matmul_tflops": tflops}
    
    def run_all_tests(self):
        results = {
            "rank": self.rank,
            "gpu_id": self.gpu_id,
            "device_name": torch.cuda.get_device_name(0)
        }
        
        # Run all tests
        results.update(self.run_p2p_test())
        results.update(self.run_allreduce_test())
        results.update(self.run_matmul_test())
        
        return results
    
    def cleanup(self):
        dist.destroy_process_group()


if __name__ == "__main__":
    args = parse_args()
    num_nodes = args.num_nodes
    num_gpus_per_machine = args.num_gpus_per_machine
    gpu_type = args.gpu_type
    print(f"Using {num_nodes} nodes, each with {num_gpus_per_machine} GPUs of type {gpu_type}")

    #############################################
    # Test each individual GPU (flops, memory)
    #############################################

    results = ray.get([benchmark_single_gpu.remote(gpu_type) for _ in range(num_gpus)])
    single_gpu_min_max = interpret_single_gpu_results(results)
    print(single_gpu_min_max)
