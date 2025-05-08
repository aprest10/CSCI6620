import os
import time
import glob
import numpy as np
import pandas as pd
import torch
import scipy.io as sio
import matplotlib.pyplot as plt
import psutil
import gc
import seaborn as sns
import pynvml
from contextlib import contextmanager
import warnings

warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state")

class SparseMatrixBenchmark:
    def __init__(self, folder_path=None, device="cuda", block_sizes=[(2,2)], num_trials=10):
        self.device = device
        self.block_sizes = block_sizes
        self.num_trials = num_trials
        self.results = []
        self.results_df = None
        self.nvml_initialized = False
        self.nvml_handle = None
        
        if folder_path is not None:
            self.results_df = self.process_mtx_folder(folder_path)

    @contextmanager
    def nvml_context(self):
        try:
            pynvml.nvmlInit()
            self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.nvml_initialized = True
            yield
        except Exception as e:
            print(f"NVML Error: {e}")
            yield
        finally:
            if self.nvml_initialized:
                pynvml.nvmlShutdown()

    @staticmethod
    def get_gpu_memory_usage():
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            return torch.cuda.memory_allocated() / 1024**2
        return 0

    def load_mtx_file(self, file_path):
        return sio.mmread(file_path)
        
    def convert_to_pytorch_sparse_formats(self, scipy_sparse):
        coo = scipy_sparse.tocoo()
        
        indices = torch.tensor(np.vstack((coo.row, coo.col)), dtype=torch.long)
        values = torch.tensor(coo.data, dtype=torch.float32)
        shape = torch.Size(coo.shape)
        
        formats = {}
        conversion_times = {}
        memory_usage = {}
        
        if self.device == "cuda":
            torch.cuda.empty_cache()
            base_memory = self.get_gpu_memory_usage()
        
        # COO
        start_time = time.time()
        formats["coo"] = torch.sparse_coo_tensor(indices, values, shape, device=self.device)
        conversion_times["coo"] = time.time() - start_time
        memory_usage["coo"] = self.get_gpu_memory_usage() - base_memory if self.device == "cuda" else 0
        
        # CSR
        if self.device == "cuda":
            torch.cuda.empty_cache()
            base_memory = self.get_gpu_memory_usage()
        
        scipy_csr = scipy_sparse.tocsr()
        start_time = time.time()
        crow_indices = torch.tensor(scipy_csr.indptr, dtype=torch.int64)
        col_indices = torch.tensor(scipy_csr.indices, dtype=torch.int64)
        values = torch.tensor(scipy_csr.data, dtype=torch.float32)
        formats["csr"] = torch.sparse_csr_tensor(crow_indices, col_indices, values, shape, device=self.device)
        conversion_times["csr"] = time.time() - start_time
        memory_usage["csr"] = self.get_gpu_memory_usage() - base_memory if self.device == "cuda" else 0
        
        # CSC 
        if self.device == "cuda":
            torch.cuda.empty_cache()
            base_memory = self.get_gpu_memory_usage()
        
        scipy_csc = scipy_sparse.tocsc()
        start_time = time.time()
        ccol_indices = torch.tensor(scipy_csc.indptr, dtype=torch.int64)
        row_indices = torch.tensor(scipy_csc.indices, dtype=torch.int64)
        values = torch.tensor(scipy_csc.data, dtype=torch.float32)
        formats["csc"] = torch.sparse_csc_tensor(ccol_indices, row_indices, values, shape, device=self.device)
        conversion_times["csc"] = time.time() - start_time
        memory_usage["csc"] = self.get_gpu_memory_usage() - base_memory if self.device == "cuda" else 0
        
        # BSR
        for block_size in self.block_sizes:
            block_h, block_w = block_size
            if shape[0] % block_h == 0 and shape[1] % block_w == 0:
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                    base_memory = self.get_gpu_memory_usage()
                try:
                    scipy_bsr = scipy_sparse.T.tobsr(blocksize=(block_w, block_h)).T
                    start_time = time.time()
                    ccol_indices = torch.tensor(scipy_bsr.indptr, dtype=torch.int64)
                    row_indices = torch.tensor(scipy_bsr.indices, dtype=torch.int64)
                    values = torch.tensor(scipy_bsr.data, dtype=torch.float32).reshape(-1, block_h, block_w)
                    formats[f"bsr_{block_h}x{block_w}"] = torch.sparse_bsr_tensor(
                        ccol_indices, row_indices, values, 
                        torch.Size([shape[0], shape[1]]),
                        device=self.device
                    )
                    conversion_times[f"bsr_{block_h}x{block_w}"] = time.time() - start_time
                    memory_usage[f"bsr_{block_h}x{block_w}"] = self.get_gpu_memory_usage() - base_memory if self.device == "cuda" else 0
                except Exception as e:
                    print(f"bsr conversion error with blocksize {block_size}: {e}")
                    formats[f"bsr_{block_h}x{block_w}"] = None
                    conversion_times[f"bsr_{block_h}x{block_w}"] = np.nan
                    memory_usage[f"bsr_{block_h}x{block_w}"] = np.nan
            else:
                print(f"Matrix shape {shape} not compatible with block size {block_size}")
                fmt = f"bsr_{block_h}x{block_w}"
                formats[fmt] = None
                conversion_times[fmt] = np.nan
                memory_usage[fmt] = np.nan
        
        return formats, conversion_times, memory_usage
        
    def measure_matmul_performance(self, sparse_formats, shape):
        matmul_times = {}
        gpu_utilization = {}
        
        with self.nvml_context():
            for format_name, sparse_tensor in sparse_formats.items():
                if sparse_tensor is None:
                    matmul_times[format_name] = np.nan
                    gpu_utilization[format_name] = np.nan
                    continue

                x = torch.randn(shape[1], 1, device=self.device)

                if format_name.startswith(("coo", "csr", "csc")):
                    _ = torch.matmul(sparse_tensor, x)
                elif format_name.startswith("bsr"):
                    _ = sparse_tensor @ x
                    
                torch.cuda.synchronize() if self.device == "cuda" else None
                
                times = []
                util_samples = []
                for _ in range(self.num_trials):
                    if self.device == "cuda":
                        start_event = torch.cuda.Event(enable_timing=True)
                        end_event = torch.cuda.Event(enable_timing=True)
                        samples = []
                        
                        _ = sparse_tensor @ x
                        torch.cuda.synchronize()

                        start_event.record()
                        if format_name.startswith(("coo", "csr", "csc")):
                            _ = torch.matmul(sparse_tensor, x)
                        elif format_name.startswith("bsr"):
                            _ = sparse_tensor @ x
                        end_event.record()

                        while not end_event.query():
                            if self.nvml_initialized:
                                try:
                                    util = pynvml.nvmlDeviceGetUtilizationRates(self.nvml_handle)
                                    samples.append(util.gpu)
                                except pynvml.NVMLError:
                                    pass
                            time.sleep(0.001)
                        
                        torch.cuda.synchronize()
                        times.append(start_event.elapsed_time(end_event))
                        util_samples.append(np.mean(samples) if samples else np.nan)
                    else:
                        start_time = time.time()
                        if format_name.startswith(("coo", "csr", "csc")):
                            _ = torch.matmul(sparse_tensor, x)
                        elif format_name.startswith("bsr"):
                            _ = sparse_tensor @ x
                        times.append((time.time() - start_time) * 1000)
                        
                matmul_times[format_name] = np.nanmean(times)
                
                if self.device == "cuda":
                    gpu_utilization[format_name] = np.nanmean(util_samples)
                else:
                    gpu_utilization[format_name] = np.nan
                
        return matmul_times, gpu_utilization
        
    def process_mtx_file(self, file_path):
        try:
            scipy_mat = self.load_mtx_file(file_path)
            
            matrix_info = {
                "filename": os.path.basename(file_path),
                "shape": scipy_mat.shape,
                "nnz": scipy_mat.nnz,
                "density": scipy_mat.nnz / (scipy_mat.shape[0] * scipy_mat.shape[1])
            }
            
            formats, conversion_times, memory_usage = self.convert_to_pytorch_sparse_formats(scipy_mat)
            
            matmul_times, gpu_util = self.measure_matmul_performance(formats, scipy_mat.shape)
            
            file_results = []
            for format_name in formats.keys():
                result = {
                    **matrix_info,
                    "format": format_name,
                    "conversion_time": conversion_times.get(format_name, np.nan),
                    "memory_usage_mb": memory_usage.get(format_name, np.nan),
                    "matmul_time_ms": matmul_times.get(format_name, np.nan),
                    "gpu_utilization": gpu_util.get(format_name, np.nan)
                }
                file_results.append(result)
                self.results.append(result)
            
            for tensor in formats.values():
                if tensor is not None:
                    del tensor
            torch.cuda.empty_cache() if self.device == "cuda" else None
            gc.collect()
            
            return file_results
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return []
        
    def process_mtx_folder(self, folder_path):
        mtx_files = glob.glob(os.path.join(folder_path, "*.mtx"))
        
        if not mtx_files:
            print(f"No MTX files found in {folder_path}")
            return None
        
        for mtx_file in mtx_files:
            self.process_mtx_file(mtx_file)
        
        self.results_df = pd.DataFrame(self.results)
        return self.results_df
        
    def get_results_dataframe(self):
        if self.results_df is None and self.results:
            self.results_df = pd.DataFrame(self.results)
        return self.results_df
        
    def plot_results(self, save_path=None):
        df = self.get_results_dataframe()
        
        if df is None or df.empty:
            print("No data to plot.")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        sns.boxplot(x="format", y="conversion_time", data=df, ax=axes[0, 0])
        axes[0, 0].set_title("Conversion Time by Format")
        axes[0, 0].set_ylabel("Time (seconds)")
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        sns.boxplot(x="format", y="memory_usage_mb", data=df, ax=axes[0, 1])
        axes[0, 1].set_title("Memory Usage by Format")
        axes[0, 1].set_ylabel("Memory (MB)")
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        sns.boxplot(x="format", y="matmul_time_ms", data=df, ax=axes[1, 0])
        axes[1, 0].set_title("Matrix-Vector Multiplication Time by Format")
        axes[1, 0].set_ylabel("Time (ms)")
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        for format_name in df["format"].unique():
            format_df = df[df["format"] == format_name]
            axes[1, 1].scatter(format_df["density"], format_df["matmul_time_ms"], label=format_name, alpha=0.7)
        
        axes[1, 1].set_title("Matrix-Vector Multiplication Time vs Density")
        axes[1, 1].set_xlabel("Density")
        axes[1, 1].set_ylabel("Time (ms)")
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
            return None
        else:
            return None
