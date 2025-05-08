import os
import torch
import numpy as np
from typing import Tuple, List, Dict, Optional, Union
from enum import Enum

from .sparse_matrix_types import SparseMatrixType

class SparseMatrixGenerator:
    def __init__(self, seed: int = 42):
        self.device = 'cuda'
        self.seed = seed
        self._set_seed(seed)

    def _set_seed(self, seed: int) -> None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    def generate(self, matrix_type: Union[str, SparseMatrixType], size: int, **kwargs) -> torch.Tensor:
        if isinstance(matrix_type, str):
            try:
                matrix_type = SparseMatrixType(matrix_type.lower())
            except ValueError:
                raise ValueError(f"Unknown matrix type: {matrix_type}. Valid types are: {[t.value for t in SparseMatrixType]}")
        
        if matrix_type == SparseMatrixType.RANDOM:
            return self.create_random_sparse(size, **kwargs)
        elif matrix_type == SparseMatrixType.BANDED:
            return self.create_banded_sparse(size, **kwargs)
        elif matrix_type == SparseMatrixType.BLOCK_DIAGONAL:
            return self.create_block_diagonal_sparse(size, **kwargs)
        elif matrix_type == SparseMatrixType.TRIDIAGONAL:
            return self.create_tridiagonal_sparse(size, **kwargs)
        elif matrix_type == SparseMatrixType.CHECKERBOARD:
            return self.create_checkerboard_sparse(size, **kwargs)
        else:
            raise ValueError(f"Unsupported matrix type: {matrix_type}")

    def create_random_sparse(self, size: int, density: float = 0.05, value_range: Tuple[float, float] = (-1.0, 1.0), seed: Optional[int] = None) -> torch.Tensor:
        min_val, max_val = value_range
        total_elements = size * size
        num_nonzero = min(int(total_elements * density), total_elements)

        if num_nonzero == 0:
            return torch.sparse_coo_tensor(
                indices=torch.empty((2, 0), dtype=torch.long, device=self.device),
                values=torch.empty((0,), device=self.device),
                size=(size, size),
                device=self.device
            )

        linear_indices = torch.randperm(total_elements, device=self.device)[:num_nonzero]
        rows = linear_indices // size
        cols = linear_indices % size
        indices = torch.stack([rows, cols])

        values = torch.rand(num_nonzero, device=self.device) * (max_val - min_val) + min_val

        return torch.sparse_coo_tensor(indices=indices, values=values, size=(size, size), device=self.device).coalesce()

    def create_banded_sparse(self, size: int, num_bands: int = 3, band_width: int = 1, value_range: Tuple[float, float] = (0.5, 1.5), random_values: bool = False, seed: Optional[int] = None) -> torch.Tensor:
        indices = [[], []]
        values = []

        min_val, max_val = value_range

        for k in range(-num_bands//2, num_bands//2 + 1):
            band_start = max(0, k)
            band_end = min(size, size + k)

            for i in range(band_start, band_end):
                j = i - k
                if 0 <= j < size:
                    for offset in range(-band_width//2, band_width//2 + 1):
                        if 0 <= j + offset < size:
                            indices[0].append(i)
                            indices[1].append(j + offset)
                            if random_values:
                                values.append(min_val + (max_val - min_val) * torch.rand(1).item())
                            else:
                                values.append((min_val + max_val) / 2)
        
        return torch.sparse_coo_tensor(torch.tensor(indices, device=self.device), torch.tensor(values, device=self.device), (size, size))

    def create_block_diagonal_sparse(self, size: int, num_blocks: int = 4, block_density: float = 0.3, value_range: Tuple[float, float] = (-1.0, 1.0), seed: Optional[int] = None) -> torch.Tensor:
        block_size = size // num_blocks
        indices = [[], []]
        values = []

        min_val, max_val = value_range

        for b in range(num_blocks):
            row_offset = b * block_size
            col_offset = b * block_size

            nnz = int(block_size * block_size * block_density)

            for _ in range(nnz):
                i = row_offset + torch.randint(0, block_size, (1,)).item()
                j = col_offset + torch.randint(0, block_size, (1,)).item()
                indices[0].append(i)
                indices[1].append(j)
                values.append(min_val + (max_val - min_val) * torch.rand(1).item())

        return torch.sparse_coo_tensor(torch.tensor(indices, device=self.device), torch.tensor(values, device=self.device), (size, size))

    def create_tridiagonal_sparse(self, size: int, main_diag_value: float = 2.0, off_diag_value: float = -1.0, random_variation: float = 0.0, seed: Optional[int] = None) -> torch.Tensor:
        indices = [[], []]
        values = []

        for i in range(size):
            indices[0].append(i)
            indices[1].append(i)
            if random_variation > 0:
                variation = (2 * torch.rand(1).item() - 1) * random_variation
                values.append(main_diag_value + variation)
            else:
                values.append(main_diag_value)

        for i in range(size-1):
            indices[0].append(i)
            indices[1].append(i+1)
            if random_variation > 0:
                variation = (2 * torch.rand(1).item() - 1) * random_variation
                values.append(off_diag_value + variation)
            else:
                values.append(off_diag_value)

        for i in range(size-1):
            indices[0].append(i+1)
            indices[1].append(i)
            if random_variation > 0:
                variation = (2 * torch.rand(1).item() - 1) * random_variation
                values.append(off_diag_value + variation)
            else:
                values.append(off_diag_value)

        return torch.sparse_coo_tensor(torch.tensor(indices, device=self.device), torch.tensor(values, device=self.device), (size, size))

    def create_checkerboard_sparse(self, size: int, value: float = 1.0, random_values: bool = False, value_range: Tuple[float, float] = (0.5, 1.5), seed: Optional[int] = None) -> torch.Tensor:
        indices = [[], []]
        values = []

        min_val, max_val = value_range

        for i in range(size):
            for j in range(size):
                if (i + j) % 2 == 0:
                    indices[0].append(i)
                    indices[1].append(j)
                    if random_values:
                        values.append(min_val + (max_val - min_val) * torch.rand(1).item())
                    else:
                        values.append(value)

        return torch.sparse_coo_tensor(torch.tensor(indices, device=self.device), torch.tensor(values, device=self.device), (size, size))

    def save_to_matrix_market(self, sparse_tensor: torch.Tensor, filename: str) -> None:
        tensor_cpu = sparse_tensor.coalesce().cpu()
        
        indices = tensor_cpu.indices()
        values = tensor_cpu.values()
        rows, cols = indices[0].numpy() + 1, indices[1].numpy() + 1
        data = values.numpy()
        
        order = np.lexsort((cols, rows))
        rows = rows[order]
        cols = cols[order]
        data = data[order]
        
        with open(filename, 'w') as f:
            f.write("%%MatrixMarket matrix coordinate real general\n")
            f.write(f"{sparse_tensor.size(0)} {sparse_tensor.size(1)} {len(rows)}\n")
            for r, c, v in zip(rows, cols, data):
                f.write(f"{r} {c} {v:.9e}\n")

    def generate_dataset(self, sizes: List[int], types: List[Union[str, SparseMatrixType]], num_samples: int, output_dir: str = "./", filename_prefix: str = "matrix", filename_fmt: str = "{prefix}_{index:06d}_{type}_size{size}.mtx", overwrite: bool = False, type_weights: Optional[List[float]] = None, size_weights: Optional[List[float]] = None, params_by_type: Optional[Dict] = None, seed: Optional[int] = None) -> List[str]:
        os.makedirs(output_dir, exist_ok=True)

        type_weights = type_weights or [1.0] * len(types)
        size_weights = size_weights or [1.0] * len(sizes)
        params_by_type = params_by_type or {}

        type_weights = np.array(type_weights) / sum(type_weights)
        size_weights = np.array(size_weights) / sum(size_weights)

        file_paths = []

        for idx in range(num_samples):
            matrix_type = np.random.choice(types, p=type_weights)
            size = np.random.choice(sizes, p=size_weights)

            if isinstance(matrix_type, SparseMatrixType):
                type_str = matrix_type.value
            else:
                type_str = str(matrix_type).lower()

            filename = filename_fmt.format(
                prefix=filename_prefix,
                index=idx,
                type=type_str,
                size=size
            )
            file_path = os.path.join(output_dir, filename)

            if os.path.exists(file_path) and not overwrite:
                raise FileExistsError(f"File {file_path} already exists. Use overwrite=True to replace.")

            params = params_by_type.get(matrix_type, {})

            matrix = self.generate(matrix_type, size, **params)
            self.save_to_matrix_market(matrix, file_path)

            file_paths.append(file_path)

            del matrix

        return file_paths

    def generate_train_val_sets(self, sizes: List[int], types: List[Union[str, SparseMatrixType]], num_train: int, num_val: int, train_dir: str = "./train", val_dir: str = "./test", train_prefix: str = "train", val_prefix: str = "test", filename_fmt: str = "{prefix}_{index:06d}_{type}_size{size}.mtx", overwrite: bool = False, type_weights: Optional[List[float]] = None, size_weights: Optional[List[float]] = None, params_by_type: Optional[Dict] = None, train_seed: int = 42, val_seed: int = 43) -> Tuple[List[str], List[str]]:
        train_files = self.generate_dataset(sizes=sizes, types=types, num_samples=num_train, output_dir=train_dir, filename_prefix=train_prefix, filename_fmt=filename_fmt, overwrite=overwrite, type_weights=type_weights, size_weights=size_weights, params_by_type=params_by_type, seed=train_seed)

        val_files = self.generate_dataset(sizes=sizes, types=types, num_samples=num_val, output_dir=val_dir, filename_prefix=val_prefix, filename_fmt=filename_fmt, overwrite=overwrite, type_weights=type_weights, size_weights=size_weights, params_by_type=params_by_type, seed=val_seed)

        return train_files, val_files
