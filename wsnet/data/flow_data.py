# ANSYS FLuent Simulation Sequence Dataset Module
# Author: Shengning Wang

import re
import glob
import torch
import numpy as np
from pathlib import Path
from torch import Tensor
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from typing import List, Dict, Optional, Union, Tuple

from wsnet.utils.hue_logger import hue, logger


class FlowData(Dataset):
    """
    High-performance dataset for autoregressive CFD simulation sequences.

    This class handles the ingestion of ANSYS Fluent data, supporting automated 
    caching, spatial/temporal subsampling, and sliding-window augmentation.

    Attributes:
        seqs (List[Tensor]): List of sequence tensors. Each: (seq_len, num_nodes, num_channels).
        coords (List[Tensor]): List of coordinate tensors. Each: (num_nodes, spatial_dim).
        spatial_dim (int): Dimensionality of the simulation (2 or 3).
    """

    def __init__(self, data_dir: Union[str, Path], case_names: List[str], spatial_dim: int = 2,
        limits: Optional[Tuple[int, int]] = None, force_rebuild: bool = False) -> None:
        """
        Initializes the dataset by loading cached tensors or parsing raw text files.

        Args:
            data_dir: Root directory of the dataset (contains raw_data/ and .pt files).
            case_names: List of case identifiers (e.g., ["case_1000"]).
            spatial_dim: Spatial dimensionality (2 or 3).
            limits: (max_seq_len, max_nodes) for downsampling.
            force_rebuild: If True, ignores .pt cache and re-parses raw data.
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw_data"
        self.case_names = case_names
        self.spatial_dim = spatial_dim
        self.limits = limits

        self.seqs: List[Tensor] = []
        self.coords: List[Tensor] = []
        self.start_ts_norm: List[float] = []
        self.dts_norm: List[float] = []

        logger.info(f"initializing dataset with {hue.m}{len(case_names)}{hue.q} cases...")

        for name in tqdm(self.case_names, desc="[FlowData] data loading", leave=False, dynamic_ncols=True):
            cache_path = self.data_dir / f"{name}.pt"

            if cache_path.exists() and not force_rebuild:
                data = torch.load(cache_path, weights_only=True)
            else:
                data = self._ingest_raw_case(name, cache_path)

            if data:
                states_tensor = data["states"]   # (seq_len, num_nodes, num_channels)
                coords_tensor = data["coords"]   # (num_nodes, spatial_dim)

                # optional subsampling
                if self.limits:
                    states_tensor, coords_tensor = self._subsample(states_tensor, coords_tensor)

                self.seqs.append(states_tensor)
                self.coords.append(coords_tensor)

                T = states_tensor.shape[0]
                self.start_ts_norm.append(0.0)
                self.dts_norm.append(1.0 / (T - 1) if T > 1 else 0.0)

        logger.info(f"{hue.g}dataset initialized.{hue.q} cases: {hue.m}{len(self.seqs)}{hue.q}, "
                    f"frames: {hue.m}{self.seqs[0].shape[0]}{hue.q}, "
                    f"nodes: {hue.m}{self.seqs[0].shape[1]}{hue.q}, "
                    f"channels: {hue.m}{self.seqs[0].shape[2]}{hue.q}")

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, float, float]:
        return self.seqs[idx], self.coords[idx], self.start_ts_norm[idx], self.dts_norm[idx]

    def get_stats(self) -> Tuple[Tensor, Tensor]:
        """Calculates mean and std across the feature dimension (C)."""
        all_seqs = torch.cat(self.seqs, dim=0).view(-1, self.seqs[0].shape[-1])
        mean = all_seqs.mean(dim=0).view(1, 1, -1)
        std = all_seqs.std(dim=0).view(1, 1, -1)
        std[std < 1e-7] = 1.0
        return mean, std

    def get_bbox(self) -> Tensor:
        """Calculates the global spatial bounding box [[min_dims], [max_dims]]."""
        all_c = torch.cat(self.coords, dim=0)
        return torch.stack([all_c.min(dim=0)[0], all_c.max(dim=0)[0]], dim=0)

    def _ingest_raw_case(self, case_name: str, save_path: Path) -> Optional[Dict[str, Tensor]]:
        """Parses Fluent .txt files from the raw_data directory."""
        case_path = self.raw_dir / case_name

        def _natural_sort(s):
            name = Path(s).name
            priority = 0 if "-" not in name else 1
            parts = [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", name)]
            return [priority] + parts

        files = sorted(glob.glob(str(case_path / "*.txt")), key=_natural_sort)
        if not files: return None

        states: List[np.ndarray] = []
        coords: Optional[np.ndarray] = None

        for f in files:
            # format: [Index, x, y, (z), P, Vx, Vy, (Vz), T]
            data = np.loadtxt(f, skiprows=1, dtype=np.float32)

            if self.spatial_dim == 2:
                # [x:1, y:2], [P:3, V_x:4, V_y:5, T:6]
                c = data[:, 1:3]
                s = data[:, [4, 5, 3, 6]]
            else:
                # [x:1, y:2, z:3], [P:4, V_x:5, V_y:6, V_z:7, T:8]
                c = data[:, 1:4]
                s = data[:, [5, 6, 7, 4, 8]]

            states.append(s)
            if coords is None: coords = c

        states_tensor = torch.from_numpy(np.stack(states, axis=0))
        coords_tensor = torch.from_numpy(coords)

        payload = {"states": states_tensor, "coords": coords_tensor}
        torch.save(payload, save_path)
        return payload

    def _subsample(self, states: Tensor, coords: Tensor) -> Tuple[Tensor, Tensor]:
        """Performs temporal and spatial downsampling via equidistant indexing."""
        max_t, max_n = self.limits
        curr_t, curr_n, _ = states.shape

        if curr_t > max_t:
            logger.info(f'subsampling sequence length to {hue.m}{max_t}{hue.q}...')
            indices = torch.linspace(0, curr_t - 1, max_t).long()
            states = torch.index_select(states, 0, indices)

        if curr_n > max_n:
            logger.info(f'subsampling nodes number to {hue.m}{max_n}{hue.q}...')
            indices = torch.linspace(0, curr_n - 1, max_n).long()
            states = torch.index_select(states, 1, indices)
            coords = torch.index_select(coords, 0, indices)

        return states, coords

    @staticmethod
    def discover_cases(data_dir: Union[str, Path] = "./dataset", prefix: str = "case") -> List[str]:
        """
        Identifies cases by checking for .pt caches first, falling back to raw_data.

        Args:
            data_dir (Union[str, Path]): Root directory of the dataset.
            prefix: The naming prefix for cases.

        Returns:
            List[str]: Sorted unique case names.
        """
        path = Path(data_dir)
        raw_path = path / "raw_data"

        # priority 1: find existing .pt files in root
        pt_cases = [f.stem for f in path.glob(f"{prefix}_*.pt")]

        # priority 2: find raw directories if no .pt or for new data
        dir_cases = []
        if raw_path.exists():
            dir_cases = [d.name for d in raw_path.iterdir() if d.is_dir() and d.name.startswith(prefix)]

        # merge and sort
        all_cases = sorted(list(set(pt_cases + dir_cases)))
        return all_cases

    @staticmethod
    def augment_windows(dataset: "FlowData", win_len: int, win_stride: int) -> None:
        """Applies vectorized sliding window slicing for temporal augmentation."""
        new_seqs: List[Tensor] = []
        new_coords: List[Tensor] = []
        new_start_ts_norm: List[float] = []
        new_dts_norm: List[float] = []

        logger.info(f"augmenting dataset with {hue.m}{len(dataset)}{hue.q} cases...")

        pbar = tqdm(zip(dataset.seqs, dataset.coords, dataset.start_ts_norm, dataset.dts_norm),
                    total=len(dataset),
                    desc=f"[FlowData] data augmenting", leave=False, dynamic_ncols=True)

        for seq, coord, _, _ in pbar:
            if seq.shape[0] < win_len: continue

            T = seq.shape[0]
            dt_norm = 1.0 / (T - 1) if T > 1 else 0.0

            # unfold seq: (win_size, win_len, num_nodes, num_channels)
            unfolded = seq.unfold(0, win_len, win_stride).permute(0, 3, 1, 2)

            for i in range(unfolded.shape[0]):
                new_seqs.append(unfolded[i])
                new_coords.append(coord)
                new_start_ts_norm.append(i * win_stride * dt_norm)
                new_dts_norm.append(dt_norm)

        # shuffle across all cases
        combined = list(zip(new_seqs, new_coords, new_start_ts_norm, new_dts_norm))
        rng = np.random.default_rng(seed=42)
        rng.shuffle(combined)

        dataset.seqs, dataset.coords, dataset.start_ts_norm, dataset.dts_norm = zip(*combined)
        dataset.seqs = list(dataset.seqs)
        dataset.coords = list(dataset.coords)
        dataset.start_ts_norm = list(dataset.start_ts_norm)
        dataset.dts_norm = list(dataset.dts_norm)

        logger.info(f"{hue.g}data augmented.{hue.q} cases: {hue.m}{len(dataset)}{hue.q}, "
                    f"frames: {hue.m}{dataset.seqs[0].shape[0]}{hue.q}, "
                    f"nodes: {hue.m}{dataset.seqs[0].shape[1]}{hue.q}, "
                    f"channels: {hue.m}{dataset.seqs[0].shape[2]}{hue.q}")

    @staticmethod
    def spawn(data_dir: Union[str, Path] = "./dataset", split_counts: Tuple[int, int] = (2, 1),
        spatial_dim: int = 2, limits: Optional[Tuple[int, int]] = None, win_len: int = 16, win_stride: int = 1
    ) -> Tuple["FlowData", "FlowData", "FlowData"]:
        """
        Factory method to partition and augment the dataset into train / val / test splits.

        Args:
            data_dir (Union[str, Path]): Root directory of the dataset.
            split_counts (Tuple[int, int]): Case counts for (val, test).
            spatial_dim (int): Spatial dimension of the data (2 or 3).
            limits (Optional[Tuple[int, int]]): Maximum number of seq_len and nodes to keep (subsampling).
            win_len (int): Length of each temporal window (rollout_steps + 1).
            win_stride (int): Step size between sliding windows.

        Returns:
            Tuple[FlowData, FlowData, FlowData]: (train set, validation set, test set).
        """
        all_cases = FlowData.discover_cases(data_dir)

        rng = np.random.default_rng(seed=42)
        rng.shuffle(all_cases)

        if sum(split_counts) > len(all_cases):
            raise ValueError(f"total counts {sum(split_counts)} exceed available cases {len(all_cases)}")

        num_train = len(all_cases) - split_counts[0] - split_counts[1]
        num_val = split_counts[0]

        splits = {
            "train": all_cases[: num_train],
            "val": all_cases[num_train : num_train + num_val],
            "test": all_cases[num_train + num_val :]
        }

        datasets = []
        for mode, names in splits.items():
            ds = FlowData(data_dir, names, spatial_dim, limits)
            if mode != "test":
                FlowData.augment_windows(ds, win_len, win_stride)
            datasets.append(ds)

        return tuple(datasets)
