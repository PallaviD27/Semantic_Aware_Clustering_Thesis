import ast
import pathlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

# try:
#     import rasterio
# except ImportError as e:
#     raise ImportError(
#         "rasterio is required for reading BigEarthNet GeoTIFF Files. "
#         "Install it using: pip install rasterio"
#     ) from e
# Defining the 19 labels in the dataset (as per the Description document)

BIGEARTHNET_19_CLASSES = [
    "Urban fabric",
    "Industrial or commercial units",
    "Arable land",
    "Permanent crops",
    "Pastures",
    "Complex cultivation patterns",
    "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Agro-forestry areas",
    "Broad-leaved forest",
    "Coniferous forest",
    "Mixed forest",
    "Natural grassland and sparsely vegetated areas",
    "Moors, heathland and sclerophyllous vegetation",
    "Transitional woodland, shrub",
    "Beaches, dunes, sands",
    "Inland wetlands",
    "Coastal wetlands",
    "Inland waters",
    "Marine waters",
]

# Creating dictionary of labels
CLASS_TO_IDX ={
    class_name: idx
    for idx, class_name in enumerate(BIGEARTHNET_19_CLASSES)
}

# Sentinel-2 level 2A Bands
#  B10 is not used
DEFAULT_S2_BANDS = [
    "B01", "B02", "B03", "B04",
    "B05", "B06", "B07", "B08",
    "B8A", "B09", "B11", "B12",
]

# Dictionary with tasks for each client
CLIENT_TASKS = {
    # Client 0: Forest/Vegetation
    0: [
        CLASS_TO_IDX["Broad-leaved forest"],
        CLASS_TO_IDX["Coniferous forest"],
        CLASS_TO_IDX["Mixed forest"],
        CLASS_TO_IDX["Transitional woodland, shrub"],
        CLASS_TO_IDX["Natural grassland and sparsely vegetated areas"],
        CLASS_TO_IDX["Moors, heathland and sclerophyllous vegetation"],
        ],

    # Client 1: agriculture + vegetation
    1:[
        CLASS_TO_IDX["Arable land"],
        CLASS_TO_IDX["Permanent crops"],
        CLASS_TO_IDX["Pastures"],
        CLASS_TO_IDX["Complex cultivation patterns"],
        CLASS_TO_IDX["Land principally occupied by agriculture, with significant areas of natural vegetation"],
        CLASS_TO_IDX["Agro-forestry areas"],
        CLASS_TO_IDX["Broad-leaved forest"],
        CLASS_TO_IDX["Coniferous forest"],
        CLASS_TO_IDX["Mixed forest"],
        CLASS_TO_IDX["Natural grassland and sparsely vegetated areas"],
        CLASS_TO_IDX["Transitional woodland, shrub"],
        CLASS_TO_IDX["Moors, heathland and sclerophyllous vegetation"],
    ],

    # Client 2: urban + water
    2: [
        CLASS_TO_IDX["Urban fabric"],
        CLASS_TO_IDX["Industrial or commercial units"],
        CLASS_TO_IDX["Beaches, dunes, sands"],
        CLASS_TO_IDX["Inland wetlands"],
        CLASS_TO_IDX["Coastal wetlands"],
        CLASS_TO_IDX["Inland waters"],
        CLASS_TO_IDX["Marine waters"],
    ],
}

def _parse_labels(labels):
    """
    Convert labels into a normal Python list.

    If labels come from parquet, they may already be list-like.
    If labels come from metadata_light.csv, they are stored like:
        Arable land|Broad-leaved forest|Mixed forest
    """

    if isinstance(labels, str):
        labels = labels.split("|")

    if isinstance(labels, np.ndarray):
        labels = labels.tolist()

    return list(labels)

def labels_to_multihot(labels):
    '''
     Converts:
        ["Pastures", "Arable land", "Mixed forest"]

    into:
        tensor([0, 0, 1, 0, 1, ..., 0])

    Shape:
        [19]

    dtype:
        torch.float32
    '''
    labels = _parse_labels(labels)

    y = torch.zeros(len(BIGEARTHNET_19_CLASSES), dtype=torch.float32)

    for label in labels:
        if label in CLASS_TO_IDX:
            y[CLASS_TO_IDX[label]] = 1.0

    return y

#  Not sure why this is needed
def patch_id_to_tile_id(patch_id):
    '''
    Example patch_id:
        S2B_MSIL2A_20180421T100029_N9999_R122_T33TWM_00_00

    Tile folder:
        S2B_MSIL2A_20180421T100029_N9999_R122_T33TWM

    So we remove the last two parts: H-order and V-order.
    '''

    parts = patch_id.split("_")
    return "_".join(parts[:-2])

class BigEarthNetS2Dataset(Dataset):
    '''
     PyTorch Dataset for BigEarthNet-S2.

    Returns:
        image: Tensor [12, image_size, image_size], dtype float32
        label: Tensor [19], dtype float32

    This is different from MNIST:
        MNIST label: one integer class
        BigEarthNet label: multi-hot vector
    '''

    def __init__(
            self,
            s2_dir,
            metadata_path,
            split,
            bands=None,
            image_size = 120,
    ):
        self.s2_dir = Path(s2_dir)
        self.metadata_path = Path(metadata_path)
        self.split = split.lower()
        self.bands = bands if bands is not None else DEFAULT_S2_BANDS
        self.image_size =image_size

        if not self.s2_dir.exists():
            raise FileNotFoundError(f"BigEarthNet-S2 folder not found: {self.s2_dir}")

        if not self.metadata_path.exists():
            raise FileNotFoundError(f"metadata.parquet not found: {self.metadata_path}")

        print("[BigEarthNetS2Dataset] Checking paths...")
        print("S2 dir:", self.s2_dir)
        print("Metadata:", self.metadata_path)

        print("[BigEarthNetS2Dataset] Reading metadata file...", flush=True)

        if self.metadata_path.suffix == ".csv":
            df = pd.read_csv(self.metadata_path)
        else:
            df = pd.read_parquet(self.metadata_path)
            df = df[["patch_id", "labels", "split"]]

        print("[BigEarthNetS2Dataset] metadata loaded", flush=True)
        print("Metadata shape:", df.shape, flush=True)
        print("Metadata columns:", df.columns.tolist(), flush=True)

        print("[BigEarthNetS2Dataset] Split counts:", flush=True)
        print(df["split"].value_counts(), flush=True)

        required_columns = ["patch_id", "labels", "split"]
        for column in required_columns:
            if column not in df.columns:
                raise ValueError(f"metadata.parquet does not contain required column: {column}")

        df["split"] = df["split"].astype(str).str.lower()
        df = df[df["split"] == self.split].reset_index(drop=True)

        if len(df)==0:
            raise ValueError(f"No samples found in the split = '{split}'.")

        self.df=df

        print(f"[BigEarthNetS2Dataset] After split filtering: {len(self.df)} samples", flush=True)
        print("[BigEarthNetS2Dataset] Creating multi-hot label targets...", flush=True)

        # Pre-compute labels only. Images are not loaded
        self.targets = torch.stack([labels_to_multihot(row_labels)
                                    for row_labels in self.df["labels"]
                                    ])

        print("[BigEarthNetS2Dataset] Multi-hot label targets created", flush=True)

        print(f"[BigEarthNetS2Dataset] split = {self.split}, samples = {len(self.df)}")
        print(f"[BigEarthNetS2Dataset] image dtype: float32")
        print(f"[BigEarthNetS2Dataset] label dtype: float32 multi-hot[19]")

    def __len__(self):
        return len(self.df)

    def _read_one_band(self, tif_path):
        try:
            import rasterio
        except ImportError as e:
            raise ImportError(
                "rasterio is required for reading BigEarthNet GeoTIFF files. "
                "Install it using: pip install rasterio"
            ) from e
        with rasterio.open(tif_path) as src:
            arr = src.read(1)

        x = torch.from_numpy(arr.astype(np.float32))

        # Basic cleaning and normalization.
        # Sentinel-2 values are commonly scaled reflectance-like values.

        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = torch.clamp(x, min=0.0, max=10000.0) / 10000.0

        # Resize every band to same size
        # F.interpolate expects [N, C, H, W]
        x = x.unsqueeze(0).unsqueeze(0)

        x = F.interpolate(
            x,
            size = (self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

        x = x.squeeze(0).squeeze(0)

        return x

    def __getitem__(self, index):
        row = self.df.iloc[index]

        patch_id = row["patch_id"]
        tile_id = patch_id_to_tile_id(patch_id)

        patch_dir = self.s2_dir / tile_id / patch_id

        if not patch_dir.exists():
            raise FileNotFoundError(f"Patch folder not found: {patch_dir}")

        image_bands =[]

        for band in self.bands:
            tif_path = patch_dir / f"{patch_id}_{band}.tif"

            if not tif_path.exists():
                raise FileNotFoundError(
                    f"Missing band file: {tif_path}\n"
                    f"Check whether your extracted BigEarthNet-S2 folder has this band."
                )

            band_tensor = self._read_one_band(tif_path)
            image_bands.append(band_tensor)

        image = torch.stack(image_bands, dim=0).float()
        label = self.targets[index].float()

        return image, label

def build_client_partitions(
        dataset,
        client_tasks,
        max_samples_per_client=1000,
        seed=42
):
    '''
    Creates client partitions.
    Each sample has a 19-dimensional multi-hot label vector.
    For each client, counting how many of that client's task labels are present
    in a sample.
    The sample goes to the best matching client.
    '''

    rng = np.random.default_rng(seed)

    user_groups = {
        client_id: []
        for client_id in client_tasks.keys()

    }

    indices = rng.permutation(len(dataset))

    targets = dataset.targets.numpy()

    for index in indices:
        label_vector = targets[index]

        scores ={}

        for client_id, task_indices in client_tasks.items():
            score = label_vector[task_indices].sum()

            scores[client_id] = score

        best_score = max(scores.values())

        if best_score == 0:
            continue

        best_clients = [
            client_id
            for client_id, score in scores.items()
            if score == best_score

        ]

        chosen_client = int(rng.choice(best_clients))

        if len(user_groups[chosen_client]) < max_samples_per_client:
            user_groups[chosen_client].append(int(index))

        full = all(len(user_groups[client_id]) >= max_samples_per_client
                   for client_id in user_groups.keys()
                   )
        if full:
            break

    for client_id in user_groups:
        user_groups[client_id] = np.array(user_groups[client_id], dtype=np.int64)

    print("\n[BigEarthNet partition]")

    for client_id, indices_for_client in user_groups.items():
        print(f"Client {client_id}: {len(indices_for_client)} samples")

    return user_groups

def print_bigearthnet_client_distribution(dataset, user_groups, client_tasks, title):
    print(f"\n=========={title}===========")

    for client_id, indices in user_groups.items():

        print(f"Client {client_id}: {len(indices)} samples")

        if len(indices) == 0:
            print("No samples assigned")
            continue

        label_counts = dataset.targets[indices].sum(dim=0)

        task_indices = set(client_tasks[client_id])

        for class_idx, class_name in enumerate(BIGEARTHNET_19_CLASSES):
            count = int(label_counts[class_idx].item())

            marker = "*" if class_idx in task_indices else " "

            if count >0:
                print(f"{marker} {class_idx:02d} | {class_name} : {count}")

        print("* = this client's task label")




