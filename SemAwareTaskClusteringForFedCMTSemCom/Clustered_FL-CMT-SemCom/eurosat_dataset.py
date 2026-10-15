from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset


# Explicit global class order; it does not depend on folder sorting.
EUROSAT_CLASSES = (
    "AnnualCrop",                 # 0
    "Forest",                    # 1
    "HerbaceousVegetation",      # 2
    "Highway",                   # 3
    "Industrial",                # 4
    "Pasture",                   # 5
    "PermanentCrop",             # 6
    "Residential",               # 7
    "River",                     # 8
    "SeaLake",                   # 9
)

EUROSAT_CLASS_TO_IDX = {
    class_name: class_index
    for class_index, class_name in enumerate(EUROSAT_CLASSES)
}


# Order of the bands inside each original EuroSAT-MS GeoTIFF.
EUROSAT_TIFF_BANDS = (
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B09",
    "B10",
    "B11",
    "B12",
    "B8A",
)

# Same 12 Sentinel-2 bands as the current BigEarthNet experiment.
EUROSAT_SELECTED_BANDS = (
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B09",
    "B11",
    "B12",
)

# Rasterio uses one-based band indexes. This becomes:
# (1, 2, 3, 4, 5, 6, 7, 8, 13, 9, 11, 12), # B10 is stored at Rasterio index 10 and is excluded.
EUROSAT_RASTERIO_BAND_INDEXES = tuple(
    EUROSAT_TIFF_BANDS.index(band_name) + 1
    for band_name in EUROSAT_SELECTED_BANDS
)


EUROSAT_CLIENT_TASKS = {
    0: (# 3 class vegetation type classification
        EUROSAT_CLASS_TO_IDX["Forest"],
        EUROSAT_CLASS_TO_IDX["HerbaceousVegetation"],
        EUROSAT_CLASS_TO_IDX["Pasture"],
    ),
    1: (# 5 class agriculture/vegetation land-cover classification
        EUROSAT_CLASS_TO_IDX["AnnualCrop"],
        EUROSAT_CLASS_TO_IDX["PermanentCrop"],
        EUROSAT_CLASS_TO_IDX["Forest"],
        EUROSAT_CLASS_TO_IDX["HerbaceousVegetation"],
        EUROSAT_CLASS_TO_IDX["Pasture"],
    ),
    2: (# 3-class water body classification
        EUROSAT_CLASS_TO_IDX["River"],
        EUROSAT_CLASS_TO_IDX["SeaLake"],
        EUROSAT_CLASS_TO_IDX["Residential"],
    ),
}


def _find_class_root(root):
    """Find the directory that directly contains all ten class folders."""

    root = Path(root).expanduser().resolve()

    if not root.exists():
        raise FileNotFoundError(
            f"EuroSAT directory does not exist: {root}\n"
            "Download and extract EuroSAT_MS.zip from Zenodo first."
        )

    candidates = [root]
    candidates.extend(
        annual_crop_dir.parent
        for annual_crop_dir in root.rglob("AnnualCrop")
        if annual_crop_dir.is_dir()
    )

    valid_roots = []
    for candidate in candidates:
        if all((candidate / class_name).is_dir()
               for class_name in EUROSAT_CLASSES):
            resolved = candidate.resolve()
            if resolved not in valid_roots:
                valid_roots.append(resolved)

    if len(valid_roots) == 1:
        return valid_roots[0]

    if not valid_roots:
        raise FileNotFoundError(
            "Could not find the ten EuroSAT class folders below:\n"
            f"{root}\n"
            "Check that EuroSAT_MS.zip was fully extracted."
        )

    raise RuntimeError(
        "More than one EuroSAT class-directory root was found:\n"
        + "\n".join(str(path) for path in valid_roots)
        + "\nPass the exact directory containing the ten class folders."
    )


class EuroSATMSDataset(Dataset):
    """Original 13-band EuroSAT GeoTIFFs, loaded as selected 12-band tensors."""

    def __init__(self, root, reflectance_scale=10000.0):
        self.class_root = _find_class_root(root)
        self.classes = list(EUROSAT_CLASSES)
        self.class_to_idx = dict(EUROSAT_CLASS_TO_IDX)
        self.reflectance_scale = float(reflectance_scale)

        if self.reflectance_scale <= 0:
            raise ValueError("reflectance_scale must be greater than zero.")

        self.samples = []
        self.targets = []

        for class_name in EUROSAT_CLASSES:
            class_index = EUROSAT_CLASS_TO_IDX[class_name]
            class_dir = self.class_root / class_name

            image_paths = sorted(
                path
                for path in class_dir.iterdir()
                if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
            )

            if not image_paths:
                raise RuntimeError(f"No GeoTIFF files found in {class_dir}")

            for image_path in image_paths:
                self.samples.append((image_path, class_index))
                self.targets.append(class_index)

        if len(self.samples) != 27000:
            print(
                "[EuroSAT warning] Expected approximately 27,000 samples, "
                f"but found {len(self.samples)}."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]

        with rasterio.open(image_path) as src:
            if src.count != 13:
                raise RuntimeError(
                    f"Expected 13 bands in {image_path}, found {src.count}."
                )
            if (src.height, src.width) != (64, 64):
                raise RuntimeError(
                    f"Expected a 64×64 image in {image_path}, "
                    f"found {src.height}×{src.width}."
                )

            image = src.read(EUROSAT_RASTERIO_BAND_INDEXES)

        # Convert original Sentinel-2 digital values to a stable float scale.
        # Later apply the identical rule to BigEarthNet for a controlled comparison.
        image = image.astype(np.float32) / self.reflectance_scale
        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        image = np.ascontiguousarray(image)

        image_tensor = torch.from_numpy(image)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return image_tensor, label_tensor


class EuroSATSubset(Dataset):
    """Subset wrapper that exposes targets relative to the subset."""

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        self.targets = torch.tensor(
            [dataset.targets[int(index)] for index in self.indices],
            dtype=torch.long,
        )
        self.classes = dataset.classes
        self.class_to_idx = dataset.class_to_idx

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        original_index = int(self.indices[index])
        return self.dataset[original_index]


def stratified_train_test_split(targets, train_fraction=0.8, seed=42):
    """Create a deterministic split while retaining every class in both sets."""

    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between zero and one.")

    rng = np.random.default_rng(seed)
    targets = np.asarray(targets, dtype=np.int64)
    train_indices = []
    test_indices = []

    for class_index in range(len(EUROSAT_CLASSES)):
        class_indices = np.flatnonzero(targets == class_index)
        rng.shuffle(class_indices)

        split_position = int(round(train_fraction * len(class_indices)))
        train_indices.extend(class_indices[:split_position])
        test_indices.extend(class_indices[split_position:])

    rng.shuffle(train_indices)
    rng.shuffle(test_indices)

    return (
        np.asarray(train_indices, dtype=np.int64),
        np.asarray(test_indices, dtype=np.int64),
    )


def get_eurosat_datasets(root, train_fraction=0.8, seed=42):
    """Load the original Zenodo data and return train/test subsets."""

    full_dataset = EuroSATMSDataset(root=root)

    train_indices, test_indices = stratified_train_test_split(
        full_dataset.targets,
        train_fraction=train_fraction,
        seed=seed,
    )

    train_dataset = EuroSATSubset(full_dataset, train_indices)
    test_dataset = EuroSATSubset(full_dataset, test_indices)

    print(f"[EuroSAT] class root: {full_dataset.class_root}")
    print(f"[EuroSAT] full samples:  {len(full_dataset)}")
    print(f"[EuroSAT] train samples: {len(train_dataset)}")
    print(f"[EuroSAT] test samples:  {len(test_dataset)}")
    print("[EuroSAT] image shape:   [12, 64, 64]")
    print("[EuroSAT] label shape:   scalar integer")

    return train_dataset, test_dataset


def build_eurosat_client_partitions(
    dataset,
    client_tasks,
    max_samples_per_client,
    seed=42,
):
    """Build class-balanced, mutually disjoint client partitions."""

    if max_samples_per_client <= 0:
        raise ValueError("max_samples_per_client must be positive.")

    largest_task = max(len(classes) for classes in client_tasks.values())

    if max_samples_per_client < largest_task:
        raise ValueError(
            f"At least {largest_task} samples per client are required "
            "so every task class can be represented."
        )

    rng = np.random.default_rng(seed)
    targets = torch.as_tensor(dataset.targets).cpu().numpy()

    requested_by_class = {}

    # Calculate the number of samples required from every class.
    for client_id, task_classes in client_tasks.items():
        samples_per_class, remainder = divmod(
            max_samples_per_client,
            len(task_classes),
        )

        for position, global_label in enumerate(task_classes):
            requested_count = samples_per_class

            if position < remainder:
                requested_count += 1

            requested_by_class.setdefault(global_label, []).append(
                (client_id, requested_count)
            )

    user_groups = {
        client_id: []
        for client_id in client_tasks
    }

    # Allocate disjoint samples class by class.
    for global_label, client_requests in requested_by_class.items():
        class_indices = np.flatnonzero(targets == global_label)
        rng.shuffle(class_indices)

        total_requested = sum(
            count for _, count in client_requests
        )

        if total_requested > len(class_indices):
            class_name = EUROSAT_CLASSES[global_label]

            raise ValueError(
                f"Class {class_name} contains {len(class_indices)} samples, "
                f"but the clients require {total_requested}."
            )

        offset = 0

        for client_id, requested_count in client_requests:
            selected = class_indices[
                offset:offset + requested_count
            ]

            user_groups[client_id].extend(selected.tolist())
            offset += requested_count

    # Shuffle samples within every client.
    for client_id, indices in user_groups.items():
        indices = np.asarray(indices, dtype=np.int64)
        rng.shuffle(indices)
        user_groups[client_id] = indices

        assert len(indices) == max_samples_per_client

    return user_groups


def build_eurosat_label_mappings(client_tasks):
    """Map each global EuroSAT class index to a local task-head index."""

    return {
        client_id: {
            global_label: local_label
            for local_label, global_label in enumerate(task_indices)
        }
        for client_id, task_indices in client_tasks.items()
    }


def print_eurosat_client_distribution(
    dataset,
    user_groups,
    client_tasks,
    title,
):
    print(f"\n======= {title} =======")
    targets = torch.as_tensor(dataset.targets)

    for client_id, indices in user_groups.items():
        print(f"\nClient {client_id} | samples={len(indices)}")

        if len(indices) == 0:
            continue

        counts = torch.bincount(
            targets[torch.as_tensor(indices, dtype=torch.long)],
            minlength=len(EUROSAT_CLASSES),
        )

        for global_label in client_tasks[client_id]:
            class_name = EUROSAT_CLASSES[global_label]
            count = int(counts[global_label].item())
            print(f"{global_label:02d} | {class_name}: {count}")