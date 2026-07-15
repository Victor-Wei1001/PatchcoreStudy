import os
from enum import Enum
from pathlib import Path

import PIL.Image
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode


CLASSNAMES = [
    "breakfast_box",
    "juice_bottle",
    "pushpins",
    "screw_bag",
    "splicing_connectors",
]
ANOMALY_TYPES = ["logical_anomalies", "structural_anomalies"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "validation"
    TEST = "test"


class MVTecLoCoDataset(torch.utils.data.Dataset):
    """Reader for the official MVTec LOCO AD directory layout."""

    def __init__(
        self,
        source,
        classname,
        resize=256,
        imagesize=224,
        split=DatasetSplit.TRAIN,
        test_anomaly_types=None,
        **kwargs,
    ):
        super().__init__()
        self.source = Path(source)
        self.classname = str(classname)
        self.split = split
        self.test_anomaly_types = list(test_anomaly_types or ANOMALY_TYPES)
        self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()

        self.transform_img = transforms.Compose(
            [
                transforms.Resize(resize, interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(imagesize),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
        self.transform_mean = IMAGENET_MEAN
        self.transform_std = IMAGENET_STD
        self.transform_mask = transforms.Compose(
            [
                transforms.Resize(resize, interpolation=InterpolationMode.NEAREST),
                transforms.CenterCrop(imagesize),
                transforms.ToTensor(),
            ]
        )
        self.imagesize = (3, imagesize, imagesize)

    def __getitem__(self, idx):
        classname, anomaly, image_path, mask_dir = self.data_to_iterate[idx]
        image = PIL.Image.open(image_path).convert("RGB")
        image = self.transform_img(image)

        if self.split == DatasetSplit.TEST and mask_dir is not None:
            mask = self._combined_mask(Path(mask_dir))
        else:
            mask = torch.zeros([1, *image.size()[1:]])

        return {
            "image": image,
            "mask": mask,
            "classname": classname,
            "anomaly": anomaly,
            "is_anomaly": int(anomaly != "good"),
            "image_name": os.path.join(*Path(image_path).parts[-4:]),
            "image_path": str(image_path),
        }

    def __len__(self):
        return len(self.data_to_iterate)

    @staticmethod
    def _sorted_files(path):
        if not path.exists():
            return []
        return sorted(
            [item for item in path.iterdir() if item.is_file()],
            key=lambda item: item.stem,
        )

    def _combined_mask(self, mask_dir):
        """Merge all mask components belonging to one logical anomaly image."""
        mask_files = self._sorted_files(mask_dir)
        if not mask_files:
            raise FileNotFoundError("No mask files found in {}".format(mask_dir))

        merged = None
        for mask_path in mask_files:
            mask = PIL.Image.open(mask_path).convert("L")
            mask = (self.transform_mask(mask) > 0).float()
            merged = mask if merged is None else torch.maximum(merged, mask)
        return merged

    def get_image_data(self):
        category_dir = self.source / self.classname
        if not category_dir.exists():
            raise FileNotFoundError("MVTec LoCo category not found: {}".format(category_dir))

        rows = []
        if self.split in (DatasetSplit.TRAIN, DatasetSplit.VAL):
            normal = self._sorted_files(category_dir / self.split.value / "good")
            rows = [("good", path, None) for path in normal]
        elif self.split == DatasetSplit.TEST:
            normal = self._sorted_files(category_dir / "test" / "good")
            rows.extend(("good", path, None) for path in normal)
            for anomaly_type in self.test_anomaly_types:
                image_dir = category_dir / "test" / anomaly_type
                mask_root = category_dir / "ground_truth" / anomaly_type
                for path in self._sorted_files(image_dir):
                    mask_dir = mask_root / path.stem
                    if not mask_dir.exists():
                        raise FileNotFoundError(
                            "Mask directory not found for {}: {}".format(path, mask_dir)
                        )
                    rows.append((anomaly_type, path, str(mask_dir)))
        else:
            raise ValueError("Unsupported split: {}".format(self.split))

        if not rows:
            raise ValueError(
                "No MVTec LoCo images selected for category={} split={}".format(
                    self.classname, self.split.value
                )
            )

        data_to_iterate = [
            [self.classname, anomaly, str(image_path), mask_dir]
            for anomaly, image_path, mask_dir in rows
        ]
        imgpaths_per_class = {self.classname: {"good": [], "anomaly": []}}
        for classname, anomaly, image_path, mask_dir in data_to_iterate:
            key = "good" if anomaly == "good" else "anomaly"
            imgpaths_per_class[classname][key].append(image_path)
        return imgpaths_per_class, data_to_iterate
