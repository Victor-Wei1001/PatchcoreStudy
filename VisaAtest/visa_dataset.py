import csv
import os
from enum import Enum
from pathlib import Path

import PIL
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode


CLASSNAMES = [
    "candle",
    "capsules",
    "cashew",
    "chewinggum",
    "fryum",
    "macaroni1",
    "macaroni2",
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4",
    "pipe_fryum",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class VisaDataset(torch.utils.data.Dataset):
    """VisA one-class split reader backed by split_csv/1cls.csv."""

    def __init__(
        self,
        source,
        classname,
        resize=256,
        imagesize=224,
        split=DatasetSplit.TRAIN,
        train_val_split=1.0,
        split_csv=None,
        **kwargs,
    ):
        super().__init__()
        self.source = Path(source)
        self.split = split
        self.classname = classname
        self.train_val_split = train_val_split
        self.split_csv = Path(split_csv) if split_csv else self.source / "split_csv" / "1cls.csv"

        if not self.split_csv.exists():
            raise FileNotFoundError("VisA split CSV not found: {}".format(self.split_csv))

        self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()

        self.transform_img = transforms.Compose(
            [
                transforms.Resize(resize),
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
        classname, anomaly, image_path, mask_path = self.data_to_iterate[idx]
        image = PIL.Image.open(image_path).convert("RGB")
        image = self.transform_img(image)

        if self.split == DatasetSplit.TEST and mask_path is not None:
            mask = PIL.Image.open(mask_path).convert("L")
            mask = (self.transform_mask(mask) > 0).float()
        else:
            mask = torch.zeros([1, *image.size()[1:]])

        return {
            "image": image,
            "mask": mask,
            "classname": classname,
            "anomaly": anomaly,
            "is_anomaly": int(anomaly != "good"),
            "image_name": "/".join(Path(image_path).parts[-4:]),
            "image_path": str(image_path),
        }

    def __len__(self):
        return len(self.data_to_iterate)

    def _read_rows(self):
        with self.split_csv.open("r", encoding="utf-8-sig", newline="") as file:
            rows = list(csv.DictReader(file))
        required = {"object", "split", "label", "image", "mask"}
        missing = required - set(rows[0].keys() if rows else [])
        if missing:
            raise ValueError("VisA CSV is missing columns: {}".format(sorted(missing)))
        return rows

    def get_image_data(self):
        rows = [row for row in self._read_rows() if row["object"] == self.classname]
        if not rows:
            raise ValueError("No rows for VisA class {!r}".format(self.classname))

        train_normal = [
            row for row in rows if row["split"] == "train" and row["label"] == "normal"
        ]
        if self.split == DatasetSplit.TRAIN:
            if self.train_val_split < 1.0:
                end = int(len(train_normal) * self.train_val_split)
                selected = train_normal[:end]
            else:
                selected = train_normal
        elif self.split == DatasetSplit.VAL:
            start = int(len(train_normal) * self.train_val_split)
            selected = train_normal[start:]
        elif self.split == DatasetSplit.TEST:
            selected = [row for row in rows if row["split"] == "test"]
        else:
            raise ValueError("Unsupported VisA split: {}".format(self.split))

        if not selected:
            raise ValueError(
                "No VisA rows selected for class={!r}, split={!r}".format(
                    self.classname, self.split
                )
            )

        data_to_iterate = []
        imgpaths_per_class = {self.classname: {}}
        for row in selected:
            image_path = self.source / row["image"]
            if not image_path.exists():
                raise FileNotFoundError("VisA image not found: {}".format(image_path))

            is_anomaly = row["label"] == "anomaly"
            # The existing PatchCore metric code treats "good" as normal.
            anomaly = "anomaly" if is_anomaly else "good"
            mask_path = None
            if self.split == DatasetSplit.TEST and is_anomaly:
                raw_mask = (row.get("mask") or "").strip()
                if not raw_mask:
                    raise ValueError("Missing mask path for VisA anomaly: {}".format(image_path))
                mask_path = self.source / raw_mask
                if not mask_path.exists():
                    raise FileNotFoundError("VisA mask not found: {}".format(mask_path))

            imgpaths_per_class[self.classname].setdefault(anomaly, []).append(
                str(image_path)
            )
            data_to_iterate.append(
                [self.classname, anomaly, str(image_path), str(mask_path) if mask_path else None]
            )

        return imgpaths_per_class, data_to_iterate
