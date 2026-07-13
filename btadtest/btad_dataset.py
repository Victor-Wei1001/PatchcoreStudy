import os
from enum import Enum
from pathlib import Path

import PIL
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode


CLASSNAMES = ["01", "02", "03"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class BTADDataset(torch.utils.data.Dataset):
    """BTAD reader for BTech_Dataset_transformed."""

    def __init__(
        self,
        source,
        classname,
        resize=256,
        imagesize=224,
        split=DatasetSplit.TRAIN,
        train_val_split=1.0,
        **kwargs,
    ):
        super().__init__()
        self.source = Path(source)
        if (self.source / "BTech_Dataset_transformed").exists():
            self.source = self.source / "BTech_Dataset_transformed"
        self.classname = str(classname)
        self.split = split
        self.train_val_split = train_val_split
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

    @staticmethod
    def _sorted_images(path):
        return sorted(
            [item for item in path.iterdir() if item.is_file()],
            key=lambda item: item.stem,
        )

    @staticmethod
    def _find_mask(mask_dir, stem):
        matches = sorted(mask_dir.glob(stem + ".*"))
        return str(matches[0]) if matches else None

    def get_image_data(self):
        category_dir = self.source / self.classname
        if not category_dir.exists():
            raise FileNotFoundError("BTAD category not found: {}".format(category_dir))

        train_normal = self._sorted_images(category_dir / "train" / "ok")
        if self.split == DatasetSplit.TRAIN:
            end = int(len(train_normal) * self.train_val_split)
            selected = train_normal[:end]
            rows = [("good", path, None) for path in selected]
        elif self.split == DatasetSplit.VAL:
            start = int(len(train_normal) * self.train_val_split)
            selected = train_normal[start:]
            rows = [("good", path, None) for path in selected]
        elif self.split == DatasetSplit.TEST:
            normal = self._sorted_images(category_dir / "test" / "ok")
            anomaly = self._sorted_images(category_dir / "test" / "ko")
            mask_dir = category_dir / "ground_truth" / "ko"
            rows = [("good", path, None) for path in normal]
            for path in anomaly:
                mask_path = self._find_mask(mask_dir, path.stem)
                if mask_path is None:
                    raise FileNotFoundError(
                        "BTAD mask not found for anomaly image: {}".format(path)
                    )
                rows.append(("anomaly", path, mask_path))
        else:
            raise ValueError("Unsupported BTAD split: {}".format(self.split))

        if not rows:
            raise ValueError(
                "No BTAD images selected for category={} split={}".format(
                    self.classname, self.split.value
                )
            )

        data_to_iterate = [
            [self.classname, anomaly, str(image_path), mask_path]
            for anomaly, image_path, mask_path in rows
        ]
        imgpaths_per_class = {self.classname: {"good": [], "anomaly": []}}
        for classname, anomaly, image_path, mask_path in data_to_iterate:
            imgpaths_per_class[classname].setdefault(anomaly, []).append(image_path)
        return imgpaths_per_class, data_to_iterate
