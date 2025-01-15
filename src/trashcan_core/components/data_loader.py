import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import json
from typing import Literal, Callable, Tuple

from trashcan_core.components.mask_generator import MaskGenerator, BinaryMaskGenerator


class SegmentationDataset(Dataset):
    def __init__(
        self,
        image_paths: str,
        set: Literal["train", "val"],
        mask_generator: MaskGenerator,
        transform: Callable[
            [torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]
        ] = None,
    ):
        self.image_paths = image_paths
        self.set = set
        self.mask_generator = mask_generator
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.transform = transform

        self.images_annots = self._get_images_annots()

    def _get_images_annots(self):
        annots_path = f"{self.image_paths}/instances_{self.set}_trashcan.json"

        with open(annots_path, "r") as f:
            coco_data = json.load(f)

        annots = coco_data["annotations"]
        ims = coco_data["images"]

        annots_dict = {}
        k = 1 if self.set == "train" else 6000

        for annot in annots:
            im_id = annot["image_id"]
            im_name = ims[im_id - k]["file_name"]

            if im_name in annots_dict:
                annots_dict[im_name].append(
                    {
                        "label": coco_data["categories"][annot["category_id"] - 1][
                            "name"
                        ],
                        "points": annot["segmentation"],
                    }
                )
            else:
                annots_dict[im_name] = [
                    {
                        "label": coco_data["categories"][annot["category_id"] - 1][
                            "name"
                        ],
                        "points": annot["segmentation"],
                    }
                ]

            if self.set == "val" and im_id - k >= 1203:
                break

        return annots_dict

    def __len__(self):
        return len(self.images_annots)

    def __getitem__(self, idx):
        ims_path = f"{self.image_paths}/{self.set}"
        im_name, im_annots = list(self.images_annots.items())[idx]

        im_path = f"{ims_path}/{im_name}"
        im = Image.open(im_path).convert("RGB")

        mask = self.mask_generator(im=im, im_annots=im_annots)

        image_tensor = (
            torch.tensor(np.array(im), dtype=torch.float32).permute(2, 0, 1) / 255.0
        )

        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)

        if self.transform:
            image_tensor, mask = self.transform(image_tensor, mask)

        return image_tensor.to(self.device), mask.to(self.device)


class CustomDataLoader(DataLoader):
    def __init__(
        self,
        image_paths: str,
        batch_size: int = 32,
        shuffle: bool = True,
    ):
        self.image_paths = image_paths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.mask_generator = MaskGenerator()
        self.crop_size = None

        self.train_dataset = SegmentationDataset(
            image_paths, set="train", mask_generator=self.mask_generator
        )
        self.val_dataset = SegmentationDataset(
            image_paths, set="val", mask_generator=self.mask_generator
        )

        super().__init__(
            dataset=self.train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.collate_fn,
        )

    def collate_fn(self, batch):
        images, masks = zip(*batch)
        return torch.stack(images), torch.stack(masks)

    def set_transform(self, transform: Callable):
        self.train_dataset.transform = transform
        self.val_dataset.transform = transform

    def crop(self, height: int, width: int) -> None:
        def crop_transform(image: torch.Tensor, mask: torch.Tensor):
            original_height, original_width = image.shape[-2], image.shape[-1]

            if height > original_height or width > original_width:
                raise ValueError(
                    f"Target dimensions ({height}, {width}) must not exceed the original dimensions ({original_height}, {original_width})."
                )

            top = (original_height - height) // 2
            left = (original_width - width) // 2
            bottom = top + height
            right = left + width

            cropped_image = image[:, top:bottom, left:right]
            cropped_mask = mask[:, top:bottom, left:right]
            return cropped_image, cropped_mask

        self.set_transform(crop_transform)
        print(f"Set lazy cropping to size ({height}, {width}).")

    def get_train_loader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            collate_fn=self.collate_fn,
        )

    def get_val_loader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
        )


class BinaryCustomDataLoader(CustomDataLoader):
    def __init__(self, image_paths: str, batch_size: int = 32, shuffle: bool = True):
        super().__init__(image_paths, batch_size, shuffle)
        self.mask_generator = BinaryMaskGenerator()
        self.train_dataset = SegmentationDataset(
            image_paths, set="train", mask_generator=self.mask_generator
        )
        self.val_dataset = SegmentationDataset(
            image_paths, set="val", mask_generator=self.mask_generator
        )


if __name__ == "__main__":
    from trashcan_core.components.constants import MATERIAL_VERSION_PATH

    batch_size = 16

    data_loader = CustomDataLoader(
        image_paths=MATERIAL_VERSION_PATH, batch_size=batch_size, shuffle=True
    )

    train_loader = data_loader.get_train_loader()
    val_loader = data_loader.get_val_loader()

    for batch_idx, (images, masks) in enumerate(train_loader):
        print(f"Batch {batch_idx + 1}:")
        print(f"Images shape: {images.shape}")
        print(f"Masks shape: {masks.shape}")

    for batch_idx, (images, masks) in enumerate(val_loader):
        print(f"Validation Batch {batch_idx + 1}:")
        print(f"Images shape: {images.shape}")
        print(f"Masks shape: {masks.shape}")
