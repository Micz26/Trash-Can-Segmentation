import torch
from PIL import Image
import numpy as np

import json
from typing import Literal

from trashcan_core.components.mask_generator import MaskGenerator


class DataLoader:
    def __init__(
        self,
        image_paths: str,
        batch_size: int = 32,
        shuffle: int = True,
    ) -> None:
        self.image_paths = image_paths
        self.batch_size = batch_size
        self.train_images_annots = self._get_images_annots(set="train")
        self.val_images_annots = self._get_images_annots(set="val")
        self.mask_generator = MaskGenerator()
        self.shuffle = shuffle
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.x_train, self.y_train, self.x_val, self.y_val = None, None, None, None

    def __len__(self) -> int:
        dataset_size = len(
            self.train_images_annots
            if self.x_train is not None
            else self.val_images_annots
        )
        return dataset_size // self.batch_size

    def __getitem__(self, index):
        if self.x_train is None or self.y_train is None:
            raise ValueError(
                "Dataset has not been generated. Call the DataLoader instance first."
            )

        start_idx = index * self.batch_size
        end_idx = start_idx + self.batch_size
        x_data = (
            self.x_train[start_idx:end_idx]
            if self.x_train is not None
            else self.x_val[start_idx:end_idx]
        )
        y_data = (
            self.y_train[start_idx:end_idx]
            if self.y_train is not None
            else self.y_val[start_idx:end_idx]
        )

        return x_data, y_data

    def __call__(self):
        self.x_train, self.y_train = map(list, zip(*self._generate_masks(set="train")))
        self.x_val, self.y_val = map(list, zip(*self._generate_masks(set="val")))

    def _get_images_annots(
        self, set: Literal["train", "val"]
    ) -> dict[str, list[dict[str, any]]]:
        annots_path = self.image_paths + "/instances_" + set + "_trashcan.json"

        with open(annots_path, "r") as f:
            coco_data = json.load(f)

        annots = coco_data["annotations"]
        ims = coco_data["images"]

        annots_dict = {}
        k = 1 if set == "train" else 6000

        for annot in annots:
            im_id = annot["image_id"]
            print(f"idx: {im_id - k}")
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
            if set == "val" and im_id - k >= 1203:
                break

        return annots_dict

    def _generate_masks(self, set: Literal["train", "val"]):
        ims_path = self.image_paths + "/" + set
        annots_dict = (
            self.train_images_annots if set == "train" else self.val_images_annots
        )

        images, masks = [], []

        l = 500 if set == "train" else 30

        for i, (im_name, im_annots) in enumerate(annots_dict.items()):
            im_path = ims_path + "/" + im_name

            im = Image.open(im_path).convert("RGB")

            if im.size[1] > 256:
                im = im.crop((0, 0, im.size[0], 256))

            mask = self.mask_generator(im=im, im_annots=im_annots)

            image_tensor = (
                torch.tensor(np.array(im), dtype=torch.float32).permute(2, 0, 1) / 255.0
            )

            if len(mask.shape) == 2:
                mask = mask.unsqueeze(0)

            images.append(image_tensor)
            masks.append(mask)

            if (i + 1) % self.batch_size == 0:
                yield (
                    torch.stack(images).to(self.device),
                    torch.stack(masks).to(self.device),
                )
                images, masks = [], []
            if i >= l:
                break

        if images and masks:
            yield (
                torch.stack(images).to(self.device),
                torch.stack(masks).to(self.device),
            )


if __name__ == "__main__":
    from trashcan_core.components.constants import MATERIAL_VERSION_PATH

    data_loader = DataLoader(MATERIAL_VERSION_PATH)
    data_loader()
