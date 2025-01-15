import torch
import cv2
from PIL import Image
from torchvision import transforms

from trashcan_core.components.constants import CATEGORIES


class MaskGenerator:
    def __init__(self) -> None:
        pass

    def __call__(self, im: Image, im_annots: list[dict[str, any]]):
        return self._create_masks(im=im, im_annots=im_annots)

    def _create_masks(self, im: Image, im_annots: list[dict[str, any]]):
        transform = transforms.ToTensor()
        im_tensor = transform(im)

        channels = []
        labels = [x["label"] for x in im_annots]
        polys = [
            torch.tensor(x["points"][0], dtype=torch.int32).view(-1, 2)
            for x in im_annots
        ]

        label2poly = dict(zip(labels, polys))
        background = torch.zeros(
            (im_tensor.shape[1], im_tensor.shape[2]), dtype=torch.float32
        )

        for label in CATEGORIES:
            blank = torch.zeros(
                (im_tensor.shape[1], im_tensor.shape[2]), dtype=torch.float32
            )

            if label in labels:
                poly = label2poly[label]
                cv2.fillPoly(blank.numpy(), [poly.numpy()], 255)
                cv2.fillPoly(background.numpy(), [poly.numpy()], 255)

            channels.append(blank)

        if channels:
            _, background = cv2.threshold(
                background.numpy(), 127, 255, cv2.THRESH_BINARY_INV
            )
            background = torch.tensor(background, dtype=torch.float32)
            channels.append(background)

        Y = torch.stack(channels, dim=0) / 255.0

        return Y


class BinaryMaskGenerator(MaskGenerator):
    def __init__(self):
        super().__init__()

    def _create_masks(self, im: Image, im_annots: list[dict[str, any]]):
        transform = transforms.ToTensor()
        im_tensor = transform(im)

        mask = torch.zeros(
            (im_tensor.shape[1], im_tensor.shape[2]), dtype=torch.float32
        )

        for annot in im_annots:
            poly = torch.tensor(annot["points"][0], dtype=torch.int32).view(-1, 2)
            cv2.fillPoly(mask.numpy(), [poly.numpy()], 1)

        mask = torch.clamp(mask, 0, 1)

        mask = mask.unsqueeze(0)

        return mask
