import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy
import pandas
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

RESIZE = 224

def create_train_transform(height: int = 224, width: int = 224):
    return A.Compose(
        [
            A.Resize(height, width),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=25, p=.5),
            A.Downscale(scale_min=0.25, scale_max=0.25, p=0.5),
            A.Blur(blur_limit=7, p=0.5),
            ToTensorV2(),
        ],
        additional_targets={
            "image": "image",
            "mask": "image",
            "trimap": "image",
            "fg": "image",
            "bg": "image",
        },
    )

def create_test_transform(height: int = 224, width: int = 224):
    return A.Compose(
        [
            A.Resize(height, width),
            ToTensorV2(),
        ],
        additional_targets={
            "image": "image",
            "mask": "image",
            "trimap": "image",
            "fg": "image",
            "bg": "image",
        },
    )


def gen_trimap_with_dilate(alpha, kernel_size):
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, 
        (
            kernel_size, kernel_size,
        )
    )
    fg_and_unknown = numpy.array(numpy.not_equal(alpha, 0).astype(numpy.float32))
    fg = numpy.array(numpy.equal(alpha, 255).astype(numpy.float32))
    dilate = cv2.dilate(fg_and_unknown, kernel, iterations=1)
    erode = cv2.erode(fg, kernel, iterations=1)
    trimap = erode * 255 + (dilate-erode)*128
    return trimap.astype(numpy.uint8)


class MattingDataset(Dataset):
    def __init__(self, annotations_df: pandas.DataFrame, train: bool = True, transform : A.Compose = None) -> None:
        super(MattingDataset, self).__init__()

        self.annotations_df = annotations_df
        self.train = train
        self.transform = transform
        if transform == None:
            if self.train:
                self.transform = create_train_transform()
            else:
                self.transform = create_test_transform()
        

    def __len__(self):
        return len(self.annotations_df)

    def __getitem__(self, index):
        image_path = self.annotations_df.iloc[index, 0]
        mask_path = self.annotations_df.iloc[index, 1]
        trimap_path = self.annotations_df.iloc[index, 2]

        image_b = Image.open(image_path)
        mask_b = Image.open(mask_path).convert("L")
        trimap_b = Image.open(trimap_path).convert("L")

        w, h = image_b.size

        max_crop_size = max(min(w, h), RESIZE)

        image = numpy.array(image_b)
        mask = numpy.array(mask_b)
        trimap = numpy.array(trimap_b)

        fg = image*(mask.copy()[:, :, numpy.newaxis])
        bg = image*(1-(mask.copy()[:, :, numpy.newaxis]))

        # transformed = self.transform(image=image, mask=mask, trimap=trimap, fg=fg, bg=bg)
        # image = transformed["image"]
        # mask = transformed["mask"]
        # trimap = transformed["trimap"]
        # fg = transformed["fg"]
        # bg = transformed["bg"]

        resizeTransform = A.Compose(
        [
            A.CenterCrop(height=max_crop_size, width=max_crop_size),
            A.Resize(height=RESIZE,width=RESIZE),
            A.HorizontalFlip(p=0.5),
            # A.Rotate(limit=25, p=.5),
            A.Blur(blur_limit=7, p=0.5),
            ToTensorV2(),
        ],
        additional_targets={
            "image": "image",
            "mask": "image",
            "trimap": "image",
            "fg": "image",
            "bg": "image",
        },
        )

        transformed = resizeTransform(image=image, mask=mask, trimap=trimap, fg=fg, bg=bg)
        image = transformed["image"]
        mask = transformed["mask"]
        trimap = transformed["trimap"]
        fg = transformed["fg"]
        bg = transformed["bg"]

        return image, mask, trimap, fg, bg
