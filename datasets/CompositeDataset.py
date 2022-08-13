import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy
import pandas
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

# def gen_trimap_with_dilate(alpha, kernel_size):
#     kernel = cv2.getStructuringElement(
#         cv2.MORPH_ELLIPSE, 
#         (
#             kernel_size, kernel_size,
#         )
#     )
#     fg_and_unknown = numpy.array(numpy.not_equal(alpha, 0).astype(numpy.float32))
#     fg = numpy.array(numpy.equal(alpha, 255).astype(numpy.float32))
#     dilate = cv2.dilate(fg_and_unknown, kernel, iterations=1)
#     erode = cv2.erode(fg, kernel, iterations=1)
#     trimap = erode * 255 + (dilate-erode)*128
#     return trimap.astype(numpy.uint8)


class CompositeDataset(Dataset):
    def __init__(self, annotations_df) -> None:
        super(CompositeDataset, self).__init__()
        self.annotations_df = annotations_df
        

    def __len__(self):
        return len(self.annotations_df)

    def __getitem__(self, index):
        image_path = self.annotations_df.iloc[index, 0]
        mask_path = self.annotations_df.iloc[index, 1]
        trimap_path = self.annotations_df.iloc[index, 2]
        bg_path = self.annotations_df.iloc[index, 3]

        image_b = Image.open(image_path)
        mask_b = Image.open(mask_path).convert("L")
        trimap_b = Image.open(trimap_path).convert("L")
        bg_b = Image.open(bg_path)

        w, h = image_b.size

        image = numpy.array(image_b)
        mask = numpy.array(mask_b)
        trimap = numpy.array(trimap_b)
        bg = numpy.array(bg_b)

        fg = image*(mask.copy()[:, :, numpy.newaxis])

        # bg = image*(1-(mask.copy()[:, :, numpy.newaxis]))

        transform = A.Compose(
        [
            A.CenterCrop(height=w, width=h),
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

        transformed = transform(image=image, mask=mask, trimap=trimap, fg=fg, bg=bg)
        image = transformed["image"]
        mask = transformed["mask"]
        trimap = transformed["trimap"]
        fg = transformed["fg"]
        bg = transformed["bg"]

        return image, mask, trimap, fg, bg
