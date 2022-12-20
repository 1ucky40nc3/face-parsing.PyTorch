#!/usr/bin/python
# -*- encoding: utf-8 -*-
from typing import Any, List, Optional

import os
import argparse

import numpy as np

import cv2
from PIL import Image

import torch
import torchvision.transforms as T

from model import BiSeNet


# Colors for all 20 parts
PART_COLORS = [
    [255, 0, 0],  # skin
    [255, 85, 0],  # left brow
    [255, 170, 0],  # right brow
    [255, 0, 85],  # left eye
    [255, 0, 170],  # right eye
    [0, 255, 0],  # glasses
    [85, 255, 0],  # left ear
    [170, 255, 0],  # right ear
    [0, 255, 85],  # ear rings
    [0, 255, 170],  # nose
    [0, 0, 255],  # mouth
    [85, 0, 255],  # upper lip
    [170, 0, 255],  # lower lip
    [0, 85, 255],  # neck
    [0, 170, 255],  # cloth
    [255, 255, 0],  # hair
    [255, 255, 85],  # hat
    [255, 255, 170],
    [255, 0, 255],
    [255, 85, 255],
    [255, 170, 255],
    [0, 255, 255],
    [85, 255, 255],
    [170, 255, 255],
]


FACE_CLASSES = [1, 2, 3, 4, 4, 5, 6, 10, 11, 12, 13]


def save_image(path: str, image: Any) -> None:
    cv2.imwrite(path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


def overlay_maps_on_image(image: Image, anno: np.ndarray, stride: int) -> Any:
    image = np.array(image).copy().astype(np.uint8)
    anno = anno.copy().astype(np.uint8)
    anno = cv2.resize(
        anno,
        None,
        fx=stride,
        fy=stride,
        interpolation=cv2.INTER_NEAREST,
    )
    anno_color = (
        np.zeros((anno.shape[0], anno.shape[1], 3))
        + 255
    )

    n_classes = np.max(anno)

    for pi in range(1, n_classes + 1):
        index = np.where(anno == pi)
        anno_color[index[0], index[1], :] = PART_COLORS[pi]

    anno_color = anno_color.astype(np.uint8)
    image = cv2.addWeighted(
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR), 0.4, anno_color, 0.6, 0
    )

    return image


def mask_image_with_maps(image: Image, anno: np.ndarray, stride: int = 1) -> List[Any]:
    image = np.array(image).copy().astype(np.uint8)
    anno = anno.copy().astype(np.uint8)
    anno = cv2.resize(
        anno,
        None,
        fx=stride,
        fy=stride,
        interpolation=cv2.INTER_NEAREST,
    )
    masked_images = []

    n_classes = np.max(anno)

    for i in range(1, n_classes + 1):
        mask = anno[:, :, np.newaxis] != i
        masked_image = np.where(mask, np.ones_like(image), image)
        if masked_image.shape[0] > 0:
            masked_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)
            masked_images.append((i, masked_image))

    return masked_images


def mask_image_custom(image: Image, anno: np.ndarray, classes: List[int], stride: int = 1) -> Any:
    image = np.array(image).copy().astype(np.uint8)
    anno = anno.copy().astype(np.uint8)
    anno = cv2.resize(
        anno,
        None,
        fx=stride,
        fy=stride,
        interpolation=cv2.INTER_NEAREST,
    )

    mask = [anno[:, :, np.newaxis] == i for i in classes]
    mask = np.logical_or.reduce(mask)
    mask = np.invert(mask)
    masked_image = np.where(mask, np.ones_like(image), image)
    masked_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)

    return masked_image


def visualize_anno(
    image: Image,
    anno: np.ndarray,
    output_dir: str,
    filename: str,
    stride: int = 1,
    do_overlay_maps_on_image: bool = False,
    do_mask_image_with_maps: bool = False,
    do_mask_image_face: bool = False,
    do_mask_image_custom: bool = False,
    map_ids: Optional[List[int]] = None
) -> None:
    filename, _ = os.path.splitext(filename)

    if do_overlay_maps_on_image:
        overlayed_on_image = overlay_maps_on_image(image, anno, stride)
        save_path = os.path.join(output_dir, f"{filename}.jpg")
        save_image(save_path, overlayed_on_image)
    
    if do_mask_image_with_maps:
        masked_images = mask_image_with_maps(image, anno, stride)
        for i, img in masked_images:
            img_path = os.path.join(output_dir, f"{filename}-mask-{i}.jpg")
            save_image(img_path, img)
    
    if do_mask_image_face:
        face_image = mask_image_custom(image, anno, FACE_CLASSES, stride)
        face_path = os.path.join(output_dir, f"{filename}-face.jpg")
        save_image(face_path, face_image)
    
    if do_mask_image_custom:
        face_image = mask_image_custom(image, anno, map_ids, stride)
        face_path = os.path.join(output_dir, f"{filename}-custom.jpg")
        save_image(face_path, face_image)


def eval_transforms() -> T.Compose:
    return T.Compose(
        [
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )


def load_image(path: str) -> Image:
    image = Image.open(path)
    image = image.resize((512, 512), Image.BILINEAR)
    return image


def evaluate(
    output_dir: str = "./res/test_res",
    input_dir: str = "./data",
    checkpoint: str = "model_final_diss.pth",
    do_overlay_maps_on_image: bool = False,
    do_mask_image_with_maps: bool = False,
    do_mask_image_face: bool = False,
    do_mask_image_custom: bool = False,
    map_ids: Optional[List[int]] = None
) -> None:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    net.load_state_dict(torch.load(checkpoint))
    net.eval()

    transform = eval_transforms()

    with torch.no_grad():
        for f in os.listdir(input_dir):
            path = os.path.join(input_dir, f)
            image = load_image(path)
            x = transform(image)
            x = torch.unsqueeze(x, 0)
            x = x.cuda()
            y = net(x)[0]
            anno = y.squeeze(0).cpu().numpy().argmax(0)

            visualize_anno(
                image,
                anno,
                output_dir=output_dir,
                filename=f,
                do_overlay_maps_on_image=do_overlay_maps_on_image,
                do_mask_image_with_maps=do_mask_image_with_maps,
                do_mask_image_face=do_mask_image_face,
                do_mask_image_custom=do_mask_image_custom,
                map_ids=map_ids
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        default="test-img/",
        type=str,
        help="The path to a directory with test images.",
    )
    parser.add_argument(
        "--output_dir",
        default="res/test_res/",
        type=str,
        help="The output directory for generated images.",
    )
    parser.add_argument(
        "--checkpoint",
        default="res/cp/79999_iter.pth",
        type=str,
        help="The path to a model checkpoint",
    )
    parser.add_argument(
        "--do_overlay_maps_on_image",
        action="store_true",
        default=False,
        help="Wether to output the input image overlayed with all found maps."
    )
    parser.add_argument(
        "--do_mask_image_with_maps",
        action="store_true",
        default=False,
        help="Wether to output all versions of the masked input image with predicted maps."
    )
    parser.add_argument(
        "--do_mask_image_face",
        action="store_true",
        default=False,
        help="Wether to output the input image masked with all found maps part of a face."
    )
    parser.add_argument(
        "--do_mask_image_custom",
        action="store_true",
        default=False,
        help="Wether to output the input image masked with all found and specified maps."
    )
    parser.add_argument(
        "--map_ids",
        nargs="+",
        default=None,
        type=int,
        help="The ids of maps for custom image masking."
    )
    

    args = parser.parse_args()

    evaluate(**vars(args))
