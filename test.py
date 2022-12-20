#!/usr/bin/python
# -*- encoding: utf-8 -*-
from typing import Any, List

import os
import argparse

import numpy as np

import cv2
from PIL import Image

import torch
import torchvision.transforms as transforms

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


def mask_image_with_maps(image: Image, anno: np.ndarray, stride: int) -> List[Any]:
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
            masked_images.append(masked_image)

    return masked_images


def mask_image_face(image: Image, anno: np.ndarray, stride: int) -> Any:
    image = np.array(image).copy().astype(np.uint8)
    anno = anno.copy().astype(np.uint8)
    anno = cv2.resize(
        anno,
        None,
        fx=stride,
        fy=stride,
        interpolation=cv2.INTER_NEAREST,
    )

    face_classes = [1, 2, 3, 4, 4, 5, 6, 10, 11, 12, 13]
    mask = [anno[:, :, np.newaxis] == i for i in face_classes]
    mask = np.logical_or.reduce(mask)
    mask = np.invert(mask)
    masked_image = np.where(mask, np.ones_like(image), image)
    masked_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)

    return masked_image


def vis_parsing_maps(
    image: Image,
    anno: np.ndarray,
    stride: int,
    save_image: bool = False,
    save_path: str = "vis_results/parsing_map_on_image.jpg",
) -> None:
    overlayed_on_image = overlay_maps_on_image(image, anno, stride)
    masked_images = mask_image_with_maps(image, anno, stride)
    face_image = mask_image_face(image, anno, stride)

    # Save result or not
    if save_image:
        path, filename = os.path.split(save_path)
        filename, _ = os.path.splitext(filename)

        cv2.imagewrite(save_path, overlayed_on_image, [int(cv2.imageWRITE_JPEG_QUALITY), 100])
        
        for image in masked_images:
            image_path = os.path.join(path, f"{filename}-mask.jpg")
            cv2.imagewrite(image_path, image, [int(cv2.imageWRITE_JPEG_QUALITY), 100])
        
        face_path = os.path.join(path, f"{filename}-face.jpg")
        cv2.imagewrite(face_path, face_image, [int(cv2.imageWRITE_JPEG_QUALITY), 100])


def eval_transform() -> transforms.Compose():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )


def load_image(path: str) -> Image:
    image = Image.open(path)
    image = image.resize((512, 512), image.BILINEAR)
    return image


def evaluate(
    output_dir: str = "./res/test_res",
    input_dir: str = "./data",
    checkpoint: str = "model_final_diss.pth",
) -> None:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    net.load_state_dict(torch.load(checkpoint))
    net.eval()

    transform = eval_transform()

    with torch.no_grad():
        for f in os.listdir(input_dir):
            path = os.path.join(input_dir, f)
            image = load_image(path)
            image = transform(image)
            image = torch.unsqueeze(image, 0)
            image = image.cuda()
            out = net(image)[0]
            anno = out.squeeze(0).cpu().numpy().argmax(0)

            vis_parsing_maps(
                image,
                anno,
                stride=1,
                save_image=True,
                save_path=os.path.join(output_dir, path),
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        default="test-img/",
        type=str,
        help="The path to a directory with test imageages.",
    )
    parser.add_argument(
        "--output_dir",
        default="res/test_res/",
        type=str,
        help="The output directory for generated imageages.",
    )
    parser.add_argument(
        "--checkpoint",
        default="res/cp/79999_iter.pth",
        type=str,
        help="The path to a model checkpoint",
    )

    args = parser.parse_args()

    evaluate(**vars(args))
