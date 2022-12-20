#!/usr/bin/python
# -*- encoding: utf-8 -*-
from typing import Any

import os
import os.path as osp
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


def overlay_maps_on_im(im: Image, anno: np.ndarray, stride: int) -> Any:
    im = np.array(im).copy().astype(np.uint8)
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
    im = cv2.addWeighted(
        cv2.cvtColor(im, cv2.COLOR_RGB2BGR), 0.4, anno_color, 0.6, 0
    )

    return im


def vis_parsing_maps(
    im: Image,
    parsing_anno: np.ndarray,
    stride: int,
    save_im: bool = False,
    save_path: str = "vis_results/parsing_map_on_im.jpg",
) -> None:
    overlayed_on_im = overlay_maps_on_im(im, parsing_anno, stride)

    # Save result or not
    if save_im:
        cv2.imwrite(save_path, overlayed_on_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


def evaluate(
    respth: str = "./res/test_res",
    dspth: str = "./data",
    cp: str = "model_final_diss.pth",
):
    if not os.path.exists(respth):
        os.makedirs(respth)

    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    save_pth = osp.join("res/cp", cp)
    net.load_state_dict(torch.load(save_pth))
    net.eval()

    to_tensor = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    with torch.no_grad():
        for image_path in os.listdir(dspth):
            img = Image.open(osp.join(dspth, image_path))
            image = img.resize((512, 512), Image.BILINEAR)
            img = to_tensor(image)
            img = torch.unsqueeze(img, 0)
            img = img.cuda()
            out = net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
            # print(parsing)
            print(np.unique(parsing))

            vis_parsing_maps(
                image,
                parsing,
                stride=1,
                save_im=True,
                save_path=osp.join(respth, image_path),
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dspth",
        default="test-img/",
        type=str,
        help="The path to a directory with test images.",
    )
    parser.add_argument(
        "--respth",
        default="res/test_res/",
        type=str,
        help="The output directory for generated images.",
    )
    parser.add_argument(
        "--cp",
        default="79999_iter.pth",
        type=str,
        help="The path to a model checkpoint",
    )

    args = parser.parse_args()

    evaluate(**vars(args))
