import json
from os import path as osp
import sys
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils import data
from torchvision import transforms
import tqdm
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--data_txt", type=str, required=True)
parser.add_argument("--load_height", type=int, default=512)
parser.add_argument("--load_width", type=int, default=384)
parser.add_argument("--cloth_type", default=str, required=True)

opt = parser.parse_args()

load_width = opt.load_width
load_height = opt.load_height
data_path = opt.data_path
cloth_type = opt.cloth_type


def get_img_agnostic(img, label, parse, pose_data):
    label_array = np.array(label)
    parse_array = np.array(parse)

    parse_head = (
        (label_array == 1).astype(np.float32)
        + (label_array == 2).astype(np.float32)
        # + (label_array == 11).astype(np.float32)
        + (label_array == 3).astype(np.float32)
    )
    parse_upper = (
        (label_array == 4).astype(np.float32)
        + (label_array == 14).astype(np.float32)
        + (label_array == 15).astype(np.float32)
        - (parse_array == 3).astype(np.float32)
        - (parse_array == 4).astype(np.float32)
    )
    parse_lower = (
        (label_array == 5).astype(np.float32)
        + (label_array == 6).astype(np.float32)
        + (label_array == 7).astype(np.float32)
        + (label_array == 8).astype(np.float32)
        + (label_array == 12).astype(np.float32)
        + (label_array == 13).astype(np.float32)
    )

    parse_hand = (parse_array == 3).astype(np.float32) + (parse_array == 4).astype(
        np.float32
    )

    # masking

    r = 10
    img = np.array(img)
    if cloth_type == "upper_body" or cloth_type == "dresses":
        img[parse_upper > 0, :] = 0

    if cloth_type == "lower_body" or cloth_type == "dresses":
        img[parse_lower > 0, :] = 0

    img = Image.fromarray(img)

    agnostic = img.copy()
    agnostic_draw = ImageDraw.Draw(agnostic)

    # recalculate keypoint
    length_a = np.linalg.norm(pose_data[5] - pose_data[2] + 1e-8)
    length_b = np.linalg.norm(pose_data[11] - pose_data[8] + 1e-8)
    point = (pose_data[8] + pose_data[11]) / 2
    pose_data[8] = point + (pose_data[8] - point) / length_b * length_a
    pose_data[11] = point + (pose_data[11] - point) / length_b * length_a

    if cloth_type == "upper_body" or cloth_type == "dresses":
        # mask arms
        agnostic_draw.line([tuple(pose_data[i]) for i in [2, 5]], "black", width=r * 7)

        for i in [2, 5]:
            pointx, pointy = pose_data[i]
            agnostic_draw.ellipse(
                (pointx - r * 4, pointy - r * 4, pointx + r * 5, pointy + r * 4),
                "black",
                "black",
            )

        for i in [3, 4, 6, 7]:
            if (pose_data[i - 1, 0] == 0.0 and pose_data[i - 1, 1] == 0.0) or (
                pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0
            ):
                continue
            agnostic_draw.line(
                [tuple(pose_data[j]) for j in [i - 1, i]], "black", width=r * 10
            )
            pointx, pointy = pose_data[i]
            if i in [4, 7]:
                pass  # agnostic_draw.ellipse((pointx-r, pointy-r, pointx+r, pointy+r), 'black', 'black')
            else:
                agnostic_draw.ellipse(
                    (pointx - r * 4, pointy - r * 4, pointx + r * 4, pointy + r * 4),
                    "black",
                    "black",
                )

        # mask torso
        for i in [8, 11]:
            pointx, pointy = pose_data[i]
            if pointx < 1 and pointy < 1:
                continue
            agnostic_draw.ellipse(
                (pointx - r * 3, pointy - r * 6, pointx + r * 3, pointy + r * 6),
                "black",
                "black",
            )
        line_points = []
        for i in [2, 8]:
            if pose_data[i][0] < 1 and pose_data[i][1] < 1:
                continue
            line_points.append(tuple(pose_data[i]))
        agnostic_draw.line(line_points, "black", width=r * 6)
        line_points = []
        for i in [5, 11]:
            if pose_data[i][0] < 1 and pose_data[i][1] < 1:
                continue
            line_points.append(tuple(pose_data[i]))
        agnostic_draw.line(line_points, "black", width=r * 6)
        line_points = []
        for i in [8, 11]:
            if pose_data[i][0] < 1 and pose_data[i][1] < 1:
                continue
            line_points.append(tuple(pose_data[i]))
        agnostic_draw.line(line_points, "black", width=r * 12)
        line_points = []
        for i in [2, 5, 11, 8]:
            if pose_data[i][0] < 1 and pose_data[i][1] < 1:
                continue
            line_points.append(tuple(pose_data[i]))
        if len(line_points) > 1 and len(line_points[0]) > 1:
            agnostic_draw.polygon(line_points, "black", "black")

        # mask neck
        pointx, pointy = pose_data[1]
        agnostic_draw.rectangle(
            (pointx - r * 4, pointy - r * 4, pointx + r * 4, pointy + r * 4),
            "black",
            "black",
        )

    if cloth_type == "lower_body" or cloth_type == "dresses":
        if cloth_type == "dresses":
            r = r * 2

        # mask legs
        agnostic_draw.line([tuple(pose_data[i]) for i in [8, 11]], "black", width=r * 5)

        for i in [9, 12]:
            # for i in [8, 9, 11, 12]:
            pointx, pointy = pose_data[i]
            agnostic_draw.ellipse(
                (pointx - r * 2, pointy - r * 2, pointx + r * 2, pointy + r * 2),
                "black",
                "black",
            )
        for i in [9, 10, 12, 13]:
            if (pose_data[i - 1, 0] == 0.0 and pose_data[i - 1, 1] == 0.0) or (
                pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0
            ):
                continue
            agnostic_draw.line(
                [tuple(pose_data[j]) for j in [i - 1, i]], "black", width=r * 5
            )

            pointx, pointy = pose_data[i]
            if i in [10, 13]:
                pass  # agnostic_draw.ellipse((pointx-r, pointy-r, pointx+r, pointy+r), 'black', 'black')
            else:
                agnostic_draw.ellipse(
                    (pointx - r * 2, pointy - r * 2, pointx + r * 2, pointy + r * 2),
                    "black",
                    "black",
                )

    if cloth_type == "upper_body":
        agnostic.paste(img, None, Image.fromarray(np.uint8(parse_head * 255), "L"))
        agnostic.paste(img, None, Image.fromarray(np.uint8(parse_lower * 255), "L"))
        agnostic.paste(img, None, Image.fromarray(np.uint8(parse_hand * 255), "L"))
    elif cloth_type == "lower_body":
        agnostic.paste(img, None, Image.fromarray(np.uint8(parse_head * 255), "L"))
        agnostic.paste(img, None, Image.fromarray(np.uint8(parse_upper * 255), "L"))
        agnostic.paste(img, None, Image.fromarray(np.uint8(parse_hand * 255), "L"))
    elif cloth_type == "dresses":
        agnostic.paste(img, None, Image.fromarray(np.uint8(parse_head * 255), "L"))
        agnostic.paste(img, None, Image.fromarray(np.uint8(parse_hand * 255), "L"))

    return agnostic


def process(img_name, save_dirname="img_agnostic"):
    index = img_name.split("_")[0]

    label = Image.open(f"{data_path}/{cloth_type}/label_maps/{index}_4.png")
    label = transforms.Resize(load_width, interpolation=0)(label)

    parse = Image.open(f"{data_path}/{cloth_type}/dense/{index}_5.png")
    parse = transforms.Resize(load_width, interpolation=0)(parse)

    with open(f"{data_path}/{cloth_type}/keypoints/{index}_2.json", "r") as f:
        pose_label = json.load(f)
        pose_data = pose_label["keypoints"]
        pose_data = np.array(pose_data)
        pose_data = pose_data[:, :2]

    img = Image.open(f"{data_path}/{cloth_type}/images/{index}_0.jpg")
    img = transforms.Resize(load_width, interpolation=0)(img)
    img = get_img_agnostic(img, label, parse, pose_data)
    img = img.convert("RGB")
    img.save(f"{data_path}/{cloth_type}/{save_dirname}/{index}" + "_agnostic.png")


def main():
    img_names = []
    os.makedirs(f"{data_path}/{cloth_type}/img_agnostic", exist_ok=True)
    with open(opt.data_txt, "r") as f:
        for line in tqdm.tqdm(f.readlines()):
            img_name = osp.splitext(line.split()[0])[0]
            img_names.append(img_name)
            process(img_name)


if __name__ == "__main__":
    main()
