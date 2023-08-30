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
parser.add_argument("--load_height", type=int, default=256)
parser.add_argument("--load_width", type=int, default=192)
parser.add_argument("--onlypose", action="store_true")
parser.add_argument("--cloth_type", default=str, required=True)

opt = parser.parse_args()

load_width = opt.load_width
load_height = opt.load_height
data_path = opt.data_path
cloth_type = opt.cloth_type

label_to_rgb = {
    0: (0, 0, 0),
    1: (128, 128, 0),
    2: (55, 126, 34),
    3: (128, 0, 0),
    4: (0, 0, 123),
    5: (0, 128, 128),
    6: (55, 126, 127),
    7: (128, 128, 128),
    8: (64, 0, 0),
    9: (80, 127, 35),
    10: (176, 36, 24),
    11: (183, 131, 43),
    12: (176, 36, 125),
    13: (58, 6, 123),
    14: (183, 131, 129),
    15: (80, 127, 127),
    16: (24, 63, 12),
    17: (128, 64, 0),
}


def get_img_agnostic_onlypose(img, pose_data):
    for pair in [[3, 4], [6, 7]]:
        pointx, pointy = (
            pose_data[pair[1]] + (pose_data[pair[1]] - pose_data[pair[0]]) * 0.3
        )
        pointx, pointy = int(pointx), int(pointy)

    r = 10
    img = np.array(img)
    img = Image.fromarray(img)
    agnostic = img.copy()
    agnostic_draw = ImageDraw.Draw(agnostic)

    length_a = np.linalg.norm(pose_data[5] - pose_data[2] + 1e-8)
    length_b = np.linalg.norm(pose_data[11] - pose_data[8] + 1e-8)
    point = (pose_data[8] + pose_data[11]) / 2
    pose_data[8] = point + (pose_data[8] - point) / length_b * length_a
    pose_data[11] = point + (pose_data[11] - point) / length_b * length_a
    # mask arms
    agnostic_draw.line([tuple(pose_data[i]) for i in [2, 5]], "black", width=r * 3)

    for i in [2, 5]:
        pointx, pointy = pose_data[i]
        agnostic_draw.ellipse(
            (pointx - r * 3, pointy - r * 3, pointx + r * 3, pointy + r * 3),
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
    if cloth_type == "upper":
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
            #         print(pose_data[i])
            line_points.append(tuple(pose_data[i]))
        if len(line_points) > 1 and len(line_points[0]) > 1:
            agnostic_draw.polygon(line_points, "black", "black")

        # mask neck
        pointx, pointy = pose_data[1]
        agnostic_draw.rectangle(
            (pointx - r * 2, pointy - r * 2, pointx + r * 2, pointy + r * 2),
            "black",
            "black",
        )
    elif cloth_type == "bottom":
        pass
    elif cloth_type == "dress":
        pass
    else:
        ValueError("Incorrect cloth type")

    return agnostic


def get_img_agnostic(img, parse, pose_data):
    parse_array = np.array(parse)
    parse_array = np.array(parse)
    parse_head = (parse_array == 23).astype(np.float32) + (parse_array == 24).astype(
        np.float32
    )
    parse_lower = (
        (parse_array == 7).astype(np.float32)
        + (parse_array == 8).astype(np.float32)
        + (parse_array == 9).astype(np.float32)
        + (parse_array == 10).astype(np.float32)
        + (parse_array == 11).astype(np.float32)
        + (parse_array == 12).astype(np.float32)
        + (parse_array == 13).astype(np.float32)
        + (parse_array == 14).astype(np.float32)
    )

    parse_upper = (
        (parse_array == 1).astype(np.float32)
        + (parse_array == 2).astype(np.float32)
        + (parse_array == 15).astype(np.float32)
        + (parse_array == 16).astype(np.float32)
        + (parse_array == 17).astype(np.float32)
        + (parse_array == 18).astype(np.float32)
        + (parse_array == 19).astype(np.float32)
        + (parse_array == 20).astype(np.float32)
        + (parse_array == 21).astype(np.float32)
        + (parse_array == 22).astype(np.float32)
    )

    parse_hand = (parse_array == 3).astype(np.float32) + (parse_array == 4).astype(
        np.float32
    )

    # additional processing for hand -- ?
    parse_hand_tmp = parse_hand.copy()
    r = 0
    for pair in [[3, 4], [6, 7]]:
        pointx, pointy = (
            pose_data[pair[1]] + (pose_data[pair[1]] - pose_data[pair[0]]) * 0.3
        )
        pointx, pointy = int(pointx), int(pointy)
        if pointx > 0 and pointy > 0:
            parse_hand_tmp[pointy - r : pointy + r, pointx - r : pointx + r] = 0
    parse_hand[parse_hand_tmp > 0] = 0

    r = 10
    img = np.array(img)
    img[parse_upper > 0, :] = 0
    img = Image.fromarray(img)
    agnostic = img.copy()
    agnostic_draw = ImageDraw.Draw(agnostic)

    length_a = np.linalg.norm(pose_data[5] - pose_data[2] + 1e-8)
    length_b = np.linalg.norm(pose_data[11] - pose_data[8] + 1e-8)
    point = (pose_data[8] + pose_data[11]) / 2
    pose_data[8] = point + (pose_data[8] - point) / length_b * length_a
    pose_data[11] = point + (pose_data[11] - point) / length_b * length_a

    if cloth_type == "upper":
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
        agnostic.paste(img, None, Image.fromarray(np.uint8(parse_head * 255), "L"))
        agnostic.paste(img, None, Image.fromarray(np.uint8(parse_lower * 255), "L"))
        agnostic.paste(img, None, Image.fromarray(np.uint8(parse_hand * 255), "L"))
    elif cloth_type == "lower":
        # mask legs
        agnostic_draw.line([tuple(pose_data[i]) for i in [8, 11]], "black", width=r * 5)
        agnostic.show()

        for i in [9, 12]:
            # for i in [8, 9, 11, 12]:
            pointx, pointy = pose_data[i]
            agnostic_draw.ellipse(
                (pointx - r * 4, pointy - r * 4, pointx + r * 4, pointy + r * 4),
                "black",
                "black",
            )
        agnostic.show()
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

        # agnostic.paste(img, None, Image.fromarray(np.uint8(parse_head * 255), 'L'))
        agnostic.paste(img, None, Image.fromarray(np.uint8(parse_upper * 255), "L"))
        agnostic.paste(img, None, Image.fromarray(np.uint8(parse_hand * 255), "L"))

        agnostic.show()
    elif cloth_type == "dress":
        # upper + lower 같이 하고
        # agnostic.paste를 head, hand만 하면 dress?
        pass
    else:
        ValueError("Error: Incorrect value for cloth type")

    return agnostic


def process(img_name, save_dirname="img_agnostic"):
    parse_name = img_name.replace(".jpg", ".png")
    parse = Image.open(osp.join(data_path, "image-parse-new", parse_name))
    parse = transforms.Resize(load_width, interpolation=0)(parse)
    pose_name = img_name.replace(".jpg", "_keypoints.json")
    with open(osp.join(data_path, "pose", pose_name), "r") as f:
        pose_label = json.load(f)
        pose_data = pose_label["keypoints"]
        pose_data = np.array(pose_data)
        pose_data = pose_data.reshape((-1, 3))[:, :2]
    # load parsing image
    img = Image.open(osp.join(data_path, "image", img_name))
    img = get_img_agnostic(img, parse, pose_data)
    img = img.convert("RGB")
    img.save(osp.join(data_path, save_dirname, img_name))


# processing without parsing data.
def process_onlypose(img_name):
    parse_name = img_name.replace(".jpg", ".png")
    parse = Image.open(osp.join(data_path, "image-parse-new", parse_name))
    parse = transforms.Resize(load_width, interpolation=0)(parse)
    pose_name = img_name.replace(".jpg", "_keypoints.json")
    with open(osp.join(data_path, "pose", pose_name), "r") as f:
        pose_label = json.load(f)
        pose_data = pose_label["keypoints"]
        pose_data = np.array(pose_data)
        pose_data = pose_data.reshape((-1, 3))[:, :2]

    img = Image.open(osp.join(data_path, "image", img_name))
    img = get_img_agnostic_onlypose(img, pose_data)
    img = img.convert("RGB")
    img.save(osp.join(data_path, "img_agnostic_onlypose", img_name))

    # load person image


def main():
    img_names = []
    if opt.onlypose:
        os.makedirs(osp.join(data_path, "img_agnostic_onlypose"), exist_ok=True)
    else:
        os.makedirs(osp.join(data_path, "img_agnostic"), exist_ok=True)
    with open(opt.data_txt, "r") as f:
        for line in tqdm.tqdm(f.readlines()):
            img_name = line.strip()
            img_names.append(img_name)
            if opt.onlypose:
                process_onlypose(img_name)
            else:
                process(img_name)


if __name__ == "__main__":
    main()
