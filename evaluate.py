import argparse
import json
import os

import numpy as np
from shapely.geometry import Polygon


def evaluate(pred_json_path, annotation_json_path, logs):
    """
    Calculate the evaluation metric for the pipeline by matching the
    predicted segmentation and the ground truth polygons and thus
    comparing the predicted OCR texts and the gt texts.

    Args:
        pred_json_path (str): Path to json with predictions for each image.
            Should have image names from annotation_json_path as keys. 
            The json should have the following format:
            {
                "img_0.jpg": {
                        "predictions": [
                            {
                                 "polygon": list,  # the coordinates of the polygon [ [x1,y1], [x2,y2], ..., [xN,yN] ]
                                "text": str  # predicted text
                            },
                            ...
                        ]
                    }
                    "img_1.jpg": {
                    "predictions": [
                    ...
                    ]
                    }
            }

        annotation_json_path (str): Path to the segmentation annotation json
            in COCO format. The json should have the following format:
            {
                "images": [
                    {
                        "file_name": str,  # name of the image file
                        "id": int  # image id
                    },
                    ...
                ],
                "annotations": [
                    {
                        "image_id": int,  # the index of the image on which the polygon is located
                        "category_id": int,  # the polygonвЂ™s category index
                        "attributes": {"translation": str},  # text in the polygon
                        "segmentation": list  # the coordinates of the polygon
                    },
                    ...
                ]
            }
    """
    with open(annotation_json_path, "r") as f:
        data = json.load(f)
    with open(pred_json_path, "r") as f:
        pred_data = json.load(f)
        
    cer_avg = AverageMeter()
    for data_img in data["images"]:
        img_name = data_img["file_name"]
        image_id = data_img["id"]

        texts_from_image, polygons_from_image = get_data_from_image(data, image_id)


        for prediction in pred_data[img_name]["predictions"]:
            polygon = prediction["polygon"]
            prediction["shapely_polygon"] = to_shapely(polygon)

        pred_texts = []
        for gt_polygon in polygons_from_image:
            pred_texts.append(get_pred_text_for_gt_polygon(gt_polygon, pred_data[img_name]))

        # to penalty false positive prediction, that were not matched with gt
        for prediction in pred_data[img_name]["predictions"]:
            if prediction.get("matched") is None:
                pred_texts.append(prediction["text"])
                texts_from_image.append("")

        num_samples = len(pred_texts)
        cer_avg.update(cer(texts_from_image, pred_texts), num_samples)
    return {"CER": cer_avg.avg}


def get_data_from_image(data, image_id):
    texts = []
    polygons = []
    for idx, data_ann in enumerate(data["annotations"]):
        if (
            data_ann["image_id"] == image_id
            and data_ann["attributes"]
            and data_ann["attributes"]["translation"]
            and data_ann["segmentation"]
        ):
            polygon = numbers2coords(data_ann["segmentation"][0])
            polygons.append(polygon)
            texts.append(data_ann["attributes"]["translation"])
    return texts, polygons


def get_pred_text_for_gt_polygon(gt_polygon, pred_data):
    max_iou = 0
    pred_text_for_gt_bbox = ""
    matching_idx = None
    gt_polygon = to_shapely(gt_polygon)
    for idx, prediction in enumerate(pred_data["predictions"]):
        if prediction.get("matched") is None:
            shapely_polygon = prediction["shapely_polygon"]
            pred_text = prediction["text"]
            iou = iou_polygon(gt_polygon, shapely_polygon)
            if iou > max_iou:
                max_iou = iou
                pred_text_for_gt_bbox = pred_text
                matching_idx = idx

    # to prevent matching one predicted bbox to several ground true bboxes
    if matching_idx is not None:
        pred_data["predictions"][matching_idx]["matched"] = True
    return pred_text_for_gt_bbox


def to_shapely(polygon):
    shapely_polygon = Polygon(polygon)
    if shapely_polygon.is_valid:
        return shapely_polygon
    return None


def iou_polygon(polygon1, polygon2):
    if polygon1 is not None and polygon2 is not None:
        intersect = polygon1.intersection(polygon2).area
        union = polygon1.union(polygon2).area
        iou = intersect / union
        return iou
    else:
        return 0


def levenshtein_distance(first, second):
    distance = [[0 for _ in range(len(second) + 1)] for _ in range(len(first) + 1)]
    for i in range(len(first) + 1):
        for j in range(len(second) + 1):
            if i == 0:
                distance[i][j] = j
            elif j == 0:
                distance[i][j] = i
            else:
                diag = distance[i - 1][j - 1] + (first[i - 1] != second[j - 1])
                upper = distance[i - 1][j] + 1
                left = distance[i][j - 1] + 1
                distance[i][j] = min(diag, upper, left)
    return distance[len(first)][len(second)]


def cer(gt_texts, pred_texts):
    assert len(pred_texts) == len(gt_texts)
    lev_distances, num_gt_chars = 0, 0
    for pred_text, gt_text in zip(pred_texts, gt_texts):
        lev_distances += levenshtein_distance(pred_text, gt_text)
        num_gt_chars += len(gt_text)
    return lev_distances / num_gt_chars


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def numbers2coords(list_of_numbers):
    """Convert list of numbers to list of tuple coords x, y."""
    bbox = [
        [list_of_numbers[i], list_of_numbers[i + 1]]
        for i in range(0, len(list_of_numbers), 2)
    ]
    return np.array(bbox)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ref_path",
        type=str,
        default="annotation.json",
        help="Path to the segmentation annotation json in COCO format.",
    )
    parser.add_argument(
        "--pred_path",
        type=str,
        default="prediction.json",
        help="Path to json with predictions for each image.",
    )

    args = parser.parse_args()

    string_cer = evaluate(args.pred_path, args.ref_path, "")

    print(string_cer)
