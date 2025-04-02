from pathlib import Path

from typing import List, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import patches
from rich import print

from data import common as cmn


class YoloPredictionBoundingBox(cmn.JsonBaseModel):
    bbox: List
    label: int
    confidence: float

    @classmethod
    def from_yolo_bbox(cls, det: torch.Tensor) -> "YoloPredictionBoundingBox":
        *xyxy, conf, label = det.cpu()
        coords = [int(xyxy[1]), int(xyxy[3]), int(xyxy[0]), int(xyxy[2])]
        # coords can directly be used as img[coords[0]: coords[1], coords[2]: coords{3]]
        return cls(label=int(label), confidence=conf, bbox=coords)


class YoloFramePredictions(cmn.JsonBaseModel):
    preds: List[YoloPredictionBoundingBox]

    def append(self, x: YoloPredictionBoundingBox):
        self.preds.append(x)

    def save_vis_on_image(self, image: np.ndarray, save_path: Path):
        bboxes = [pred.bbox for pred in self.preds]
        plot_bboxes(image=image, bboxes=bboxes, save_path=save_path)


def plot_bboxes(
    image: np.ndarray,
    bboxes: List[List[float]],
    save_path: Path,
    labels: Optional[List[str]] = None,
):
    """
    Args:
      image_file: str specifying the image file path
      bboxes: list of bounding box annotations for all the detections
      xywh: bool, if True, the bounding box annotations are specified as
        [xmin, ymin, width, height]. If False the annotations are specified as
        [xmin, ymin, xmax, ymax]. If you are unsure what the mode is try both
        and check the saved image to see which setting gives the
        correct visualization.

    """
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    plt.imshow(image)

    # Iterate over all the bounding boxes
    for i, bbox in enumerate(bboxes):
        ymin, ymax, xmin, xmax = bbox
        w = xmax - xmin
        h = ymax - ymin

        # add bounding boxes to the image
        box = patches.Rectangle((xmin, ymin), w, h, edgecolor="red", facecolor="none")

        ax.add_patch(box)

        if labels is not None:
            rx, ry = box.get_xy()
            cx = rx + box.get_width() / 2.0
            cy = ry + box.get_height() / 8.0
            l = ax.annotate(
                labels[i],
                (cx, cy),
                fontsize=8,
                fontweight="bold",
                color="white",
                ha="center",
                va="center",
            )
            l.set_bbox(dict(facecolor="red", alpha=0.5, edgecolor="red"))

    plt.axis("off")
    fig.savefig(str(save_path))
    plt.close()
    print(f"Saved image with detections to {save_path}")
