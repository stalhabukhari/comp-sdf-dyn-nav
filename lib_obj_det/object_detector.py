from pathlib import Path
from typing import Tuple, List

import numpy as np
import torch

from data import common as cmn
from lib_obj_det.object_detection_utils import (
    YoloPredictionBoundingBox,
    YoloFramePredictions,
)
from yolov5.detect import (
    DetectMultiBackend,
    check_img_size,
    non_max_suppression,
    scale_boxes,
)
from yolov5.utils.augmentations import letterbox


class YoloV5Detector:
    def __init__(
        self,
        weights_filepath: Path,
        dataset_config_filepath: Path,
        conf_thres: float,
        iou_thres: float,
        imgsz: int,
        device: torch.device,
    ):
        # TODO: If needed, use ONNX runtime for performance boosts
        assert ".pt" in str(weights_filepath)
        assert ".yaml" in str(dataset_config_filepath)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = device
        self.classes = list(
            cmn.parse_yaml_file(dataset_config_filepath)["names"].keys()
        )
        self.model = DetectMultiBackend(
            str(weights_filepath),
            device=device,
            dnn=False,
            data=str(dataset_config_filepath),
            fp16=False,
        )
        stride, names, pt = self.model.stride, self.model.names, self.model.pt
        self.pt = pt
        self.img_size = imgsz
        self.stride = stride
        bs = 1
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        self.model.warmup(
            imgsz=(1 if pt or self.model.triton else bs, 3, *imgsz)
        )  # warmup

    def inference(self, img: torch.Tensor):
        # assumes input is pre-processed, batched, in channels-first
        pred = self.model(img, augment=False, visualize=False)
        pred = non_max_suppression(
            pred,
            conf_thres=self.conf_thres,
            iou_thres=self.iou_thres,
            classes=self.classes,
            agnostic=False,
            max_det=20,
        )
        return pred

    def preprocess_image(self, img: np.ndarray) -> np.ndarray:
        # provided image should be in RGB format
        assert isinstance(img, np.ndarray)
        img = letterbox(img, self.img_size, stride=self.stride, auto=self.pt)[0]
        return img

    def get_bboxes_from_pred(
        self, pred: torch.Tensor, img_shape_resize: Tuple, img_shape_orig: Tuple
    ) -> List:
        # TODO: Should this be a static method?
        ho, wo = img_shape_orig
        h, w = img_shape_resize
        per_frame_preds = []
        for det in pred:
            det[:, :4] = scale_boxes((h, w), det[:, :4], (ho, wo)).round()
            # det is in xyxy format, instead of xywh
            frame_preds = YoloFramePredictions(preds=[])
            for bbox in det:
                yolo_pred = YoloPredictionBoundingBox.from_yolo_bbox(det=bbox)
                frame_preds.append(yolo_pred)
            per_frame_preds.append(frame_preds)
        return per_frame_preds

    def infer_bboxes_from_img(self, img: np.ndarray) -> List:
        # for pipeline
        img_shape_orig = img.shape[:2]

        img = self.preprocess_image(img=img)
        img_shape_resize = img.shape[:2]

        img = img.transpose((2, 0, 1))  # channels-first format
        img = np.ascontiguousarray(img).astype(np.float32)

        img = torch.from_numpy(img).to(self.device)
        img /= 255
        img = img[None]

        pred = self.inference(img=img)
        bboxes = self.get_bboxes_from_pred(
            pred=pred, img_shape_orig=img_shape_orig, img_shape_resize=img_shape_resize
        )
        return bboxes
