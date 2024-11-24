from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Union

import funcnodes_opencv
from ultralytics.engine.results import Boxes, Results
from ultralytics import YOLO

from funcnodes_images import ImageFormat
import funcnodes as fn
import numpy as np

MODELS = {}


class YOLOResultsEntry:
    def __init__(
        self,
        results: YOLOResults,
        index: int,
    ):
        self._results = results
        self._index = index

    @property
    def label(self) -> str:
        return self._results.labels[self._index]

    @property
    def conf(self) -> float:
        return self._results.conf[self._index]

    @property
    def xyxy(self) -> np.ndarray:
        return self._results.xyxy[self._index]

    @property
    def xywh(self) -> np.ndarray:
        return self._results.xywh[self._index]

    @property
    def x1(self) -> float:
        return self.xyxy[0]

    @property
    def y1(self) -> float:
        return self.xyxy[1]

    @property
    def x2(self) -> float:
        return self.xyxy[2]

    @property
    def y2(self) -> float:
        return self.xyxy[3]

    @property
    def w(self) -> float:
        return self.xywh[2]

    @property
    def h(self) -> float:
        return self.xywh[3]

    @property
    def img(self) -> np.ndarray:
        xyxy = self.xyxy
        return self._results._img.crop(*xyxy)


class YOLOResults:
    def __init__(self, results: Results, model: YOLO, img: ImageFormat):
        self._results = results
        self._model = model
        self._img = img

    @property
    def available_labels(self) -> Dict[int, str]:
        return self._model.names

    @property
    def boxes(self) -> Boxes:
        return self._results.boxes

    @property
    def labels(self) -> List[str]:
        return [self._model.names[i] for i in self.boxes.cls.int().cpu().tolist()]

    @property
    def conf(self) -> np.ndarray:
        return self.boxes.conf.cpu().numpy()

    def __len__(self) -> int:
        return len(self.boxes)

    def filter(self, indices: List[int]) -> YOLOResults:
        return YOLOResults(self._results[indices], self._model, self._img)

    def __iter__(self):
        for i in range(len(self)):
            yield YOLOResultsEntry(self, i)

    @property
    def xyxy(self) -> np.ndarray:
        return self._results.boxes.xyxy.cpu().numpy()

    @property
    def xywh(self) -> np.ndarray:
        return self._results.boxes.xywh.cpu().numpy()

    def __getitem__(self, index: int) -> YOLOResultsEntry:
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of range.")
        return YOLOResultsEntry(self, index)


@fn.NodeDecorator(
    id="fn_yolo.yolov8",
    name="YOLOv8",
    outputs=[
        {"name": "annotated_img"},
        {"name": "result"},
    ],
)
def yolov8(img: ImageFormat) -> Tuple[ImageFormat, YOLOResults]:
    data = img.to_cv2().data

    if "yolov8" not in MODELS:
        MODELS["yolov8"] = YOLO("yolov8n.pt")
    model = MODELS["yolov8"]
    results = model.predict(data, verbose=False)
    result = results[0]
    annotated_frame = result.plot()

    return (
        funcnodes_opencv.OpenCVImageFormat(annotated_frame),
        YOLOResults(result, model, img),
    )


@fn.NodeDecorator(
    id="fn_yolo.filter_yolo",
    name="Filter YOLO Results",
    outputs=[
        {"name": "positive"},
        {"name": "negative"},
    ],
)
def filter_yolo(
    yolo: YOLOResults,
    labels: Optional[Union[str, List[str]]] = None,
    conf: Optional[float] = None,
) -> Tuple[YOLOResults, YOLOResults]:
    all_indices = list(range(len(yolo)))
    positive_inidices = list(range(len(yolo)))
    negative_indices = []
    if labels is not None:
        if isinstance(labels, str):
            labels = [labels]
        box_labels = yolo.labels
        for i, label in enumerate(box_labels):
            if label not in labels:
                if i in positive_inidices:
                    positive_inidices.remove(i)

    if conf is not None:
        conf = float(conf)
        box_conf = yolo.conf
        for i, boxconf in enumerate(box_conf):
            if boxconf < conf:
                if i in positive_inidices:
                    positive_inidices.remove(i)
    negative_indices = [i for i in all_indices if i not in positive_inidices]
    return yolo.filter(positive_inidices), yolo.filter(negative_indices)


@fn.NodeDecorator(
    id="fn_yolo.get_box_params",
    name="Get YOLO Box Parameters",
    outputs=[
        {"name": "img"},
        {"name": "label"},
        {"name": "conf"},
        {"name": "x1"},
        {"name": "y1"},
        {"name": "w"},
        {"name": "h"},
        {"name": "x2", "hidden": True},
        {"name": "y2", "hidden": True},
    ],
)
def get_box_params(
    box: YOLOResultsEntry,
) -> Tuple[ImageFormat, str, float, float, float, float, float, float, float]:
    return (
        box.img,
        box.label,
        box.conf,
        box.x1,
        box.y1,
        box.w,
        box.h,
        box.x2,
        box.y2,
    )


NODE_SHELF = fn.Shelf(
    nodes=[yolov8, filter_yolo, get_box_params],
    name="YOLO",
    description="YOLO models for object detection.",
    subshelves=[],
)
