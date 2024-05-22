from typing import List, Tuple


import funcnodes_opencv
from ultralytics import YOLO, checks as ultralytics_checks
from funcnodes_images import ImageFormat
import funcnodes as fn

MODELS = {}


@fn.NodeDecorator(
    id="yolov8",
    name="YOLOv8",
    outputs=[
        {"name": "annotated_img"},
        {"name": "labels"},
        {"name": "conf"},
    ],
)
def yolov8(img: ImageFormat) -> Tuple[ImageFormat, List[str], List[float]]:
    data = img.to_cv2().data

    if "yolov8" not in MODELS:
        MODELS["yolov8"] = YOLO("yolov8n.pt")
    model = MODELS["yolov8"]
    results = model(data, verbose=False)
    result = results[0]
    labels = [model.names[i] for i in result.boxes.cls.int().cpu().tolist()]
    annotated_frame = result.plot()
    conf = result.boxes.conf.cpu().numpy().tolist()

    return (
        funcnodes_opencv.OpenCVImageFormat(annotated_frame),
        labels,
        conf,
    )


NODE_SHELF = fn.Shelf(
    nodes=[
        yolov8,
    ],
    name="YOLO",
    description="YOLO models for object detection.",
    subshelves=[],
)
