import unittest
import funcnodes_yolo
import cv2
import os
import funcnodes as fn


class TestYolo(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.img = cv2.imread(os.path.join(os.path.dirname(__file__), "cat.jpg"))

    async def test_model(self):
        model = funcnodes_yolo.YOLO()
        results = model(self.img, verbose=False)
        result = results[0]
        labels = [model.names[i] for i in result.boxes.cls.int().cpu().tolist()]
        self.assertEqual(len(result.boxes), 1)
        self.assertEqual(labels, ["cat"])

    async def test_node(self):
        node: fn.Node = funcnodes_yolo.yolov8()
        img = funcnodes_yolo.funcnodes_opencv.OpenCVImageFormat(self.img)

        node.get_input("img").value = img

        await node

        labels = node.get_output("labels").value
        self.assertEqual(labels, ["cat"])

        annotated_img = node.get_output("annotated_img").value

        self.assertIsInstance(
            annotated_img, funcnodes_yolo.funcnodes_opencv.OpenCVImageFormat
        )

        conf = node.get_output("conf").value
        self.assertEqual(len(conf), 1)
        self.assertGreaterEqual(conf[0], 0.75)
