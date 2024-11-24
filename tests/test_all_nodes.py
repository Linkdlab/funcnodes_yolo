import sys
import os
from typing import List
import unittest
import funcnodes as fn
import funcnodes_yolo
import numpy as np
import cv2

sys.path.append(
    os.path.dirname(os.path.abspath(__file__))
)  # in case test folder is not in sys path
from all_nodes_test_base import TestAllNodesBase  # noqa: E402

fn.config.IN_NODE_TEST = True


class TestAllNodes(TestAllNodesBase):
    # in this test class all nodes should be triggered at least once to mark them as testing

    # if you tests your nodes with in other test classes, add them here
    # this will automtically extend this test class with the tests in the other test classes
    # but this will also mean if you run all tests these tests might run multiple times
    # also the correspondinig setups and teardowns will not be called, so the tests should be
    # independently callable
    sub_test_classes: List[unittest.IsolatedAsyncioTestCase] = []

    # if you have specific nodes you dont want to test, add them here
    # But why would you do that, it will ruin the coverage?!
    # a specific use case would be ignore nodes that e.g. load a lot of data, but there we would recommend
    # to write tests with patches and not ignore them.
    ignore_nodes: List[fn.Node] = []

    def setUp(self) -> None:
        img = cv2.imread(os.path.join(os.path.dirname(__file__), "cat.jpg"))
        self.img = img
        cv2.resize(
            img,
            img.shape[:2],
            fx=0.5,
            fy=0.5,
        )

    #  in this test class all nodes should be triggered at least once to mark them as testing
    async def test_yolov8(self):
        node: fn.Node = funcnodes_yolo.yolov8()
        img = funcnodes_yolo.funcnodes_opencv.OpenCVImageFormat(self.img)

        node.get_input("img").value = img

        await node

        result: funcnodes_yolo.YOLOResults = node.get_output("result").value
        conf = result.conf
        labels = result.labels
        maxconf = np.argmax(conf)

        self.assertEqual(labels[maxconf], "cat", dict(zip(labels, conf)))

        annotated_img = node.get_output("annotated_img").value

        self.assertIsInstance(
            annotated_img, funcnodes_yolo.funcnodes_opencv.OpenCVImageFormat
        )

        self.assertEqual(len(conf), 1)
        self.assertGreaterEqual(conf[0], 0.75)

    async def test_filter_yolo(self):
        yolonode: fn.Node = funcnodes_yolo.yolov8()
        img = funcnodes_yolo.funcnodes_opencv.OpenCVImageFormat(self.img)

        yolonode.get_input("img").value = img

        filternode: fn.Node = funcnodes_yolo.filter_yolo()
        filternode.get_input("yolo").connect(yolonode.get_output("result"))

        filternode.get_input("labels").value = "cat"

        await fn.run_until_complete(filternode, yolonode)

        positive: funcnodes_yolo.YOLOResults = filternode.get_output("positive").value
        negative: funcnodes_yolo.YOLOResults = filternode.get_output("negative").value

        self.assertEqual(len(positive), 1)
        self.assertEqual(len(negative), 0)

        filternode.get_input("conf").value = 1.0

        await filternode

        positive: funcnodes_yolo.YOLOResults = filternode.get_output("positive").value
        negative: funcnodes_yolo.YOLOResults = filternode.get_output("negative").value

        self.assertEqual(len(positive), 0, positive.conf)
        self.assertEqual(len(negative), 1)

    async def test_box_params_yolo(self):
        yolonode: fn.Node = funcnodes_yolo.yolov8()
        img = funcnodes_yolo.funcnodes_opencv.OpenCVImageFormat(self.img)

        yolonode.get_input("img").value = img
        await yolonode

        boxnode: fn.Node = funcnodes_yolo.get_box_params()
        boxnode.get_input("box").value = yolonode.get_output("result").value[0]

        await boxnode

        labels = boxnode.get_output("label").value
        conf = boxnode.get_output("conf").value
        x1 = boxnode.get_output("x1").value
        y1 = boxnode.get_output("y1").value
        w = boxnode.get_output("w").value
        h = boxnode.get_output("h").value
        x2 = boxnode.get_output("x2").value
        y2 = boxnode.get_output("y2").value

        self.assertEqual(labels, "cat")
        self.assertGreaterEqual(conf, 0.75)
        self.assertLessEqual(x1, 350)
        self.assertLessEqual(y1, 200)

        self.assertGreaterEqual(x2, 700)
        self.assertGreaterEqual(y2, 1200)
        self.assertEqual(w, x2 - x1)
        self.assertEqual(h, y2 - y1)

        img = boxnode.get_output("img").value

        self.assertIsInstance(img, funcnodes_yolo.funcnodes_opencv.OpenCVImageFormat)

        self.assertAlmostEqual(img.width(), w, delta=1)
        self.assertAlmostEqual(img.height(), h, delta=1)
