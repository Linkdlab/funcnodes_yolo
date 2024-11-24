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

        labels = node.get_output("labels").value
        conf = node.get_output("conf").value

        maxconf = np.argmax(conf)

        self.assertEqual(labels[maxconf], "cat", dict(zip(labels, conf)))

        annotated_img = node.get_output("annotated_img").value

        self.assertIsInstance(
            annotated_img, funcnodes_yolo.funcnodes_opencv.OpenCVImageFormat
        )

        conf = node.get_output("conf").value
        self.assertEqual(len(conf), 1)
        self.assertGreaterEqual(conf[0], 0.75)
