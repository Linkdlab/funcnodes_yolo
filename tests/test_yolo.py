import unittest
import funcnodes_yolo
import cv2
import os


class TestYolo(unittest.TestCase):
    def setUp(self) -> None:
        img = cv2.imread(os.path.join(os.path.dirname(__file__), "cat.jpg"))
        self.img = img
        cv2.resize(
            img,
            img.shape[:2],
            fx=0.5,
            fy=0.5,
        )

    def test_model(self):
        model = funcnodes_yolo.YOLO(model="yolov8n.pt")
        results = model.predict(self.img, verbose=True)
        self.assertEqual(len(results), 1)
        result = results[0]

        print(result.boxes)

        labels = [result.names[i] for i in result.boxes.cls.int().cpu().tolist()]
        self.assertEqual(len(result.boxes), 1)
        self.assertEqual(labels, ["cat"])
