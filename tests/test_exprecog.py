import unittest
import cv2

from exprecog.exceptions import InvalidImage
from exprecog.exprecog import Exprecog

detector = None


class TestExprecog(unittest.TestCase):
    def setUpClass():
        global detector
        detector = Exprecog()

    def test_detect_emotions(self):
        """
        Exprecog is able to detect faces image
        :return:
        """
        justin = cv2.imread("justin.jpg")

        result = detector.detect_emotions(justin)  # type: list

        self.assertEqual(len(result), 1)

        first = result[0]

        self.assertIn('box', first)
        self.assertIn('emotions', first)
        self.assertTrue(len(first['box']), 1)

    def test_detect_faces_invalid_content(self):
        """
        Exprecog detects invalid images
        :return:
        """
        justin = cv2.imread("example.py")

        with self.assertRaises(InvalidImage):
            result = detector.detect_emotions(justin)  # type: list

    def test_detect_no_faces_on_no_faces_content(self):
        """
        Exprecog successfully reports an empty list when no faces are detected.
        :return:
        """
        justin = cv2.imread("no-faces.jpg")

        result = detector.detect_emotions(justin)  # type: list
        self.assertEqual(len(result), 0)

    def tearDownClass():
        global detector
        del detector


if __name__ == '__main__':
    unittest.main()
