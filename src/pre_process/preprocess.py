import cv2
import numpy as np


class PreProcess:

    def extract_words(self, image):
        words = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((3, 3), np.uint8)
        dilation = cv2.dilate(thresh, kernel, iterations=5)
        contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            if w * h > 100 * 100:
                word = image[y:y + h, x:x + w]
                resize_word = cv2.resize(word, (400, 200))
                gray_scale_image = cv2.cvtColor(resize_word, cv2.COLOR_BGR2GRAY)
                image_final = np.where(gray_scale_image < 100, gray_scale_image, 1)
                words.append(image_final)
        return words
