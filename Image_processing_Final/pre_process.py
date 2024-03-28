import cv2
import numpy as np
from skimage import feature
from typing import Tuple, List
from tqdm import tqdm
import mahotas
from skimage.feature import local_binary_pattern
class PreProcess:
    def __init__(self, shape: Tuple):
        self.image_shape = shape

    def resize_images(self, images):
        # print("Resizing images...")
        # processed_images = [cv2.resize(im.image_data, (self.image_shape[0], self.image_shape[1])) for im in images]
        processed_images = [cv2.resize(im, (self.image_shape[0], self.image_shape[1])) for im in images]
        for row in tqdm(processed_images):
            for i in range(len(row)):
                row[i] = row[i] / 255
        return np.array(processed_images)

    def patch_images(self, images, labels, segment_shape):
        # print("Patching images...")
        patched_images = []
        patched_labels = []

        for i in range(len(images)):
            image = images[i]
            label = labels[i]
            height = len(image)
            width = len(image[0])
            y = 0

            y_step = int(segment_shape[1] * 0.50)  # Overlap of 50% of the Y per patch
            x_step = int(segment_shape[0] * 0.50)  # Overlap of 50% of the X per patch

            while y <= height - y_step:
                if segment_shape[1] - y < y_step:
                    break
                x = 0
                while x < width - x_step:
                    if segment_shape[0] - x < x_step:
                        break
                    segment = image[y: y + segment_shape[1], x: x + segment_shape[0]].copy()
                    x += x_step
                    if segment.shape[0] < segment_shape[0] or segment.shape[1] < segment_shape[1]:
                        continue
                    patched_images.append(segment)
                    patched_labels.append(label)
                y += y_step

        return np.array(patched_images), patched_labels

    def arrange_labels_indexing_from_0(self, labels: List) -> List:
        arranged_labels = []
        print("unArrang labels", labels)
        for label in labels:
            if label == "male":
                label = 0
            else:
                label = 1
            arranged_labels.append(label)
        print("Arranged labels:", arranged_labels)
        return arranged_labels

    def preprocess_and_binarize_images(self, images):
        binarized_images = []
        for image in tqdm(images):
            # Convert to grayscale if necessary
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Apply Otsu's method for binarization
            _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            binarized_images.append(binary_image)

        return binarized_images

    def crop_text_from_reversed_binary_images(self, images, min_black_pixels=1):  # TODO: Fix, might not crop correctly
        # print("Cropping text from images...")
        cropped_images = []
        for image in tqdm(images):

            # Find top and bottom borders
            top = 0
            bottom = image.shape[0]
            for row in image:
                if sum(row) >= min_black_pixels:
                    top = max(top, row.tolist().index(0))
                    break
            for row in image[::-1]:
                if sum(row) >= min_black_pixels:
                    bottom = min(bottom, image.shape[0] - row.tolist().index(0) - 1)
                    break

            # Find left and right borders
            left = 0
            right = image.shape[1]
            for col in image.T:
                if sum(col) >= min_black_pixels:
                    left = max(left, col.tolist().index(0))
                    break
            for col in image.T[::-1]:
                if sum(col) >= min_black_pixels:
                    right = min(right, image.shape[1] - col.tolist().index(0) - 1)
                    break

            # Crop the image
            cropped_image = image[top:bottom, left:right]
            cropped_images.append(cropped_image)

        return cropped_images
