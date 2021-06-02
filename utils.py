import albumentations as A
import numpy as np
from constants import DIALATION


def find_distances(x, y, bbox):
    """Calculates the distances for bottom, top, right and left from the pixel to the edge of bbox
    :param x: pixel x
    :param y: pixel y
    :param bbox: bounding box in (xtop, ytop, w, h, a)
    :returns: tuple
    """
    x1, y1, w, h = bbox[0:4]

    top = abs(y1 - y)
    bottom = abs(top - h)
    left = abs(x - x1)
    right = abs(left - w) # edidting test

    return bottom, top, right, left


def find_bbox(x, y, bboxs):
    """Finds which bbox contains point
    :param x: pixel x
    :param y: pixel y
    :param bboxs: list of bounding boxes in (xtop, ytop, w, h, a)
    """

    for bbox in bboxs:
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]
        if x1 < x < x2 and y1 < y < y2:
            return bbox

    return None


def geomap(shape, bboxs, scale=1):
    """Generates the geomap(each element in the map represents a bounding box. Angle is 0
    :param shape: shape of desired geomap, must have 5 channels
    :param bboxs: list of bboxs
    :returns: np.ndarray
    """

    if shape[2] != 5:
        raise Exception("must have 5 channel shape")
    geo_map = np.empty((shape[0], shape[1], shape[2]))
    for y in range(shape[0]):
        for x in range(shape[1]):
            matching_bbox = find_bbox(x, y, bboxs)
            if matching_bbox is None:
                geo_map[y][x] = [0] * 5

            else:
                bottom, top, right, left = find_distances(x, y, matching_bbox)
                geo_map[y][x] = [
                    bottom * scale,
                    top * scale,
                    right * scale,
                    left * scale,
                    0  # for rotation
                ]

    return geo_map


def new_image_dims(img: np.ndarray):
    """Resizes image so that it is divisible by 32
    :param img: the image
    :returns: (int, int)
    """

    h, w, _ = img.shape
    new_h = h
    new_w = w

    if new_w % 32 != 0:
        new_w = int((new_w // 32 - 1) * 32)
    if new_h % 32 != 0:
        new_h = int((new_h // 32 - 1) * 32)

    new_h = 32 if new_h < 32 else new_h
    new_w = 32 if new_w < 32 else new_w

    return new_h, new_w


def resize_image_train(img, bboxes):
    """To Resize both the image and bounding boxes during training
    :param img: the image
    :param bboxes: list of bounding boxes
    :returns: (np.ndarray, List[List[int]])
    """
    dims = new_image_dims(img)
    transform = A.Compose([
        A.Resize(dims[0], dims[1]),
        A.RandomSizedBBoxSafeCrop(height=512, width=512),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ], bbox_params=A.BboxParams(format='coco'))
    transformed = transform(image=img, bboxes=bboxes)
    return transformed['image'], transformed['bboxes']


def clean_bbox(bboxes, img):
    """Bbox preprocessing step to make sure the whole rectangle is in the image bounds
    :param bboxes: list of bounding boxes
    :param img: the image
    """

    new_annotations = []
    h, w, _ = img.shape
    for ann in bboxes:
        if ann[0] < 0:
            ann[2] += ann[0]
            ann[0] = 0

        if ann[1] < 0:
            ann[3] += ann[1]
            ann[1] = 0

        if ann[0] + ann[2] > w:
            ann[2] = w - ann[0]

        if ann[1] + ann[3] > h:
            ann[3] = h - ann[1]

        new_annotations.append(ann)
    return new_annotations


def coco_to_pascalvoc(bboxes):
    """Converts COCO annotations to PascalVOC
    :params bboxes: list of the bounding boxes
    :returns: List of bounding boxes
    """
    new = []
    for bbox in bboxes:
        new.append([bbox[0],
                    bbox[1],
                    bbox[0] + bbox[2],
                    bbox[1] + bbox[3],
                    bbox[4]])
    return new


def train_transform(img, bboxes):
    """Final EAST transform callback. Makes sure that images are sized properly, bounding boxes are cleaned up and resized to
    the new image size.
    :params img: the image
    :params bboxes: list of bounding boxes
    :returns: (img, bboxes)
    """
    return resize_image_train(img, clean_bbox(bboxes, img))
