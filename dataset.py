import sys
import json
import cv2
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, ToPILImage
from collections import OrderedDict
import numpy as np
from utils import train_transform, coco_to_pascalvoc, geomap
from constants import DIALATION
import tqdm

TOTENSOR = ToTensor()
TOPIL = ToPILImage()

class CocoDataset(Dataset):
    def __init__(self, json_file, root_dir, transform=None, force_legibility=False, force_english=False, aabb_format='coco'):
        self.root_dir = root_dir  # root directory of images
        self.transform = transform  # transform function (for images and bboxs)
        self.son = json.loads(open(json_file, 'rb').read().decode('utf-8'), object_pairs_hook=OrderedDict) # ordered dict so that json parser indexes are kept consistent
        self.force_legibility = force_legibility
        self.force_english = force_english
        self.aabb_format = aabb_format

        self.imgs = []  # working list of all of the images of interest to be indexed
        if (not force_legibility) and (not force_english):
            self.imgs = list(self.son['imgs'].items())
            return

        for img in self.son['imgs'].items():
            english_anns = self.annotation_filter(img, 'language', 'english')
            legibility_anns = self.annotation_filter(img, 'legibility', 'legible')

            if force_legibility and force_english:
                union = self.unionize_anns(english_anns, legibility_anns)
                if len(union) > 0:
                    self.imgs.append(img)
                continue

            elif force_legibility and len(legibility_anns) > 0:
                self.imgs.append(img)

            elif force_english and len(english_anns) > 0:
                self.imgs.append(img)

    def img_anns(self, img):
        """Generator for image annotations"""
        ann_idx = self.son['imgToAnns'][img[0]]
        for i in ann_idx:
            yield self.son['anns'][str(i)]

    def annotation_filter(self, img, key, value):
        """picks the annotations which match the key and value pair"""
        anns = []
        for ann in self.img_anns(img):
            if ann[key] == value:
                anns.append(ann)
        return anns

    def unionize_anns(self, first, second):
        union = []
        for ann in first:
            if ann['id'] in [n['id'] for n in second]:
                union.append(ann)
        return union

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()

        img_ = self.imgs[idx]
        img = cv2.imread(self.root_dir + img_[1]['file_name'])

        aabbs = self.get_annotations(img_)

        if hasattr(self.transform, '__call__'):
            img, aabbs = self.transform(img, aabbs)

        scoremap = CocoDataset.scoremap(img, aabbs)
        geo_map = TOTENSOR(geomap([scoremap.shape[0], scoremap.shape[1], 5], np.array(aabbs, dtype=np.int) * DIALATION))
        scoremap = TOTENSOR(scoremap)

        if self.aabb_format == 'pascalvoc':
            aabbs = coco_to_pascalvoc(aabbs)

        img = TOTENSOR(img)

        return {'image': img, 'aabbs': aabbs, 'scoremap': scoremap, 'geomap': geo_map}

    def get_annotations(self, img_):
        annotations = self.son['imgToAnns'][img_[0]]
        aabbs = []
        for ann in annotations:
            ann = self.son['anns'][str(ann)]
            aabb = ann['bbox']
            aabb.append(0)  # rotation is set to 0.

            if (not self.force_legibility) and (not self.force_english):
                aabbs.append(aabb)
                continue

            if self.force_legibility and self.force_english:
                if ann['legibility'] == 'legible' and ann['language'] == 'english':
                    aabbs.append(aabb)
                continue

            elif self.force_legibility and ann['legibility'] == 'legible':
                aabbs.append(aabb)

            elif self.force_english and ann['language'] == 'english':
                aabbs.append(aabb)
        return aabbs

    @classmethod
    def scoremap(cls, img, aabbs, dilation=DIALATION):
        height, width, _ = img.shape
        if height % (1 / dilation) != 0 or width % (1 / dilation) != 0:
            raise Exception(f"recieved image whose dims are {(height, width)}. Not compatible with dilation factor : {dilation}")
        score_map = np.zeros((int(height * dilation), int(width * dilation), 1), dtype=np.float32)
        polys = np.empty((len(aabbs), 4, 2), dtype=np.float32)
        i = 0
        for x, y, w, h, _ in aabbs:
            polys[i] = np.array([
                [x, y],
                [x, y + h],
                [x+w, y+h],
                [x+w, y]
            ])
            i += 1
        polys *= dilation
        cv2.fillPoly(score_map, np.int32(polys), (255))
        return score_map


def _convert_to_3channel(img):
    img = img.reshape((img.shape[0], img.shape[1]))
    img2 = np.empty((img.shape[0], img.shape[1], 3))
    img2[:, :, 0] = img.copy()
    img2[:, :, 1] = img.copy()
    img2[:, :, 2] = img.copy()
    return img2


def _get_example(imid='260932'):
    son = json.loads(open('/Users/arjunbemarkar/Python/ArjunEAST/COCO/cocotext.v2.organized.only_with_anns.json', 'rb').read().decode('utf-8'), object_pairs_hook=OrderedDict)
    image = cv2.imread(f"COCO/train2014/{son['imgs'][imid]['file_name']}")
    anns = []
    for x in son['imgToAnns'][imid]:
        anns.append(son['anns'][str(x)]['bbox'])

    for bbox in anns:
        x, y, w, h = map(int, bbox)
        cv2.rectangle(image, (x, y), (x+w, y+h), color=(255, 0, 0), thickness=3)
    return image


def _example_test():
    cv2.imshow("img", _get_example('422792')) # 540965, 229078, 260932, 422792
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)


def show_imagewannotations(i, dataset, opencv, t):
    o = dataset[i]
    image = np.array(TOPIL(o['image']))
    scoremap = np.array(TOPIL(o['scoremap']))
    scoremap = scoremap.reshape((*scoremap.shape, 1))

    if len(o['aabbs']) == 0:
        print(f'no bounding boxes {i}')
    for aabb in o['aabbs']:
        x, y, x2, y2, _ = map(int, aabb)
        cv2.rectangle(image, (x, y), (x2, y2), color=(255, 0, 0), thickness=1)
    t.update(1)
    # nimg = np.concatenate((image, cv2.resize(_convert_to_3channel(scoremap), None, fy=4, fx=4)), axis=0)
    # cv2.imshow(f"image {i}", nimg)
    if opencv:
        cv2.imshow(f"im {i}", image)
        cv2.imshow(f"scr {i}", _convert_to_3channel(scoremap))
        cv2.waitKey(0)
        cv2.destroyAllWindows()



if __name__ == '__main__':
    opencv = False
    if len(sys.argv) > 1:
        if sys.argv[1] == "example":
            _get_example()
            sys.exit(0)
        if sys.argv[1] == "opencv":
           opencv = True

    dataset = CocoDataset('/Users/arjunbemarkar/Python/ArjunEAST/COCO/cocotext.v2.organized.only_with_anns.json',  # 18521
                          '/Users/arjunbemarkar/Python/ArjunEAST/COCO/train2014/',
                          transform=train_transform, force_english=True, force_legibility=True,
                          aabb_format='pascalvoc')
    print(len(dataset))
    t = tqdm.tqdm(total=len(dataset))
    for i in range(len(dataset)):
        show_imagewannotations(i, dataset, opencv, t)
    sys.exit()






