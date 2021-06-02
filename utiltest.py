from utils import find_bbox, find_distances, train_transform, geomap
import matplotlib.pyplot as plt
from dataset import CocoDataset, TOPIL, _convert_to_3channel
import numpy as np
import cv2
import random
import pickle
import pandas as pd
from constants import GEO_INDICIES

d = CocoDataset('/Users/arjunbemarkar/Python/ArjunEAST/COCO/cocotext.v2.organized.only_with_anns.json',  # 18521
                          '/Users/arjunbemarkar/Python/ArjunEAST/COCO/train2014/',
                          transform=train_transform, force_english=True, force_legibility=True)


for indx in range(len(d)):
    out = d[indx]
    geo_map = out['geomap']
    image = np.array(TOPIL(out['image']))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    anns = out['aabbs']

    print(d.aabb_format)
    print(geo_map.shape)

    fig = plt.figure(figsize=(10, 7))
    c = 2
    r = 3

    for i in range(0, 4):
        fig.add_subplot(r, c, i+1)
        plt.imshow(geo_map[:, :, i], cmap='hot', interpolation='nearest')
        plt.axis('off')
        plt.title(GEO_INDICIES[i])

    fig.add_subplot(r, c, 5)
    plt.imshow(image)
    plt.axis('off')
    plt.title('da image')

    fig.add_subplot(r, c, 6)
    plt.imshow(TOPIL(out['scoremap']))
    plt.axis('off')
    plt.title('da scoremap')
    plt.show()
    input()
    plt.close()

# plt.figure()
# plt.imshow(geo_map[:, :, 0], cmap='hot', interpolation='nearest'); plt.figure()
# plt.imshow(geo_map[:, :, 1], cmap='hot', interpolation='nearest'); plt.figure()
# plt.imshow(geo_map[:, :, 2], cmap='hot', interpolation='nearest'); plt.figure()
# plt.imshow(geo_map[:, :, 3], cmap='hot', interpolation='nearest')


# d = geomap((15, 10, 5), [
#     [1, 1, 5, 5, 0],
# ], scale=1)

# plt.imshow(d[:, :, 0], cmap='hot')  # b
# plt.imshow(d[:, :, 1], cmap='hot')  # t
# plt.imshow(d[:, :, 2], cmap='hot')  # r
# plt.imshow(d[:, :, 3], cmap='hot')  # l
# plt.show()



# a = [np.hstack(d[:, :, n]) for n in range(0, 4)]
#
# df = pd.DataFrame(d[:, :, 0])
# filepath = 'out.xlsx'
# df.to_excel(filepath, index=False)




