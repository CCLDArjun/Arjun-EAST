import os

import torch
from torch.utils.data import DataLoader
from utils import train_transform
from dataset import CocoDataset
from model import FeatureExtractor, EastOCR
from loss import EASTLoss
import time


def run():
    torch.multiprocessing.freeze_support()
    root_dir = '/Users/arjunbemarkar/Python/ArjunEAST/COCO/train2014/'
    json_file = '/Users/arjunbemarkar/Python/ArjunEAST/COCO/cocotext.v2.organized.only_with_anns.json'

    model = EastOCR(feature_extractor=FeatureExtractor())
    dataset = CocoDataset(json_file, root_dir, transform=train_transform)
    data_loader = DataLoader(dataset, batch_size=24, shuffle=True, num_workers=10, drop_last=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[m * 80 for m in range(1, 10)], gamma=0.1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ctx = EASTLoss()
    EPOCHS = 600
    save_dir = '/Users/arjunbemarkar/Python/ArjunEAST/save/'

    for x in range(EPOCHS):
        model.train()
        scheduler.step()
        e_loss = 0
        start_time = time.time()
        # {'image': img, 'aabbs': aabbs, 'scoremap': scoremap, 'geomap': geo_map}
        has_multi_gpu = False
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
            has_multi_gpu = True
        for i, data in enumerate(data_loader):
            img, gt_score, gt_aabb_map = data['image'].to(device), \
                                                      data['scoremap'].to(device), \
                                                      data['geomap'].to(device)

            pred_score, pred_rbox = model(img)
            loss = ctx(gt_score, pred_score, gt_aabb_map, pred_rbox)

            e_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Epoch [{x + 1} / {EPOCHS}], time is {int(time.time() - start_time) / 60} mins, batch loss is {loss.item()}")

        if x % 10 == 0:
            print("*" * 10)
            state = 0
            if has_multi_gpu:
                state = model.module.state_dict()
            else:
                state = model.state_dict()

            torch.save(state, os.path.join(save_dir, f'EAST_EPOCH_{x+1}.pth'))


if __name__ == '__main__':
    run()


