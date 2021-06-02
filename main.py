from model import FeatureExtractor, EastOCR
import torch

v = FeatureExtractor()
east_ocr: EastOCR = EastOCR(v)
east_ocr.load_state_dict(torch.load('/Users/arjunbemarkar/Python/ArjunEAST/IncidentalSceneTextDataset/myeast_vgg16.pth', map_location=torch.device('cpu')))
east_ocr.eval()
d = east_ocr(torch.rand(1, 3, 224, 224))
