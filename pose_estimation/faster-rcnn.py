import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import functional as F
import albumentations as A
import math

import transforms
import utils
import engine
import train
from utils import collate_fn
from engine import train_one_epoch, evaluate

!unzip data_copy.zip

def train_transform():
    return A.Compose([
        A.Sequential([
            A.RandomRotate90(p=1),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, brightness_by_max=True, always_apply=False, p=1),
        ], p=1)
    ],
    bbox_params=A.BboxParams(format='pascal_voc', label_fields=['bboxes_labels'])
    )

class ElderlyFallDataset(Dataset):
    def __init__(self, root, transform=None, demo=False):
        self.root = root
        self.transform = transform
        self.demo = demo
        self.imgs_files = sorted(os.listdir(os.path.join(root, "images")))
        self.annotations_files = sorted(os.listdir(os.path.join(root, "annotations")))
        self.W = 240
        self.H = 240

    def yolobbox_to_bbox(self, x, y, w, h):
        x1, y1 = x - w / 2, y - h / 2
        x2, y2 = x + w / 2, y + h / 2
        return x1, y1, x2, y2

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "images", self.imgs_files[idx])

        img_original = cv2.imread(img_path)
        img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)

        img_file_name = os.path.basename(img_path)
        img_file_name = os.path.splitext(img_file_name)[0]

        annotation_file_name = f'{img_file_name}_annotated.json'
        annotations_path = os.path.join(self.root, 'annotations', annotation_file_name)

        with open(annotations_path) as f:
            data = json.load(f)
            bboxes_original = []

            for ann in data["bboxes"]:
                bbox = ann
                xmin, ymin, xmax, ymax = bbox
                bboxes_original.append([xmin, ymin, xmax, ymax])

            bboxes_labels_original = ['Person' for _ in bboxes_original]

        if self.transform:
            transformed = self.transform(image=img_original, bboxes=bboxes_original, bboxes_labels=bboxes_labels_original)
            img = transformed['image']
            bboxes = transformed['bboxes']
        else:
            img, bboxes = img_original, bboxes_original

        num_instances = len(bboxes)

        inst_ids = torch.as_tensor([i for i in range(num_instances)], dtype=torch.int64)

        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        target = {}
        target["boxes"] = bboxes
        target["labels"] = torch.as_tensor([1 for _ in bboxes], dtype=torch.int64) 
        target["image_id"] = idx
        target["id"] = inst_ids
        target["area"] = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        target["iscrowd"] = torch.zeros(len(bboxes), dtype=torch.int64)
        img = F.to_tensor(img)

        bboxes_original = torch.as_tensor(bboxes_original, dtype=torch.float32)
        target_original = {}
        target_original["boxes"] = bboxes_original
        target_original["labels"] = torch.as_tensor([1 for _ in bboxes_original], dtype=torch.int64)
        target_original["image_id"] = idx
        target["id"] = inst_ids
        target_original["area"] = (bboxes_original[:, 3] - bboxes_original[:, 1]) * (bboxes_original[:, 2] - bboxes_original[:, 0])
        target_original["iscrowd"] = torch.zeros(len(bboxes_original), dtype=torch.int64)
        img_original = F.to_tensor(img_original)

        if self.demo:
            return img, target, img_original, target_original
        else:
            return img, target

    def __len__(self):
        return len(self.imgs_files)

KEYPOINTS_FOLDER_TRAIN = 'data/train'
dataset = ElderlyFallDataset(KEYPOINTS_FOLDER_TRAIN, transform=train_transform(), demo=True)
data_loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

iterator = iter(data_loader)
batch = next(iterator)

print("Original targets:\n", batch[3], "\n\n")
print("Transformed targets:\n", batch[1])

def visualize(image, bboxes, image_original=None, bboxes_original=None):
    fontsize = 18

    for bbox in bboxes:
        start_point = (bbox[0], bbox[1])
        end_point = (bbox[2], bbox[3])
        image = cv2.rectangle(image.copy(), start_point, end_point, (0,255,0), 2)

    if image_original is None:
        plt.figure(figsize=(40,40))
        plt.imshow(image)
    else:
        for bbox in bboxes_original:
            start_point = (bbox[0], bbox[1])
            end_point = (bbox[2], bbox[3])
            image_original = cv2.rectangle(image_original.copy(), start_point, end_point, (0,255,0), 2)

        f, ax = plt.subplots(1, 2, figsize=(40, 20))
        ax[0].imshow(image_original)
        ax[0].set_title('Original image', fontsize=fontsize)

        ax[1].imshow(image)
        ax[1].set_title('Transformed image', fontsize=fontsize)

image = (batch[0][0].permute(1,2,0).numpy() * 255).astype(np.uint8)
bboxes = batch[1][0]['boxes'].detach().cpu().numpy().astype(np.int32).tolist()

image_original = (batch[2][0].permute(1,2,0).numpy() * 255).astype(np.uint8)
bboxes_original = batch[3][0]['boxes'].detach().cpu().numpy().astype(np.int32).tolist()

visualize(image, bboxes, image_original, bboxes_original)

def get_model(weights_path=None, num_classes=2):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, pretrained_backbone=True)

    if weights_path:
        state_dict = torch.load(weights_path)
        model.load_state_dict(state_dict)

    return model

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

KEYPOINTS_FOLDER_TRAIN = 'data/train'
KEYPOINTS_FOLDER_TEST = 'data/test'

dataset_train = ElderlyFallDataset(KEYPOINTS_FOLDER_TRAIN, transform=train_transform(), demo=False)
dataset_test = ElderlyFallDataset(KEYPOINTS_FOLDER_TEST, transform=None, demo=False)

data_loader_train = DataLoader(dataset_train, batch_size=3, shuffle=True, collate_fn=collate_fn)
data_loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, collate_fn=collate_fn)

model = get_model()
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.3)
num_epochs = 25

for epoch in range(num_epochs):
    train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=1000)
    lr_scheduler.step()
    evaluate(model, data_loader_test, device)

torch.save(model.state_dict(), 'maskrcnn_weights_resnet101.pth')
