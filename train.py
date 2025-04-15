# train.py
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.datasets import CocoDetection
from torchvision import transforms
from torch.utils.data import DataLoader
import os
import random
from PIL import Image
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import time
from tqdm import tqdm

# Custom dataset loader with preprocessing and augmentation
class CocoDataset(CocoDetection):
    def __init__(self, img_folder, ann_file, train=True):
        super().__init__(img_folder, ann_file)
        self.train = train
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, idx):
        img, _ = super().__getitem__(idx)
        img_id = self.ids[idx]
        anno = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))

        # Convert [x, y, w, h] → [x_min, y_min, x_max, y_max]
        boxes = [[x, y, x + w, y + h] for (x, y, w, h) in [obj['bbox'] for obj in anno]]
        labels = [obj['category_id'] for obj in anno]

        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64),
            'image_id': torch.tensor([img_id])
        }

        img = img.convert("RGB")
        img = self.transform(img)

        return img, target


# Config
root = "../nycu-hw2-data/nycu-hw2-data"
train_img_folder = f'{root}/train'
train_ann_file = f'{root}/train.json'
val_img_folder = f'{root}/valid'
val_ann_file = f'{root}/valid.json'
batch_size = 2
num_classes = 11  # 0–9 digits + background

# Datasets and loaders
train_dataset = CocoDataset(train_img_folder, train_ann_file, train=True)
val_dataset = CocoDataset(val_img_folder, val_ann_file, train=False)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# Model setup
model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights='COCO_V1')
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

trainable_params = [name for name, param in model.named_parameters() if param.requires_grad]
print(trainable_params)

model = model.cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0025, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
num_epochs = 5
best_accuracy = 0.0
map_metric = MeanAveragePrecision()

# Training loop
for epoch in range(num_epochs):
    start = time.time()
    print(f'start epoch {epoch}')
    model.train()
    train_loss = 0.0

    for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1} - Training", leave=False):
        images = [img.cuda() for img in images]
        targets = [{k: v.cuda() for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        train_loss += losses.item()
    
    lr_scheduler.step()

    # Validation
    model.eval()
    map_metric.reset()
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc=f"Epoch {epoch+1} - Validation", leave=False):
            images = [img.cuda() for img in images]
            targets = [{k: v.cuda() for k, v in t.items()} for t in targets]

            outputs = model(images)
            outputs = [{k: v.cpu() for k, v in o.items()} for o in outputs]
            targets = [{k: v.cpu() for k, v in t.items()} for t in targets]

            map_metric.update(outputs, targets)

    map_results = map_metric.compute()
    print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}", end="")
    print(f", mAP@0.5 = {map_results['map_50']:.4f}, mAP@[.5:.95] = {map_results['map']:.4f}")

    if map_results['map'] > best_accuracy:
        best_accuracy = map_results['map']
        torch.save(model.state_dict(), 'fasterrcnn_digit_model.pth')
        print("Model saved.")

    print("epoch time: ", time.time() - start)

