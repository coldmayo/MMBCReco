import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import functional

from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from tqdm import tqdm
import json

data = {"total_loss": [], "numOfEpochs": 0}
numOfImgs = 50

class CustomDataset(Dataset):
    def __init__(self, transforms = None):
        self.path = "dataInfo.json"
        self.trans = transforms
        self.coco = COCO(self.path)
        self.ids = list(self.coco.imgs.keys())
        self.transforms = transforms

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join("trainData/", img_info['file_name'])
        img = Image.open(img_path).convert("RGB")

        boxes = []
        labels = []

        for ann in anns:
            xmin = ann['bbox'][0]
            ymin = ann['bbox'][0]
            xmax = xmin +ann['bbox'][2]
            ymax = ymin +ann['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(ann['category_id'])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.as_tensor([ann['iscrowd'] for ann in anns], dtype = torch.int64)

        target = {"boxes": boxes, "labels": labels, "image_id": image_id, "area": area, "iscrowd":iscrowd}

        if self.transforms:
            img = self.transforms(img)

        return img, target

def get_transform(train):
    transforms = []
    transforms.append(torchvision.transforms.ToTensor())
    if train:
        transforms.append(torchvision.transforms.RandomHorizontalFlip(0.5))
    return torchvision.transforms.Compose(transforms)

def get_model(numClasses):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn()
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, numClasses)
    return model

def train_model(model, data_loader, optimizer, device, num_epochs):
    model.train()
    perc = numOfImgs/len(data_loader)
    poss = [0,1]
    data["numOfEpochs"] = num_epochs
    for epoch in range(num_epochs):
        loss = 0
        for images, targets in tqdm(data_loader):
            num = np.random.choice(poss, replace=False, p=[1-perc, perc])   # randomly selecting 50ish images, there is probably a better way to do this...
            if num == 1:
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                data["total_loss"].append(losses.item())

                loss += losses.item()
        print("Epoch: {}, Loss: {}".format(epoch, loss))
    return model

def main():
    dataset = CustomDataset(get_transform(train=True))
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=1, collate_fn=lambda x: tuple(zip(*x)))

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_classes = 2
    model = get_model(num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    num_epochs = 5
    print("Starting training...")
    model = train_model(model, data_loader, optimizer, device, num_epochs)

    torch.save(model.state_dict(), "output/faster_rcnn.pth")

if __name__ == "__main__":
    main()
    with open('output/metrics.json', 'w') as f:
        json.dump(data, f)
