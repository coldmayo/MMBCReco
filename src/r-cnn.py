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

from coco_utils import get_coco_api_from_dataset
from coco_eval import CocoEvaluator

data = {"total_loss": [], "numOfEpochs": 0}
numOfImgs = 50

class CustomDataset(Dataset):
    def __init__(self, transforms = None, train=True):
        self.path = "dataInfo.json"
        self.dir = "trainData/"
        if train == False:
            self.path = "valInfo.json"
            self.dir = "valData/"
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
        img_path = os.path.join(self.dir, img_info['file_name'])
        img = Image.open(img_path).convert("RGB")

        boxes = []
        labels = []

        for ann in anns:
            xmin = ann['bbox'][0]
            ymin = ann['bbox'][0]
            xmax = xmin + ann['bbox'][2]
            ymax = ymin + ann['bbox'][3]
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

def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0]+box1[2], box2[0] + box2[2])
    y2 = min(box1[1]+box1[3], box2[1] + box2[3])

    int_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]

    return int_area / float(box1_area + box2_area - int_area)

def evaluate(model, data_loader, device):
    model.eval()
    total_iou = 0
    total_samples = 0

    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        outputs = model(images)
        #print(len(images), len(outputs), len(targets))
        for target, output in zip(targets, outputs):
            pred_boxes = output['boxes'].cpu().tolist()
            pred_scores = output['scores'].cpu().tolist()
            gt_boxes = target['boxes'].cpu().tolist()

            pred_data = list(zip(pred_boxes, pred_scores))
            pred_data.sort(key=lambda x: x[1], reverse=True)

            num_preds = min(len(pred_data), len(gt_boxes))

            for i in range(len(gt_boxes)):
                pred_box, score = pred_data[:num_preds][i]
                #print(pred_box, gt_boxes[i], "\n")
                iou = calculate_iou(pred_box, gt_boxes[i])
                #print(iou)
                total_iou += iou
                total_samples += 1

    average_iou = total_iou / total_samples
    print(average_iou)

def train_model(model, data_loader, data_loader_test, optimizer, device, num_epochs):
    perc = numOfImgs/len(data_loader)
    poss = [0,1]
    data["numOfEpochs"] = num_epochs
    for epoch in range(num_epochs):
        loss = 0
        model.train()
        for images, targets in tqdm(data_loader):
            num = np.random.choice(poss, replace=False, p=[1-perc, perc])   # randomly selecting 50ish images, there is probably a better way to do this...
            if num == 1:
                images = list(image.to(device) for image in images)
                #print(len(images))
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                #print(targets)

                loss_dict = model(images, targets)
                if isinstance(loss_dict, dict):
                    losses = sum(loss for loss in loss_dict.values())
                    optimizer.zero_grad()
                    losses.backward()
                    optimizer.step()

                    data["total_loss"].append(losses.item())

                    loss += losses.item()
                else:
                    print(loss_dict)
                    loss_dict = loss_dict[0]
                    print(loss_dict)
                    losses = sum(loss for loss in loss_dict.values())
                    optimizer.zero_grad()
                    losses.backward()
                    optimizer.step()

                    data["total_loss"].append(losses.item())

                    loss += losses.item()                    

                
        evaluate(model, data_loader_test, device = device)
        print("Epoch: {}, Loss: {}".format(epoch, loss))
    return model

def main():
    
    dataset = CustomDataset(get_transform(train=True), train=True)
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=1, collate_fn=lambda x: tuple(zip(*x)))
    dataset_val = CustomDataset(get_transform(train=False), train=False)
    data_loader_test = DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=1, collate_fn=lambda x: tuple(zip(*x)))
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_classes = 2
    model = get_model(num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # get baseline before training
    #evaluate(model, data_loader_test, device=device)

    # training 
    num_epochs = 5
    print("Starting training...")
    model = train_model(model, data_loader, data_loader_test, optimizer, device, num_epochs)

    # val
    evaluate(model, data_loader_test, device=device)

    torch.save(model.state_dict(), "output/faster_rcnn.pth")
    

if __name__ == "__main__":
    main()
    with open('output/metrics.json', 'w') as f:
        json.dump(data, f)
