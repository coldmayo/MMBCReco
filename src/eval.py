from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from detectron2.data import DatasetCatalog
from detectron2.data import MetadataCatalog
#from numpy import random
import random
import cv2
from detectron2.data.datasets import register_coco_instances
register_coco_instances("my_dataset_train", {}, "./dataInfo.json", "/home/wallachmayas/bubbleID/src/trainData/")
register_coco_instances("my_dataset_test", {}, "./dataInfo.json","/home/wallachmayas/bubbleID/src/testData/")

metadata = MetadataCatalog.get("my_dataset_train")

os.environ["LRU_CACHE_CAPACITY"] = '1'   # solves memory leak problem (dont ask why)

cfg = get_cfg()
cfg.MODEL.DEVICE = "cpu"
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 1
num_gpu = 1
bs = (num_gpu * 2)
cfg.SOLVER.BASE_LR = 0.00025  # pick a good Learning rate (0.00025 should work I think)
cfg.SOLVER.MAX_ITER = 150
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only one class for now (tracks)

cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
dataset_dicts = DatasetCatalog.get("my_dataset_test")
metadata = MetadataCatalog.get("my_dataset_train")
plt.figure(figsize=(10, 8), dpi=80)

for i in range(8):
    for d in random.sample(dataset_dicts, 1):   # this is one because there is only one class right now
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1],
                   metadata=metadata, 
                   scale=2, 
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
        )
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        plt.subplot(2,4,i+1)
        plt.imshow(v.get_image()[:, :, ::-1])