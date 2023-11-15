# file to test RCNN

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import cv2
import random
import os

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data import DatasetCatalog
import matplotlib.pyplot as plt
from detectron2.utils.visualizer import ColorMode
from detectron2.utils.visualizer import Visualizer

os.environ["LRU_CACHE_CAPACITY"] = '1'
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.DEVICE = "cpu"
cfg.MODEL.WEIGHTS = "MMBCReco/src/output/model_final.pth"
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.85   # set the testing threshold for this model
predictor = DefaultPredictor(cfg)

path = "MMBCReco/src/valData"
savePath = "MMBCReco/src/results"
j = 0
inText = []

for i in os.listdir(path):
    im = cv2.imread(path+"/"+i)
    output = predictor(im)
    v = Visualizer(im[:, :, ::-1],
        scale=2, 
        instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
    )
    v = v.draw_instance_predictions(output["instances"].to("cpu"))
    print("Predicted # of tracks: "+str(len(output["instances"].to("cpu"))))
    inText.append(str(len(output["instances"].to("cpu")))+", "+ str(j))
    plt.imshow(v.get_image()[:, :, ::-1])
    plt.savefig(savePath+"/pic"+str(j)+".png")
    print("image "+str(j)+" done\n")
    j = j+1

# save number of tracks for each image into text file

with open(savePath+'/results.txt', 'w') as f:
    for i in range(len(inText)):
        f.write(inText[i])
        f.write('\n')


print("check results.txt")