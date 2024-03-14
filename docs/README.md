<h1 align="center">
  <br>
  <br>
  MMBCReco
  <br>
</h1>

<h4 align="center">Using Machine Learning to Identify/Classify Bubble Chamber Tracks</h4>

<p align="center">
  <a href="#purpose">Purpose</a> •
  <a href="#features">Features</a> •
  <a href="#directory-breakdown">Directory Breakdown</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

## Abstract
The Modern Adaptive Modular Bubble chamber Archetype (MAMBA) is a design prototype for a future particle detector that will measure neutrino-nucleus interactions on light nuclear targets. Modern Bubble Chambers use image learning for the identification of nuclear recoils, but none of them have used machine learning on the Hertz timescale which is widely used by older bubble chambers. Proof-of-concept image learning and track identification on triggering timescales can be done with the Faster R-CNN, a machine learning architecture suited for object detection. The model is trained using generated images. In the future, a Deep Convolutional Generative Adversarial Network (DCGAN) will be used and it would be be trained on a video from a consumer grade cloud chamber. Progress will be presented and discussed.

## Features
- Faster R-CNN from Detectron2 for Identifying Tracks
- DCGAN using PyTorch that generates 128x128 images of tracks
- Script that generates training data in COCO format for R-CNN (will be replaced by DCGAN)

## Directory Breakdown
- dataManip
    - used to test out some image data related techniques 
- docs
    - documentation
- src
    - stores the code for the Faster R-CNN and DCGAN

## Credits
I would like to thank:
- FermiLab and FRIB for giving me jobs
- Dr. Bryan Ramson for being my advisor on this project

## License
MIT License