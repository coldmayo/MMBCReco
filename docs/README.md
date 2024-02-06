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
The Modern Modular Bubble Chamber (MMBC) is a future particle
detector which will mainly be used for displaying muon tracks. However,the amount of data produced by these experiments could be very large which could cause analysis to take a very long time. This project aims to develop a sophisticated classification and identification system for track data within the MMBC. This paper will discuss how the Faster R-CNN
was used to automate the recognition and classification of particle tracks.

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