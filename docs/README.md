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

## Purpose
The purpose of the project is to develop a sophisticated classification and identification system for track data for the Modern Modular Bubble Chamber (MMBC). Bubble chambers are essential tools in high-energy physics research, capturing the paths of charged particles as they interact with the surrounding medium. However, the data produced by these experiments is voluminous, making it challenging to analyze by hand. This project aims to harness advanced machine learning and image recognition techniques to automatically recognize and classify particle tracks, thus simplifying the process of particle identification and enabling physicists to extract meaningful insights about fundamental particle interactions.

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