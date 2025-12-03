#!/usr/bin/env bash

# INCOMPLETE - DO NOT RUN YET
# downloading pretrained dependencies
mkdir -p lib
touch __init__.py
# pretrained naturalspeech3 FACodec
cd lib && git clone https://github.com/lifeiteng/naturalspeech3_facodec.git lib
# FastSpeech2 text encoder architecture
cd lib && git clone https://github.com/ming024/FastSpeech2
# ControlSpeech and Vccm Dataset
cd lib && git clone https://github.com/jishengpeng/ControlSpeech
cd ControlSpeech && mv VccmDataset ../.. && cd ../VccmDataset
# get subset of training set of audio files from Google Drive (uses brew, Mac OS X only)
brew install gdown
gdown --fuzzy https://drive.google.com/file/d/1kNjYBqv_DohG8N3wF-J7kCSBmLxvs77N/view?usp=sharing
gdown --fuzzy https://drive.google.com/file/d/1W9DuRsQuP3tfWwxFo0Dx8-Rg-XbGCIzH/view?usp=sharing && unzip CREMA-D.zip -d temp
tar -czf CREMA-D.tar.gz -C temp/CREMA-D . && rm -rf temp