#!/usr/bin/env bash
set -e  # Exit on error

# Create lib directory
mkdir -p lib
touch lib/__init__.py

cd lib

# Pretrained NaturalSpeech3 FACodec
git clone https://github.com/lifeiteng/naturalspeech3_facodec.git

# FastSpeech2 text encoder architecture
git clone https://github.com/ming024/FastSpeech2

# ControlSpeech and VccmDataset
git clone https://github.com/jishengpeng/ControlSpeech

cd ..  # Back to project root

# Move VccmDataset to project root
mv lib/ControlSpeech/VccmDataset ./VccmDataset

# Install gdown (Linux/pip compatible)
pip install gdown

cd VccmDataset

# Download audio files from Google Drive
gdown --fuzzy https://drive.google.com/file/d/1kNjYBqv_DohG8N3wF-J7kCSBmLxvs77N/view?usp=sharing
gdown --fuzzy https://drive.google.com/file/d/1W9DuRsQuP3tfWwxFo0Dx8-Rg-XbGCIzH/view?usp=sharing
unzip CREMA-D.zip -d temp
tar -czf CREMA-D.tar.gz -C temp/CREMA-D . && rm -rf temp
