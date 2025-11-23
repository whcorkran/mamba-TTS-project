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
# Mac OS X only
brew install gdown
gdown --fuzzy https://drive.google.com/file/d/1kNjYBqv_DohG8N3wF-J7kCSBmLxvs77N/view?usp=sharing


