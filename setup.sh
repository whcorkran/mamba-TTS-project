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
# get smaller training set of audio files from Google Drive (uses brew, Mac OS X only)
brew install gdown
gdown --fuzzy https://drive.google.com/file/d/1kNjYBqv_DohG8N3wF-J7kCSBmLxvs77N/view?usp=sharing
gdown --fuzzy https://drive.google.com/file/d/1W9DuRsQuP3tfWwxFo0Dx8-Rg-XbGCIzH/view?usp=sharing && unzip CREMA-D.zip -d temp
tar -czf CREMA-D.tar.gz -C temp/CREMA-D . && rm -rf temp

# create debug dataset, contains only first 1024 samples from controlspeech_train.csv
mkdir ../debug_set
head -n 1024 ControlSpeech/VccmDataset/controlspeech_train.csv > ../debug_set/controlspeech_debug.csv && cd ../debug_set

cat controlspeech_debug.csv | while IFS= read -r line; do
    item_name=$(echo $line | cut -d ',' -f 1)
    # Copy wav file for this item_name, preserving directory structure
    # The item_name from the csv has hyphens ("-") which should be replaced by "/" for subdirectories.
    # The original wav file is found under the tarball VccmDataset/TextrolSpeech_data.tar.gz, with the path: item_name (hyphens replaced by slashes).wav
    # We will extract the .wav and place it at the equivalent location in ./ preserving the structure.

    # Skip the CSV header
    if [ "$item_name" = "item_name" ]; then
        continue
    fi

    # Replace hyphens with slashes for the path (bash parameter substitution)
    item_path=$(echo "$item_name" | sed 's/-/\//g').wav

    # Make the subdirectory if needed
    dir_path=$(dirname "$item_path")
    mkdir -p "$dir_path"

    # Extract the wav file from the tarball into the appropriate subdirectory
    tar -xzf ../VccmDataset/TextrolSpeech_data.tar.gz -C . "$item_path"

# compress extracted files into a new tarball
tar -czf debug_audio.tar.gz .



