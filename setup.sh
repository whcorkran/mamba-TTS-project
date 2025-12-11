#!/usr/bin/env bash
set -e  # Exit on error

echo "=========================================="
echo "ControlSpeech + Mamba TTS - Full Setup"
echo "=========================================="

# Create lib directory
mkdir -p lib
touch lib/__init__.py

cd lib

# Clone required repositories
echo "[1/8] Cloning repositories..."
git clone https://github.com/lifeiteng/naturalspeech3_facodec.git || true
git clone https://github.com/ming024/FastSpeech2 || true
git clone https://github.com/jishengpeng/ControlSpeech || true

cd ..  # Back to project root

# Move VccmDataset to project root (if not already done)
if [ ! -d "VccmDataset" ]; then
    mv lib/ControlSpeech/VccmDataset ./VccmDataset
fi

# Install gdown
pip install gdown

cd VccmDataset

# Download emotional datasets from Google Drive
echo "[2/8] Downloading emotional datasets from Google Drive..."
if [ ! -f "TextrolSpeech_data.tar.gz" ]; then
    gdown --fuzzy https://drive.google.com/file/d/1kNjYBqv_DohG8N3wF-J7kCSBmLxvs77N/view?usp=sharing
fi
if [ ! -f "CREMA-D.zip" ] && [ ! -f "CREMA-D.tar.gz" ]; then
    gdown --fuzzy https://drive.google.com/file/d/1W9DuRsQuP3tfWwxFo0Dx8-Rg-XbGCIzH/view?usp=sharing
    unzip CREMA-D.zip -d temp
    tar -czf CREMA-D.tar.gz -C temp/CREMA-D . && rm -rf temp CREMA-D.zip
fi

# Download LibriTTS dataset (~75GB total)
echo "[3/8] Downloading LibriTTS dataset (this may take 30+ minutes)..."
mkdir -p libritts && cd libritts
wget -c https://www.openslr.org/resources/60/train-clean-100.tar.gz &
wget -c https://www.openslr.org/resources/60/train-clean-360.tar.gz &
wget -c https://www.openslr.org/resources/60/train-other-500.tar.gz &
wait  # Wait for all downloads to complete
cd ..

# Extract all audio
echo "[4/8] Extracting audio files..."
mkdir -p audio_extracted

# Extract emotional datasets
echo "  Extracting TextrolSpeech..."
tar -xzf TextrolSpeech_data.tar.gz -C audio_extracted/

echo "  Extracting CREMA-D..."
mkdir -p audio_extracted/CREMA
tar -xzf CREMA-D.tar.gz -C audio_extracted/CREMA/

# Extract LibriTTS
echo "  Extracting LibriTTS (this may take several minutes)..."
cd libritts
for f in *.tar.gz; do 
    echo "    Extracting $f..."
    tar -xzf "$f"
done
mv LibriTTS ../audio_extracted/
cd ..

# Create transcripts and pair with audio
echo "[5/8] Creating and pairing transcripts with audio files..."
cd ..  # Back to project root

python3 << 'PYTHON_SCRIPT'
import csv
from pathlib import Path
from tqdm import tqdm

audio_root = Path("VccmDataset/audio_extracted")
csv_path = Path("VccmDataset/controlspeech_train.csv")

print("Building wav file index...")
wav_index = {}
for wav_path in tqdm(list(audio_root.rglob("*.wav")), desc="Indexing wavs"):
    wav_index[wav_path.stem] = wav_path

print(f"Indexed {len(wav_index)} wav files")

print("Creating and placing transcripts...")
with open(csv_path, 'r', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

matched = 0
unmatched = 0
for row in tqdm(rows, desc="Processing"):
    item_name = row['item_name']
    text = row['txt']
    
    # Try direct match first (LibriTTS style: 6904_262291_000067_000000)
    if item_name in wav_index:
        txt_path = wav_index[item_name].with_suffix(".txt")
        if not txt_path.exists():
            txt_path.write_text(text)
        matched += 1
        continue
    
    # Try extracting last part after dash (emotional datasets)
    parts = item_name.rsplit("-", 1)
    if len(parts) == 2:
        wav_basename = parts[1]
        if wav_basename in wav_index:
            txt_path = wav_index[wav_basename].with_suffix(".txt")
            if not txt_path.exists():
                txt_path.write_text(text)
            matched += 1
            continue
    
    # Try full path reconstruction for emotional datasets
    # e.g., "CREMA-D-AudioWAV-1001_DFA_ANG_XX" -> CREMA/AudioWAV/1001_DFA_ANG_XX.wav
    for prefix, folder in [("CREMA-D-", "CREMA/"), ("CREMA-", "CREMA/")]:
        if item_name.startswith(prefix):
            rest = item_name[len(prefix):]
            key = prefix.rstrip("-") + "-" + rest
            if key in wav_index:
                txt_path = wav_index[key].with_suffix(".txt")
                if not txt_path.exists():
                    txt_path.write_text(text)
                matched += 1
                break
    else:
        unmatched += 1

print(f"\nMatched: {matched}, Unmatched: {unmatched}")
PYTHON_SCRIPT

cd VccmDataset

# Clean up tar.gz files to save space (optional - comment out to keep)
echo "[6/8] Cleaning up downloaded archives..."
# rm -f TextrolSpeech_data.tar.gz CREMA-D.tar.gz
# rm -f libritts/*.tar.gz

echo "[7/8] Creating directories for MFA outputs..."
mkdir -p mfa_outputs

echo "[8/8] Setup complete!"
echo ""
echo "=========================================="
echo "Summary:"
echo "=========================================="
echo "Audio files extracted to: VccmDataset/audio_extracted/"
echo "Transcripts paired with audio files"
echo ""
echo "Next steps:"
echo "1. Install MFA: conda install -c conda-forge montreal-forced-aligner"
echo "2. Download MFA models:"
echo "   mfa model download acoustic english_mfa"
echo "   mfa model download dictionary english_mfa"
echo "3. Run MFA alignment:"
echo "   mfa align VccmDataset/audio_extracted english_mfa english_mfa VccmDataset/mfa_outputs"
echo "4. Run preprocessing:"
echo "   python -m data_utils.preprocess_parallel --csv_path VccmDataset/controlspeech_train.csv --output_dir processed_data/ --tarball VccmDataset/TextrolSpeech_data.tar.gz --phoneme_vocab_path ."
echo "5. Train:"
echo "   python train.py --config config.yaml"
echo "=========================================="
