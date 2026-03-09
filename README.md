# DeepTrust — Statistical Attack Dataset Generator
### Ammad Ashraf (22I-2470) | Iteration 1 | FYP 2022-2026

---

## What This Does
Generates a **Statistical Attack Dataset** by applying 4 types of statistical
manipulations to FakeAVCeleb fake video samples. The output is used to test
how well deepfake detectors hold up against real-world noise and compression.

---

## Setup (One Time)

### 1. Install Python libraries
```bash
pip install -r requirements.txt
```

### 2. Install FFmpeg
- **Windows**: Download from https://ffmpeg.org → add `bin/` folder to PATH
- **Linux**: `sudo apt install ffmpeg`
- **Mac**: `brew install ffmpeg`

### 3. Verify setup
```bash
python scripts/run.py --config-check
```

---

## Set Your Dataset Path

Once your FakeAVCeleb download finishes:

Open `scripts/config.py` and update this line:
```python
"input_dir": "./input",  # ← Change to your FakeAVCeleb folder path
```

Or pass it directly when running:
```bash
python scripts/run.py --attack gaussian --input "D:/FakeAVCeleb/FakeVideo-RealAudio"
```

---

## Running the Pipeline

### Test first with 10 videos
```bash
python scripts/run.py --attack gaussian --limit 10
```

### Run a single attack
```bash
python scripts/run.py --attack jpeg
python scripts/run.py --attack gaussian
python scripts/run.py --attack dct
python scripts/run.py --attack histogram
```

### Run all 4 attacks sequentially
```bash
python scripts/run.py --attack all
```

---

## Output Structure
```
output/
├── jpeg_compression/      ← FakeAVCeleb structure preserved inside
├── gaussian_noise/
├── dct_manipulation/
└── histogram_shift/
```

---

## The 4 Statistical Attacks

| Attack | What it does | Key parameter |
|--------|-------------|---------------|
| JPEG Compression | Re-encodes frames at low quality, destroying GAN pixel artifacts | quality=15 |
| Gaussian Noise | Adds camera-like random noise, masking GAN's unnatural perfection | sigma=25 |
| DCT Manipulation | Zeroes out high-frequency DCT coefficients containing GAN fingerprints | keep_fraction=0.2 |
| Color Histogram Shift | Shifts RGB channel distributions to disguise GAN color patterns | shift_range=(-30,30) |

---

## Resuming After Interruption
The pipeline is fully resumable. If it stops:
```bash
python scripts/run.py --attack gaussian   # Just re-run — it picks up where it stopped
```
Progress is tracked in `logs/progress.json`.

---

## How This Feeds Into The Rest of Iteration 1

- **Phase 5 (Integration):** Noor's surrogate model is tested against your 4 attack folders to measure accuracy drop
- **Iteration 2:** The unified multimodal model trains on your attacked data
- **Iteration 3:** Adversarial hardening specifically targets these 4 attack patterns
