# Seismic-ANN-Detection

Detecting seismic events in lunar (Apollo) and Mars (InSight) seismograms with a neural network. Built for the **NASA Space Apps Challenge 2024**, "Seismic Detection Across the Solar System".

The pipeline reads raw MiniSEED traces, removes noise with a high-pass filter and wavelet denoising, trains a binary classifier per planet, and writes a catalog of predictions for the test traces.

## How it works

1. **Load** MiniSEED traces and the event catalog (`data_loading.py`).
2. **Preprocess** each trace: Butterworth high-pass filter, then wavelet denoising (`preprocessing.py`).
3. **Train** a feed-forward network on the padded signals (`model.py`): Flatten → Dense(100, ReLU) → Dropout(0.5) → Dense(1, sigmoid), Adam optimizer, binary cross-entropy, 20 epochs, batch size 64.
4. **Predict** on the test traces and save a CSV (`test_model.py`).

Lunar and Mars data run through separate workflows (`lunar_workflow.py`, `mars_workflow.py`), each with its own model.

## Repository layout

```
├── main.py                      # Entry point: runs the lunar and Mars workflows
├── lunar_workflow.py            # Lunar training and testing
├── mars_workflow.py             # Mars training and testing
├── model.py                     # build_model()
├── preprocessing.py             # High-pass filter, wavelet denoising
├── data_loading.py              # MiniSEED and catalog loading
├── test_model.py                # Runs a trained model over test traces
├── seismic_noise_tomography.py  # Noise analysis on exported velocity data
├── m.py                         # Helper: export a MiniSEED trace to Excel
├── lunar_velocity_data.xlsx     # Exported lunar velocity trace
├── mars_velocity_data.xlsx      # Exported Mars velocity trace
└── results/
    ├── lunar/                   # Trained model, label encoder, predictions
    └── mars/                    # Predictions
```

## Results

Sample outputs are committed under `results/`:

- `results/lunar/lunar_output.csv`: predictions for 64 lunar test traces
- `results/mars/mars_output.csv`: predictions for 9 Mars test traces
- `results/lunar/lunar_model.h5`: the trained lunar model

Each row has the trace filename, its start time, and the predicted label.

## Running it

Requires Python 3.9+.

```bash
git clone https://github.com/Mohamed8221/Seismic-ANN-Detection.git
cd Seismic-ANN-Detection
pip install -r requirements.txt
```

The dataset is not included. Download the Space Apps 2024 seismic detection data package from the [challenge page](https://www.spaceappschallenge.org/), then **edit the paths at the top of `main.py`** (and the Excel path in `mars_workflow.py` and `m.py`) so they point at your copy. They are currently absolute Windows paths from the original development machine.

```bash
python main.py
```

Predictions are written to `results/lunar/` and `results/mars/`.

## Troubleshooting

- **Model predicts a single class:** the catalog is imbalanced. Try class weights in `model.fit(...)` or a different decision threshold in `test_model.py`.
- **`FileNotFoundError`:** check the data paths in `main.py`.
- **AVX2/FMA warnings from TensorFlow:** informational, safe to ignore.

## Resources

- [NASA Space Apps Challenge](https://www.spaceappschallenge.org/)
- [ObsPy documentation](https://docs.obspy.org/)
- [seismic-noise-tomography](https://github.com/bgoutorbe/seismic-noise-tomography)
