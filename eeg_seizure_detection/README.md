# 🧠 EEG Epileptic Seizure Detector

SVM-based seizure detection from EEG `.edf` files.  
Trained on the full **CHB-MIT Scalp EEG** dataset (all subjects).

---

## Quick Start

### 1 — Clone / download this folder
```
eeg_seizure_app/
├── app.py
├── requirements.txt
├── models/
│   └── svm_model.pkl      ← trained model (included)
└── chb20_02.edf           ← demo recording (included)
```

### 2 — Create a virtual environment (recommended)
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
```

### 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### 4 — Run the app
```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## Usage

1. Click **▶ Run on chb20_02.edf** to test with the included recording, **or**
2. Drag-and-drop any `.edf` file into the uploader.

Adjust settings in the sidebar:
| Setting | Default | Description |
|---|---|---|
| Model path | `models/svm_model.pkl` | Path to trained SVM |
| Scaler path | `models/scaler.pkl` | StandardScaler from training (optional) |
| EEG channel index | `0` | Which EEG channel to analyse |
| Segment length | `12 s` | Window size for feature extraction |
| Probability threshold | `0.50` | Seizure detection cutoff |

---

## Output

- **Probability Timeline** — seizure probability per segment, with threshold line
- **Feature Heatmap** — Z-score normalised features across the recording
- **Results Table** — downloadable CSV with per-segment predictions

---

## Optional: add the training scaler

During training (`Cell 14` in the notebook), two files are saved:
```
models/svm_model.pkl    ← ✅ included
models/scaler.pkl       ← add this for best accuracy
```

If `scaler.pkl` is absent, the app falls back to per-file StandardScaler
normalisation (suitable for demo; may shift confidence slightly).

---

## Dependencies

| Package | Purpose |
|---|---|
| `mne` | Read `.edf` EEG files |
| `scikit-learn` | SVM + StandardScaler |
| `scipy` | Skewness / kurtosis features |
| `streamlit` | Web UI |
| `plotly` | Interactive charts |
| `joblib` | Model serialisation |
