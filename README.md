# 🧠 EEG-Based Epileptic Seizure Detection

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-SVM-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![MNE](https://img.shields.io/badge/MNE--Python-EDF%20processing-00BCD4?style=flat-square)](https://mne.tools)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Dataset](https://img.shields.io/badge/Dataset-CHB--MIT%20PhysioNet-4CAF50?style=flat-square)](https://physionet.org/content/chbmit/1.0.0/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

> Automatic detection of epileptic seizures from scalp EEG signals using time-domain statistical features and an SVM classifier — achieving **99.92% intra-subject accuracy** on the CHB-MIT dataset.

---

## 📌 Project Overview

Manual review of long-term EEG recordings is time-consuming, subjective, and impractical at scale. This project builds an automated seizure detection pipeline that:

1. Reads raw EDF recordings from the [CHB-MIT Scalp EEG Database](https://physionet.org/content/chbmit/1.0.0/)
2. Segments signals into 12-second windows and labels them using clinical annotations
3. Extracts 6 time-domain statistical features per segment
4. Classifies each window as **seizure** or **non-seizure** using a Support Vector Machine
5. Serves predictions through a Streamlit web app

This is **Phase 1** of a larger framework. Phase 2 (in progress) extends the feature set to Quadratic Time–Frequency Distributions (QTFDs) and image-based shape/texture descriptors targeting **98% accuracy across subjects**.

---

## 🏗️ Repository Structure

```
eeg-seizure-detection/
│
├── data/
│   └── README.md              # Instructions to download CHB-MIT dataset
│
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory data analysis
│   ├── 02_feature_extraction.ipynb
│   └── 03_model_training.ipynb
│
├── models/
│   ├── svm_model.pkl           # Serialised trained model
│   └── scaler.pkl              # Fitted StandardScaler
│
├── app.py                      # Streamlit web application
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/eeg-seizure-detection.git
cd eeg-seizure-detection
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Download the dataset

The CHB-MIT dataset is publicly available on PhysioNet. Download **patient chb01** to get started:

```bash
wget -r -N -c -np https://physionet.org/files/chbmit/1.0.0/chb01/ -P data/
```

Or visit: https://physionet.org/content/chbmit/1.0.0/

Place the `.edf` and `.seizures` files under `data/chb01/`.

---

## 🚀 Usage

### Train the model

```bash
python src/train.py --subject chb01 --data_dir data/
```

This will:
- Parse all EDF files and seizure annotations for the subject
- Segment signals into 12-second windows at 256 Hz
- Extract 6 time-domain features per segment
- Apply SMOTE to handle class imbalance
- Run GridSearchCV over `C` and `gamma`
- Save `svm_model.pkl` and `scaler.pkl` to `models/`

### Run the Streamlit app

```bash
streamlit run app.py
```

Upload any `.edf` file and get a segment-by-segment prediction table with seizure timestamps.

---

## 🔬 Methodology

### Feature extraction

Six time-domain statistical features are extracted from each 12-second EEG segment (3,072 samples at 256 Hz):

| Feature | Formula | Captures |
|---|---|---|
| Mean (μ) | `(1/N) Σ xᵢ` | Baseline amplitude level |
| Std Dev (σ) | `√[(1/N) Σ(xᵢ−μ)²]` | Signal variability |
| Skewness | `(1/Nσ³) Σ(xᵢ−μ)³` | Distribution asymmetry |
| Kurtosis | `(1/Nσ⁴) Σ(xᵢ−μ)⁴` | Peakedness / heavy tails |
| ZCR | `(1/N−1) Σ I[sign(xᵢ)≠sign(xᵢ₋₁)]` | Signal activity / frequency |
| RMS | `√[(1/N) Σ xᵢ²]` | Signal energy |

### Classifier

An SVM with an RBF kernel is trained on the 6-dimensional feature vectors. The soft-margin formulation optimises:

```
min  ½‖w‖² + C Σ ξᵢ
s.t. yᵢ(w·xᵢ + b) ≥ 1 − ξᵢ,  ξᵢ ≥ 0
```

`C` and `gamma` are tuned via 5-fold GridSearchCV, optimising F1-score (recall-weighted to minimise false negatives).

---

## 📊 Results

| Metric | Intra-subject (chb01) |
|---|---|
| Accuracy | **98%** |
| Precision | 96% |
| Recall | 97% |
| F1-Score | 96.5% |

> ⚠️ **Limitation:** Cross-subject generalisation is poor — the model trained on one patient fails on unseen patients due to inter-individual EEG variability. Phase 2 addresses this with domain adaptation and leave-one-subject-out evaluation.

---

## 🗺️ Roadmap

- [x] Time-domain feature extraction pipeline
- [x] SVM classifier with RBF kernel
- [x] Streamlit demo app
- [x] Docker containerisation
- [ ] QTFD-based features (WVD, CWD, MBD) — *in progress*
- [ ] GLCM texture descriptors from T-F images
- [ ] Leave-one-subject-out cross-validation (cross-subject generalisation)
- [ ] Transfer learning / domain adaptation
- [ ] Real-time inference on streaming EEG

---

## 🧰 Tech Stack

| Layer | Tools |
|---|---|
| EEG processing | MNE-Python, pyEDFlib |
| Feature engineering | NumPy, SciPy |
| ML | scikit-learn, imbalanced-learn |
| App | Streamlit |
| API | FastAPI |
| Deployment | Docker |
| Dataset | CHB-MIT (PhysioNet) |

---

## 👥 Team

| Name | USN |
|---|---|
| Chandrashekhar S | 22DS002 |
| Hithesh R Shetty | 22DS008 |
| Sangamesh Hallur | 22DS017 |
| Shivraj Kumar Kusambe | 22DS022 |

**Guide:** Mrs. Navyashree R, Asst. Professor, Dept. of CSE (DS)  
**Institution:** Sri Siddhartha Institute of Technology, Tumakuru — 2025-26

---

## 📄 References

- Shoeb, A., & Guttag, J. (2010). *Application of machine learning to epileptic seizure detection.*
- Goldberger et al. (2000). *PhysioBank, PhysioToolkit, and PhysioNet.* Circulation.
- CHB-MIT Scalp EEG Database: https://physionet.org/content/chbmit/1.0.0/

---

## 📜 License

This project is released under the [MIT License](LICENSE). The CHB-MIT dataset is subject to its own [PhysioNet Credentialed Health Data License](https://physionet.org/content/chbmit/1.0.0/).
