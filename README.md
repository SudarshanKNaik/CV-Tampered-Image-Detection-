# 🕵️‍♂️ Self-Supervised Image Tampering Detection using Forensic Cues

## 📌 Project Overview

With the widespread use of image editing tools and social media platforms, digital images are often manipulated through operations such as **splicing, copy–move forgery, and retouching**. These manipulations can be subtle and visually indistinguishable to the human eye, making manual verification unreliable.

This project presents a **self-supervised image tampering detection system** that identifies manipulated images **without using labeled tampering data during training**. Instead of relying on deep learning classifiers, the system leverages **classical image forensics cues** and **unsupervised anomaly detection** to determine whether an image deviates from the statistical characteristics of authentic images.

---

## 🎯 Key Objectives

* Detect whether an image is **authentic (clean)** or **manipulated**
* Avoid the need for labeled tampering data during training
* Provide **explainable forensic evidence**, not just a binary decision
* Use lightweight, interpretable image processing techniques

---

## 🧠 Core Idea

> **Authentic images follow consistent statistical patterns.
> Manipulated images disrupt these patterns.**

The system is trained only on **authentic images** to learn what "normal" image statistics look like. During testing, any image that significantly deviates from this learned distribution is flagged as **tampered**.

This makes the approach:

* **Self-supervised**
* **Dataset-agnostic**
* **Robust to unseen manipulation types**

---

## 🔍 Forensic Features Used

The system extracts multiple forensic cues from each image:

### 🎨 Color Statistics

* Mean, standard deviation, and skewness of RGB channels
* Manipulations often introduce unnatural color distributions

### ✂️ Edge Artifacts

* Edge density and sharpness using gradient analysis
* Splicing and copy–paste operations break natural edge continuity

### 🌫️ Noise Patterns

* Sensor noise residuals
* Edited regions often have inconsistent noise characteristics

---

## 🤖 Anomaly Detection Model

* **Isolation Forest** (unsupervised)
* Trained only on authentic images
* Produces:

  * **Binary decision**: Clean / Manipulated
  * **Anomaly confidence score** (degree of suspicion)

---

## 📊 Output and Explainability

For each test image, the system reports:

* **Tampering decision** (Clean / Manipulated)
* **Anomaly confidence score**
* **Color statistics deviation**
* **Edge inconsistency strength**
* **Noise pattern abnormality**

### Example Output

```
⚠️ Manipulated | score=-0.3271 | color_dev=18.42 | edge_inconsistency=0.0813 | noise_abnormality=12.47 | Tp_034.jpg
```

This ensures the system is **interpretable**, not a black box.

---

## 📁 Project Structure

```
CV/
├── train.py                  # Self-supervised training script
├── detect.py                 # Inference with explainable output
├── requirements.txt          # Dependencies
├── data/
│   ├── raw/
│   │   ├── train/            # Authentic images only
│   │   └── test/             # Test images (clean + tampered)
├── features/
│   ├── color_stats.py
│   ├── edge_artifacts.py
│   └── noise_patterns.py
├── models/
│   └── anomaly_model.py
└── utils/
    └── preprocessing.py
```

---

## ▶️ How to Run

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Train the Model (Self-Supervised)

```bash
python train.py
```

* Uses only images from `data/raw/train`
* Saves trained model to `models/model.pkl`

### 3️⃣ Detect Image Tampering

```bash
python detect.py
```

* Tests the first 25 images by default
* Prints detailed forensic evidence per image

---

## 📚 Dataset Usage

* **Authentic images** are used for training
* **Tampered images** are used only for evaluation
* Ground-truth masks (if available) are **never used during training**

This preserves the integrity of the self-supervised setup.

---

## 🧪 Applications

* Digital image forensics
* Social media misinformation detection
* Journalism and media verification
* Law enforcement and cybercrime analysis
* Academic research in explainable computer vision

---

## 🧠 Key Takeaway

> This project demonstrates that **effective image manipulation detection does not require deep learning or labeled datasets**, and that classical image processing combined with anomaly detection can produce **robust, explainable forensic systems**.

---

## 🚀 Future Enhancements

* Patch-wise tampering localization (heatmaps)
* Ground-truth-based quantitative evaluation
* Automatic forensic report generation
* Comparison with deep learning baselines

---

## 📄 License

This project is open-source and available for academic and research purposes.

---

## 👨‍💻 Author

Sudarshan K Naik

**Repository:** https://github.com/SudarshanKNaik/CV-Tampered-Image-Detection-
