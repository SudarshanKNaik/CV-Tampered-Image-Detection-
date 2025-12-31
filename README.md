## Self-Supervised Image Manipulation Detection

This project detects subtle image manipulations such as:
- Splicing
- Retouching
- Copy-move forgery

### Key Features
- No labeled data
- Forensic image cues
- Unsupervised learning (Isolation Forest)

### Run
```bash
pip install -r requirements.txt
python train.py
python detect.py
