# NeuroSign - Traffic Sign Classifier (Tkinter + Scikit-Learn)

A lightweight, modern, and simple Python application for **Traffic Sign Classification** using **Tkinter** and **Scikit-Learn** (No heavy TensorFlow setup required).

## Features
- **8 Traffic Sign Classes**:
 - **10 Traffic Sign Classes**:
  1. 🚸 **Pedestrian Crossing Warning** (Triangular red warning border + human silhouette)
  2. 🛑 **Stop Sign** (Octagonal red sign + STOP text)
  3. ⚠️ **Yield / Give Way Sign** (Inverted red triangle)
  4. 🔞 **Speed Limit 50 km/h** (Red ring circle + 50 text)
  5. 🔞 **Speed Limit 30 km/h** (Red ring circle + 30 text)
  5. ⛔ **No Entry Sign** (Red circle + white horizontal bar)
  6. 🅿️ **No Parking** (Blue circle with red diagonal — parking prohibited)
  6. ➡️ **Turn Right Ahead** (Blue circle + white arrow)
  7. 🔄 **Roundabout Ahead** (Blue circle + 3 circular arrows)
  8. 🚦 **Traffic Signal Ahead** (Triangular warning border + red/yellow/green lights)
- **Scikit-Learn Machine Learning**: Uses a **Random Forest Classifier** trained on HSV Color Histograms, Histogram of Oriented Gradients (HOG), Shape/Contour vertex analysis, and pixel intensity vectors.
- **Out-of-the-Box Synthetic Dataset Generator**: Automatically generates sample training images and 1-click test preset images.
- **NeuroSign Modern GUI**: Built using Tkinter dark mode with confidence bars, top match breakdowns, detected shape/color analysis, description cards, and a retraining dialog.

## How to Run

1. Open your terminal in this directory:
```bash
cd C:\Users\Intel\.gemini\antigravity-ide\scratch\traffic_sign_classifier
```

2. Run the application:
```bash
python app.py
```

3. Features in the App:
   - Click any **Preset Sample Sign** button (e.g. 🚸 Pedestrian, 🛑 Stop) to test instantly.
   - Click **⬆ Upload Image** to test your own custom traffic sign images.
   - Click **🎓 Retrain ML Model** to regenerate the dataset and re-fit the Random Forest classifier.
