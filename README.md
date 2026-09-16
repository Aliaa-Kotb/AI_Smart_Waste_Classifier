# ♻️ AI Smart Waste Classifier

An AI-powered computer vision application that automatically classifies waste images into different categories using **Deep Learning, Transfer Learning, and OpenCV**.

The project uses a pretrained **MobileNetV2** model and fine-tunes it for waste classification. A **Streamlit** web application provides an easy-to-use interface where users can upload an image and receive the predicted waste category along with the model's confidence score.

The project is designed with future **Edge AI / Embedded AI** deployment in mind, with the possibility of deploying the trained model on devices such as the **ESP32-S3** or Raspberry Pi.

---

## 📌 Table of Contents

* [Project Overview](#-project-overview)
* [Objectives](#-objectives)
* [Features](#-features)
* [System Architecture](#-system-architecture)
* [Project Workflow](#-project-workflow)
* [Technologies Used](#-technologies-used)
* [Dataset](#-dataset)
* [Project Structure](#-project-structure)
* [Installation](#-installation)
* [Dataset Preparation](#-dataset-preparation)
* [Training the Model](#-training-the-model)
* [Model Evaluation](#-model-evaluation)
* [Making Predictions](#-making-predictions)
* [Running the Web Application](#-running-the-web-application)
* [Example](#-example)
* [Performance](#-performance)
* [Future Improvements](#-future-improvements)
* [Edge AI Deployment](#-edge-ai-deployment)
* [Applications](#-applications)
* [Limitations](#-limitations)
* [Contributing](#-contributing)
* [License](#-license)
* [References](#-references)
* [Author](#-author)

---

# 🔎 Project Overview

Waste sorting is an important step in recycling and environmental management. Manual waste classification can be time-consuming and inconsistent.

This project applies **Computer Vision and Machine Learning** to automatically identify the category of a waste item from an image.

The initial version focuses on three categories:

* 🧴 Plastic
* 📄 Paper
* 🔩 Metal

The system accepts an image as input, preprocesses it using OpenCV, passes it through a trained MobileNetV2-based classifier, and returns the predicted category.

### Example

```text
Input Image
     │
     ▼
Image Preprocessing
     │
     ▼
MobileNetV2
     │
     ▼
Waste Classification
     │
     ▼
Plastic - 96.4%
```

---

# 🎯 Objectives

The main objectives of this project are:

1. Build an image-based waste classification system.
2. Apply Computer Vision techniques using OpenCV.
3. Use Deep Learning for image classification.
4. Apply Transfer Learning using MobileNetV2.
5. Evaluate the trained model using standard classification metrics.
6. Build an interactive web application for predictions.
7. Provide recycling recommendations based on the predicted category.
8. Design the system so that it can potentially be deployed on an edge device.

---

# ✨ Features

## Current Features

* 📷 Image-based waste classification
* 🧠 Deep Learning classification
* 🔄 Transfer Learning with MobileNetV2
* 🖼️ Image preprocessing using OpenCV
* 📊 Model evaluation
* 📈 Accuracy and loss visualization
* 🎯 Confidence score for predictions
* 🌐 Streamlit web interface
* ♻️ Recycling recommendations

## Planned Features

* 🎥 Real-time webcam classification
* 📱 Mobile-friendly interface
* 🗑️ Additional waste categories
* 📊 Prediction history
* 📈 Classification statistics
* ⚡ TensorFlow Lite optimization
* 🔌 ESP32-S3 deployment
* 🖥️ Edge AI inference

---

# 🏗️ System Architecture

```text
                    ┌──────────────────┐
                    │   User Image     │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │     OpenCV       │
                    │ Image Processing │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │   MobileNetV2    │
                    │  Transfer Model  │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Classification   │
                    └────────┬─────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
          🧴 Plastic       📄 Paper       🔩 Metal
              │              │              │
              └──────────────┼──────────────┘
                             ▼
                    ┌──────────────────┐
                    │ Confidence Score │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Recycling Advice │
                    └──────────────────┘
```

---

# 🔄 Project Workflow

The complete workflow is:

```text
Dataset
   │
   ▼
Data Collection
   │
   ▼
Data Cleaning
   │
   ▼
Image Preprocessing
   │
   ▼
Data Augmentation
   │
   ▼
Transfer Learning
   │
   ▼
Model Training
   │
   ▼
Model Evaluation
   │
   ▼
Save Trained Model
   │
   ▼
Prediction
   │
   ▼
Streamlit Application
```

---

# 🛠️ Technologies Used

| Technology   | Purpose                   |
| ------------ | ------------------------- |
| Python       | Main programming language |
| OpenCV       | Image processing          |
| NumPy        | Numerical operations      |
| Pandas       | Data handling             |
| TensorFlow   | Deep Learning framework   |
| Keras        | Neural network API        |
| MobileNetV2  | Transfer learning model   |
| Scikit-learn | Evaluation metrics        |
| Matplotlib   | Visualization             |
| Streamlit    | Web application           |

---

# 📦 Dataset

The initial implementation can use the **TrashNet** dataset.

TrashNet is a waste image dataset containing several categories of common household waste.

For the first version of this project, the following three classes are recommended:

```text
Plastic
Paper
Metal
```

The original dataset contains additional categories that can be added later.

### Dataset Source

TrashNet:

https://github.com/garythung/trashnet

---

# 📁 Project Structure

```text
AI_Waste_Classifier/
│
├── dataset/
│   │
│   ├── train/
│   │   ├── plastic/
│   │   ├── paper/
│   │   └── metal/
│   │
│   ├── validation/
│   │   ├── plastic/
│   │   ├── paper/
│   │   └── metal/
│   │
│   └── test/
│       ├── plastic/
│       ├── paper/
│       └── metal/
│
├── models/
│   └── waste_classifier.keras
│
├── src/
│   ├── train.py
│   ├── predict.py
│   └── preprocess.py
│
├── results/
│   ├── confusion_matrix.png
│   ├── accuracy.png
│   └── loss.png
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 💻 Installation

## 1. Clone the Repository

```bash
[git clone https://github.com/Aliaa-Kotb/Ai_Smart_Waste_Classifier.git]
Move into the project directory:

```bash
cd Ai_Smart_Waste_Classifier
```

---

## 2. Create a Virtual Environment

Using Python:

```bash
python -m venv venv
```

Activate the environment on Windows:

```bash
venv\Scripts\activate
```

You should see:

```text
(venv)
```

before your terminal prompt.

---

## 3. Install Dependencies

Create a `requirements.txt` file containing:

```text
tensorflow
opencv-python
numpy
pandas
matplotlib
scikit-learn
streamlit
pillow
```

Then install:

```bash
pip install -r requirements.txt
```

---

# 📂 Dataset Preparation

Organize your dataset as follows:

```text
dataset/
│
├── train/
│   ├── plastic/
│   ├── paper/
│   └── metal/
│
├── validation/
│   ├── plastic/
│   ├── paper/
│   └── metal/
│
└── test/
    ├── plastic/
    ├── paper/
    └── metal/
```

Each folder should contain images belonging to that category.

Example:

```text
train/plastic/bottle1.jpg
train/plastic/bottle2.jpg
train/paper/paper1.jpg
train/metal/can1.jpg
```

---

# 🧹 Image Preprocessing

Images are processed before being passed to the neural network.

The preprocessing pipeline includes:

```text
Original Image
      │
      ▼
Resize
224 × 224
      │
      ▼
Pixel Normalization
      │
      ▼
Data Augmentation
      │
      ▼
Model Input
```

Possible augmentation techniques include:

* Rotation
* Horizontal flipping
* Zooming
* Translation
* Small brightness changes

Data augmentation helps the model generalize better to images that were not present in the training dataset.

---

# 🧠 Model

## MobileNetV2

The project uses **MobileNetV2** as the base model.

MobileNetV2 is a lightweight convolutional neural network designed for efficient image recognition and is particularly suitable for applications where computational resources are limited.

Instead of training the entire neural network from scratch, transfer learning is used.

### Architecture

```text
                    Input Image
                    224 × 224
                         │
                         ▼
                  MobileNetV2
                Pretrained Weights
                         │
                         ▼
                 Global Average
                    Pooling
                         │
                         ▼
                    Dropout
                         │
                         ▼
                  Dense Layer
                         │
                         ▼
                 Softmax Output
                         │
              ┌──────────┼──────────┐
              ▼          ▼          ▼
           Plastic     Paper      Metal
```

---

# 🏋️ Training the Model

The training process consists of:

1. Loading the dataset.
2. Resizing images.
3. Applying data augmentation.
4. Loading pretrained MobileNetV2.
5. Freezing the pretrained layers.
6. Adding a new classification head.
7. Training the classification layers.
8. Evaluating validation performance.
9. Fine-tuning selected layers.
10. Saving the final model.

Run:

```bash
python src/train.py
```

The trained model should be saved in:

```text
models/waste_classifier.keras
```

---

# 📊 Model Evaluation

The model should be evaluated using:

### Accuracy

Measures the percentage of correctly classified images.

```text
Accuracy =
Correct Predictions / Total Predictions
```

### Precision

Measures how many predictions of a particular class were correct.

### Recall

Measures how many samples belonging to a class were correctly identified.

### F1-Score

Provides a balance between precision and recall.

### Confusion Matrix

A confusion matrix helps identify which waste categories are being confused with each other.

Example:

```text
                  Predicted
              Plastic Paper Metal

Actual Plastic   92     5     3
       Paper      4    91     5
       Metal      2     6    92
```

---

# 📈 Training Visualization

The project should generate graphs showing:

### Training vs Validation Accuracy

```text
Accuracy
   │
   │             ╭────────
   │          ╭──╯
   │       ╭──╯
   │    ╭──╯
   │────╯
   └────────────────────────
              Epochs
```

### Training vs Validation Loss

```text
Loss
 │╲
 │ ╲
 │  ╲____
 │       ╲____
 │
 └────────────────────────
             Epochs
```

These graphs can be used to detect:

* Overfitting
* Underfitting
* Training instability
* Model convergence

---

# 🔮 Making Predictions

After training, a new image can be classified using:

```bash
python src/predict.py --image path/to/image.jpg
```

Example:

```text
Image: bottle.jpg

Prediction: Plastic
Confidence: 96.4%
```

Another example:

```text
Image: newspaper.jpg

Prediction: Paper
Confidence: 93.1%
```

---

# 🌐 Running the Web Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application provides an interface where the user can:

1. Upload an image.
2. View the image.
3. Process the image.
4. Run the trained model.
5. View the predicted class.
6. View the confidence score.
7. Receive recycling advice.

---

# 🖥️ Application Interface

The planned interface:

```text
╔══════════════════════════════════════╗
║       ♻️ AI Waste Classifier         ║
╠══════════════════════════════════════╣
║                                      ║
║        Upload Waste Image            ║
║                                      ║
║       [ Choose an Image ]            ║
║                                      ║
║              🖼️                      ║
║                                      ║
║        Prediction: Plastic           ║
║        Confidence: 96.4%             ║
║                                      ║
║        ♻️ Recycling Advice           ║
║        Recycle as plastic.           ║
║                                      ║
╚══════════════════════════════════════╝
```

---

# 🧪 Example

### Input

A photograph of a plastic bottle.

### Processing

```text
Image
 ↓
Resize to 224 × 224
 ↓
Normalize
 ↓
MobileNetV2
 ↓
Softmax
```

### Output

```text
Category: Plastic

Confidence: 96.4%

Recommendation:
Place the item in the appropriate plastic
recycling collection.
```

---

# 📊 Performance

Performance metrics should be added after training.

Example format:

| Metric    | Score |
| --------- | ----: |
| Accuracy  |   XX% |
| Precision |   XX% |
| Recall    |   XX% |
| F1-Score  |   XX% |

> **Note:** Do not put example numbers in the final README as actual results. Replace `XX%` with the results obtained from your trained model.

---

# 🚀 Future Improvements

Several improvements can be added to make the project more advanced.

## 1. More Waste Categories

Expand the classifier to include:

```text
Plastic
Paper
Metal
Glass
Cardboard
Trash
Organic Waste
```

---

## 2. Real-Time Camera Classification

Use a webcam with OpenCV:

```text
Webcam
  ↓
Video Frame
  ↓
OpenCV
  ↓
MobileNetV2
  ↓
Prediction
```

---

## 3. Object Detection

Instead of classifying an image containing one object, use an object detection model to identify multiple objects.

Example:

```text
┌───────────────────────────────────┐
│                                   │
│   ┌─────────┐                     │
│   │ Bottle  │ → Plastic 95%       │
│   └─────────┘                     │
│                                   │
│             ┌────────┐            │
│             │  Can   │ → Metal 91%│
│             └────────┘            │
│                                   │
└───────────────────────────────────┘
```

---

## 4. Mobile Application

The model could eventually be integrated into an Android application.

---

## 5. Edge AI Deployment

The model can be optimized for deployment on resource-constrained devices.

Possible platforms include:

* Raspberry Pi
* ESP32-S3
* Other TinyML-compatible microcontrollers

---

# ⚡ Edge AI Deployment

A future version of this project can move inference from a computer to an embedded device.

### Current System

```text
Camera
  ↓
Computer
  ↓
Python
  ↓
TensorFlow
  ↓
Prediction
```

### Future Edge AI System

```text
Camera
  ↓
ESP32-S3
  ↓
Preprocessing
  ↓
Optimized ML Model
  ↓
On-Device Inference
  ↓
Prediction
```

The model may need to be converted and optimized using techniques such as:

* TensorFlow Lite
* Quantization
* Model compression
* Reduced input resolution

The goal is to perform inference locally without requiring a cloud AI service.

---

# 🔐 Privacy

One advantage of an edge-based version is that images can potentially be processed locally.

Instead of:

```text
Camera → Cloud → AI Model
```

the system can use:

```text
Camera → Local Device → AI Model
```

This can reduce the need to transmit images to external servers.

---

# 🌍 Applications

The project could be adapted for:

* Smart recycling bins
* Waste sorting systems
* Educational applications
* Recycling centers
* Smart homes
* Environmental monitoring
* Smart cities
* Automated waste management

---

# ⚠️ Limitations

The initial system has several limitations:

* Classification performance depends heavily on the quality of the dataset.
* Similar-looking waste categories can be difficult to distinguish.
* Images containing multiple objects may reduce classification accuracy.
* Poor lighting can affect predictions.
* The initial model only supports a limited number of categories.
* Edge deployment requires additional model optimization and hardware testing.

---

# 📌 Project Roadmap

```text
[x] Project design
 │
 ▼
[x] Dataset selection
 │
 ▼
[ ] Dataset preparation
 │
 ▼
[ ] OpenCV preprocessing
 │
 ▼
[ ] MobileNetV2 implementation
 │
 ▼
[ ] Model training
 │
 ▼
[ ] Model evaluation
 │
 ▼
[ ] Prediction script
 │
 ▼
[ ] Streamlit application
 │
 ▼
[ ] Webcam support
 │
 ▼
[ ] TensorFlow Lite conversion
 │
 ▼
[ ] Edge AI optimization
 │
 ▼
[ ] ESP32-S3 deployment
```

---

# 🤝 Contributing

Contributions are welcome.

To contribute:

```bash
git clone https://github.com/YOUR_USERNAME/AI-Waste-Classifier.git
```

Create a new branch:

```bash
git checkout -b feature/new-feature
```

Make your changes and commit:

```bash
git add .
git commit -m "Add new feature"
```

Push the branch:

```bash
git push origin feature/new-feature
```

Then create a Pull Request.

---

# 📄 License

This project is released under the **MIT License**.

See the `LICENSE` file for more information.

---

# 📚 References

### Dataset

**TrashNet — A Dataset of Trash Images**

https://github.com/garythung/trashnet

### TensorFlow

TensorFlow documentation:

https://www.tensorflow.org/

### Transfer Learning

TensorFlow — Transfer Learning and Fine-Tuning:

https://www.tensorflow.org/tutorials/images/transfer_learning

### MobileNetV2

TensorFlow/Keras MobileNetV2 documentation:

https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV2

### OpenCV

OpenCV documentation:

https://docs.opencv.org/

### Streamlit

Streamlit documentation:

https://docs.streamlit.io/

---

# 👩‍💻 Author

**Aliaa Kotb**

Machine Learning Enthusiast | Aspiring Engineer

Interested in:

* Artificial Intelligence
* Machine Learning
* Computer Vision
* Embedded Systems
* Edge AI
* TinyML

---

# ⭐ Acknowledgments

* TensorFlow and Keras for the Deep Learning framework.
* OpenCV for image processing.
* The TrashNet dataset for waste image data.
* Streamlit for the interactive application framework.

---

## ⭐ If you find this project useful

Consider giving the repository a ⭐ on GitHub!
