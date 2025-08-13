# 🐱 Cat vs Non-Cat Classifier

A simple **binary image classifier** built from scratch using **NumPy** to distinguish between cats and non-cats.  
Trained on a small dataset (`train_catvnoncat.h5`) as part of a deep learning exercise.

---

## 📂 Project Structure
Cat Vs Non-cat/
│
├── data/ # Dataset (.h5 files) & custom images
│ ├── train_catvnoncat.h5
│ ├── test_catvnoncat.h5
│ └── Nymeria.JPG
│
├── notebooks/
│ └── model.ipynb # Jupyter notebook for training & testing
│
├── src/
│ ├── main.py # Script version for training & testing
│ └── model_utils.py # Neural network implementation
│
├── venv/ # Python virtual environment
├── requirements.txt # Dependencies
└── README.md # Project documentation


---

## 🚀 Features
- Implemented from scratch using **NumPy** only (no deep learning frameworks).
- **2-layer neural network** (1 hidden layer + output layer).
- Trains on the provided `.h5` dataset of 64x64 images.
- Works both as:
  - **Python script** (`main.py`)
  - **Jupyter Notebook** (`model.ipynb`)
- Supports prediction on **custom images** (e.g., Nymeria 🐾).

---

## 🛠 Installation
1. **Clone the repository**:
```bash
git clone https://github.com/YOUR_USERNAME/cat-vs-non-cat.git
cd cat-vs-non-cat

2. ** Create a virtual environment
python3 -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows

Nymeria — Custom Prediction Example
Here's a prediction example using my cat, Nymeria 🐱:


Prediction Output:
Prediction for Nymeria: non-cat (0.0289 confidence)
Note: This model is simple and trained on a small dataset — so it may misclassify real-world images like Nymeria.

Cost after iteration 0: 0.6931
Cost after iteration 100: 0.6483
...
Test accuracy: 72.00%
Prediction: cat
