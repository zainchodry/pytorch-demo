# 🔥 PyTorch Demo — Learning Deep Learning Step by Step

A hands-on, progressive PyTorch learning project structured into four phases — from raw tensor math to Convolutional Neural Networks (CNNs). Each phase builds on the previous one, making this an ideal reference for anyone starting their deep learning journey with PyTorch.

---

## 📁 Project Structure

```
pytorch-demo/
├── 1_tensors_setups/
│   └── auto.ipynb              # Phase 1: Tensors & Autograd (manual gradient descent)
├── 2_nn.moduls_dataloaders/
│   └── phase2.ipynb            # Phase 2: nn.Module, Optimizers & DataLoaders
├── 3_mnist_project/
│   ├── mnis_project.ipynb      # Phase 3: Fully-connected NN on MNIST (97.46% accuracy)
│   ├── mnist_model.pth         # Saved model weights (git-ignored)
│   └── data/                   # Auto-downloaded MNIST dataset (git-ignored)
├── 4_CNNS_Project/
│   ├── cnns.ipynb              # Phase 4: Convolutional Neural Network on MNIST (98.47% accuracy)
│   ├── simple_cnn_mnist.pth    # Saved CNN model weights (git-ignored)
│   └── data/                   # Auto-downloaded MNIST dataset (git-ignored)
├── venv/                       # Python virtual environment (git-ignored)
├── .gitignore
└── README.md
```

---

## 🗺️ Learning Phases

### Phase 1 — Tensors & Autograd
**Notebook:** [`1_tensors_setups/auto.ipynb`](1_tensors_setups/auto.ipynb)

The foundation of PyTorch. This notebook demonstrates:
- Creating tensors with `requires_grad=True`
- Computing **Mean Squared Error (MSE)** loss manually
- Calling `.backward()` to compute gradients automatically
- Updating a weight parameter `w` using **manual gradient descent**
- Zeroing gradients with `.grad.zero_()` after each update

**Key concept:** Understanding how PyTorch's autograd engine tracks computations and computes gradients without any high-level abstractions.

```python
# Simple example: learn w such that w * x ≈ y
w = torch.tensor(0.0, requires_grad=True)
loss = ((w * x - y_true) ** 2).mean()
loss.backward()
with torch.no_grad():
    w -= lr * w.grad
w.grad.zero_()
```

---

### Phase 2 — `nn.Module`, Optimizers & Interactive Prediction
**Notebook:** [`2_nn.moduls_dataloaders/phase2.ipynb`](2_nn.moduls_dataloaders/phase2.ipynb)

Steps up from raw autograd to PyTorch's high-level building blocks:
- Defining custom models with `nn.Module` and `nn.Linear`
- Using built-in loss functions: `nn.MSELoss()`
- Using optimizers: `optim.Adam` and `optim.SGD`
- Training loops with `optimizer.zero_grad()` → `loss.backward()` → `optimizer.step()`
- **Interactive prediction loop** — user types a number and the trained model predicts its double

**Two experiments:**
1. `MyModel` — simple `y = 2x` regression with Adam, 10 epochs
2. `MultiplyModel` — same task with SGD over 2000 epochs + interactive user inference

---

### Phase 3 — MNIST Digit Classifier (Fully-Connected NN)
**Notebook:** [`3_mnist_project/mnis_project.ipynb`](3_mnist_project/mnis_project.ipynb)

First real computer vision project using the MNIST handwritten digit dataset:
- Auto-downloads MNIST with `torchvision.datasets.MNIST`
- Normalizes images with `transforms.ToTensor()`
- Batches data with `DataLoader` (batch size 64)
- Defines a 3-layer **fully-connected network** (`DigitModel`):
  - `Linear(784 → 128) → ReLU → Linear(128 → 64) → ReLU → Linear(64 → 10)`
- Trains with `CrossEntropyLoss` + `Adam` optimizer (lr=0.001) for 5 epochs
- Evaluates on the test set and saves the model
- Visualizes a sample prediction with `matplotlib`

**Results:**
| Metric | Value |
|---|---|
| Training Epochs | 5 |
| Final Test Accuracy | **97.46%** |
| Saved Model | `mnist_model.pth` |

---

### Phase 4 — CNN on MNIST
**Notebook:** [`4_CNNS_Project/cnns.ipynb`](4_CNNS_Project/cnns.ipynb)

Introduces Convolutional Neural Networks for image classification:
- Same MNIST dataset and DataLoader setup as Phase 3
- Defines `SimpleCNN` with two convolutional blocks:
  - `Conv2d(1→8, 3×3) → ReLU → MaxPool2d(2)`
  - `Conv2d(8→16, 3×3) → ReLU → MaxPool2d(2)`
  - Classifier: `Linear(16×7×7→128) → ReLU → Linear(128→10)`
- Trains for only **3 epochs** with Adam (lr=0.001)
- Saves model weights to `simple_cnn_mnist.pth`

**Results:**
| Metric | Value |
|---|---|
| Training Epochs | 3 |
| Final Test Accuracy | **98.47%** |
| Saved Model | `simple_cnn_mnist.pth` |

> **Observation:** The CNN achieves higher accuracy in fewer epochs compared to the fully-connected network, demonstrating the power of spatial feature extraction.

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.11+
- `pip`

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/pytorch-demo.git
cd pytorch-demo
```

### 2. Create & Activate Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows
```

### 3. Install Dependencies

```bash
pip install torch torchvision matplotlib notebook
```

> **Note:** For GPU support, visit [pytorch.org](https://pytorch.org/get-started/locally/) and select the appropriate CUDA version for your system.

### 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

Then open any notebook from the browser interface.

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `torch` | Core PyTorch framework, tensors, autograd, nn |
| `torchvision` | MNIST dataset, image transforms |
| `matplotlib` | Visualizing digit images and predictions |
| `notebook` | Running Jupyter notebooks |

---

## 🧠 Key PyTorch Concepts Covered

| Concept | Phase |
|---|---|
| Tensors & `requires_grad` | 1 |
| `.backward()` & autograd | 1 |
| `nn.Module` & `nn.Linear` | 2 |
| `nn.MSELoss`, `nn.CrossEntropyLoss` | 2, 3, 4 |
| `optim.SGD`, `optim.Adam` | 2, 3, 4 |
| `DataLoader` & `Dataset` | 3, 4 |
| `torchvision` transforms | 3, 4 |
| Device management (`cpu`/`cuda`) | 3, 4 |
| `model.train()` / `model.eval()` | 3, 4 |
| `torch.no_grad()` | 3, 4 |
| `torch.save()` / `model.state_dict()` | 3, 4 |
| Convolutional layers (`Conv2d`, `MaxPool2d`) | 4 |

---

## 📊 Results Summary

| Phase | Model | Dataset | Test Accuracy | Epochs |
|---|---|---|---|---|
| 3 | Fully-Connected NN (784→128→64→10) | MNIST | **97.46%** | 5 |
| 4 | Simple CNN (Conv→Conv→FC) | MNIST | **98.47%** | 3 |

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 🙋 Author

Built as a personal deep learning learning journal — exploring PyTorch from the ground up.
