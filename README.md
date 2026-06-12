# ✋ Deep Gesture
**Industrial-Grade Gesture Control Engine for macOS**

Deep Gesture is a high-performance computer vision engine that transforms a standard webcam into a precision neural pointing device. Built from the ground up for the hackathon stage, it moves beyond simple point-tracking by implementing a custom Deep Neural Network (MLP) trained on 65-dimensional geometric invariants.

`DEEPGESTURE://NEURAL_LINK v6.0 -- SYSTEM ONLINE`

---

## 🚀 Key Features
- **Deep ML Engine:** A 4-layer MLP (128-64-32) implemented in pure NumPy, featuring Leaky ReLU activation and L2 regularization.
- **Precision Tracking:** Uses a 65-d feature vector (finger angles, inter-tip gaps, and palm distances) for ultra-stable, rotation-invariant control.
- **Aggressive Training:** Built-in training pipeline with 200x synthetic data augmentation and 1000-epoch gradient-stabilized optimization.
- **Cyberpunk Neon HUD:** A hardware-accelerated, gesture-reactive UI that provides real-time feedback with zero latency.
- **Industrial Architecture:** Decoupled design with support for headless operation and live camera switching (FaceTime HD vs. Continuity Camera).

---

## 🎮 Control Scheme

| Gesture | Action | Description |
| :--- | :--- | :--- |
| **2 Fingers (Index+Mid)** | **Move Pointer** | Midpoint-based tracking with 1-Euro smoothing. |
| **1 Finger (Index Tap)** | **Left Click** | Immediate "air tap" to select applications or buttons. |
| **4 Fingers (Palm)** | **Right Click** | Brief pose triggers a standard context menu click. |
| **4 Fingers + Motion** | **Swipe** | Fast horizontal motion switches Spaces or Tabs. |
| **Fist** | **Hold / Drag** | Closes hand to grab; opens hand to release/drop. |

---

## 🛠️ Technical Stack
- **Vision:** MediaPipe Tasks API (Hand Landmarker)
- **Engine:** NumPy (Deep MLP + Linear Algebra)
- **Input Synthesis:** PyAutoGUI + Native macOS Automation
- **UI:** OpenCV + Neon HUD Animation Layer
- **Environment:** Python 3.9+

---

## ⚡ Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the Engine
```bash
python3 virtual_mouse.py
```

### 3. Industrial Training (Highly Recommended)
Since the engine uses high-dimensional geometric features, we recommend a 30-second "Baking" session for your specific hand:
1. Press **`t`** to enter training mode.
2. Hold a unique pose and press **`1`** repeatedly (capture 10-20 angles).
3. Press **`t`** again to trigger the **1000-epoch Industrial Training cycle**.
4. The model is now custom-fitted to your hand with 200x synthetic augmentation.

---

## 📐 Accuracy Engineering
Deep Gesture minimizes false positives through **Temporal Consensus (Voting)**. The engine maintains a sliding window of predictions and requires a **3/5 majority consensus** before any action is fired on the OS bus. This eliminates "flicker" and accidental triggers.

---

## 🔗 Repository
[https://github.com/saitarrun/DeepGesture](https://github.com/saitarrun/DeepGesture)

Built with ❤️ for the Hackathon.
