"""
Custom gesture recognition — a tiny KNN classifier over hand-landmark features.

The feature vector is translation-, scale-, and rotation-invariant so a trained
gesture is recognized regardless of where the hand is, how big it appears, or
how the wrist is rotated:

  1. translate so the wrist (landmark 0) is the origin
  2. scale by the wrist -> middle-finger-MCP (landmark 9) distance
  3. rotate so that wrist->MCP vector points "up"
  4. flatten the 21 (x, y) points -> a 42-d vector

Training samples live in gestures.json: {"g1": [[...42...], ...], "g2": [...]}.
No heavyweight ML dependency — just NumPy.
"""

import json
import math
import os

import numpy as np

WRIST = 0
MIDDLE_MCP = 9


def landmark_features(lm):
    """lm: list of 21 (x, y). Returns a ~65-d invariant feature vector.
    Includes:
      - Normalized coordinates (42)
      - Finger curl angles (4)
      - Inter-tip distances (10)
      - Tip-to-palm distances (4)
      - Thumb-to-fingertip gaps (4)
    """
    pts = np.asarray(lm, dtype=float)
    # 1. Coordinate Invariance (Standard 42-d)
    wrist = pts[WRIST]
    pts_norm = pts - wrist
    ref = np.linalg.norm(pts_norm[MIDDLE_MCP])
    if ref < 1e-6: ref = 1.0
    pts_norm = pts_norm / ref
    
    ang = math.atan2(pts_norm[MIDDLE_MCP, 1], pts_norm[MIDDLE_MCP, 0])
    rot = -ang - math.pi / 2.0
    c, s = math.cos(rot), math.sin(rot)
    R = np.array([[c, -s], [s, c]])
    pts_norm = pts_norm @ R.T
    base_feat = pts_norm.reshape(-1)

    # 2. Geometric Features
    def get_angle(a, b, c):
        v1 = a - b
        v2 = c - b
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6: return 0.0
        cos = np.dot(v1, v2) / (n1 * n2)
        return math.acos(np.clip(cos, -1, 1))

    # Finger curls (Tip-PIP-MCP)
    curls = []
    for tip, pip, mcp in [(8,6,5), (12,10,9), (16,14,13), (20,18,17)]:
        curls.append(get_angle(pts[tip], pts[pip], pts[mcp]))
    
    # Inter-tip distances (normalized)
    tips = [4, 8, 12, 16, 20]
    tip_pts = pts[tips]
    it_dists = []
    for i in range(len(tips)):
        for j in range(i + 1, len(tips)):
            it_dists.append(np.linalg.norm(tip_pts[i] - tip_pts[j]) / ref)
            
    # Tip-to-wrist distances
    tip_to_wrist = [np.linalg.norm(pts[t] - wrist) / ref for t in [8, 12, 16, 20]]

    return np.concatenate([base_feat, curls, it_dists, tip_to_wrist])


class GestureClassifier:
    def __init__(self, k=3, threshold=0.75):
        self.k = k
        self.threshold = threshold
        self.samples = {}                  # label -> ndarray (n, D)
        self.weights = None                # [W1, b1, W2, b2, W3, b3, W4, b4]
        self.labels_map = []               # list of labels corresponding to output indices
        self.feature_dim = None

    # ---- training data ----------------------------------------------------- #
    def add(self, label, feat):
        feat = np.asarray(feat, dtype=float).reshape(1, -1)
        if self.feature_dim is None:
            self.feature_dim = feat.shape[1]
        elif feat.shape[1] != self.feature_dim:
            # reset if feature dimension changed (due to upgrade)
            self.samples = {}
            self.feature_dim = feat.shape[1]

        if label in self.samples:
            self.samples[label] = np.vstack([self.samples[label], feat])
        else:
            self.samples[label] = feat
        self.weights = None
        return len(self.samples[label])

    def count(self, label):
        return len(self.samples.get(label, []))

    def counts(self):
        return {k: len(v) for k, v in self.samples.items()}

    def clear(self, label=None):
        if label is None:
            self.samples = {}
        else:
            self.samples.pop(label, None)
        self.weights = None

    def train_aggresively(self, epochs=1000, lr=0.001, augment_factor=200):
        """Train a deep MLP with heavy augmentation and regularization."""
        if not self.samples:
            return

        self.labels_map = sorted(self.samples.keys())
        label_to_idx = {l: i for i, l in enumerate(self.labels_map)}
        num_classes = len(self.labels_map)
        D = self.feature_dim

        # 1. Prepare data with heavy augmentation + Negative Sampling
        X_train, y_train = [], []

        for label, samples in self.samples.items():
            idx = label_to_idx[label]
            for s in samples:
                X_train.append(s)
                y_train.append(idx)
                # Augmentation: noise, jitter, and non-linear warp
                for _ in range(augment_factor):
                    noise = np.random.normal(0, 0.015, s.shape)
                    jitter = s * (1.0 + np.random.normal(0, 0.03))
                    # simple non-linear warp (simulate tilt)
                    warp = jitter * (1.0 + np.linspace(-0.05, 0.05, len(s)))
                    X_train.append(warp + noise)
                    y_train.append(idx)

        X = np.array(X_train)
        y = np.array(y_train)
        y_oh = np.zeros((len(y), num_classes))
        y_oh[np.arange(len(y)), y] = 1

        # 2. Deep MLP (D -> 128 -> 64 -> 32 -> num_classes)
        h1, h2, h3 = 128, 64, 32
        def init_w(r, c): return np.random.randn(r, c) * np.sqrt(2.0/r)
        
        W1, b1 = init_w(D, h1), np.zeros(h1)
        W2, b2 = init_w(h1, h2), np.zeros(h2)
        W3, b3 = init_w(h2, h3), np.zeros(h3)
        W4, b4 = init_w(h3, num_classes), np.zeros(num_classes)

        # SGD with Momentum + Weight Decay + Clipping
        v = [np.zeros_like(p) for p in [W1, b1, W2, b2, W3, b3, W4, b4]]
        mom = 0.9
        wd = 0.001 # slightly higher weight decay for stability
        lr = 0.0005 # lower learning rate for deep model

        for epoch in range(epochs):
            p = np.random.permutation(len(X))
            X, y_oh = X[p], y_oh[p]

            for i in range(0, len(X), 64):
                bx, by = X[i:i+64], y_oh[i:i+64]

                # Forward with stability clips
                z1 = np.clip(bx @ W1 + b1, -50, 50); a1 = np.maximum(0.01 * z1, z1)
                z2 = np.clip(a1 @ W2 + b2, -50, 50); a2 = np.maximum(0.01 * z2, z2)
                z3 = np.clip(a2 @ W3 + b3, -50, 50); a3 = np.maximum(0.01 * z3, z3)
                z4 = np.clip(a3 @ W4 + b4, -50, 50)
                
                shift_z4 = z4 - np.max(z4, axis=1, keepdims=True)
                exp = np.exp(np.clip(shift_z4, -20, 20))
                probs = exp / (np.sum(exp, axis=1, keepdims=True) + 1e-10)

                # Backprop
                dz4 = (probs - by) / len(bx)
                dW4 = a3.T @ dz4; db4 = np.sum(dz4, axis=0)

                da3 = dz4 @ W4.T; dz3 = da3 * np.where(z3 > 0, 1.0, 0.01)
                dW3 = a2.T @ dz3; db3 = np.sum(dz3, axis=0)

                da2 = dz3 @ W3.T; dz2 = da2 * np.where(z2 > 0, 1.0, 0.01)
                dW2 = a1.T @ dz2; db2 = np.sum(dz2, axis=0)

                da1 = dz2 @ W2.T; dz1 = da1 * np.where(z1 > 0, 1.0, 0.01)
                dW1 = bx.T @ dz1; db1 = np.sum(dz1, axis=0)

                # Update with aggressive Clipping + Momentum
                grads = [dW1, db1, dW2, db2, dW3, db3, dW4, db4]
                params = [W1, b1, W2, b2, W3, b3, W4, b4]
                
                # Global norm clipping (approximate per-layer)
                for j in range(len(params)):
                    g = np.clip(grads[j], -0.1, 0.1) # Aggressive per-parameter clip
                    if j % 2 == 0: g += wd * params[j] # weight decay
                    v[j] = mom * v[j] - lr * g
                    params[j] += v[j]

            if (epoch + 1) % 200 == 0:
                loss = -np.mean(np.log(probs[np.arange(len(by)), np.argmax(by, axis=1)] + 1e-10))
                print(f"[gesture_ml] Epoch {epoch+1}/{epochs}, Loss: {loss:.6f}")

        self.weights = params
        print(f"[gesture_ml] Training complete. Precision-optimized for {num_classes} classes.")

    # ---- inference --------------------------------------------------------- #
    def predict(self, feat):
        """Return (label, confidence). Uses deep MLP with soft-unknown gating."""
        if not self.samples or not self.weights:
            return ("none", 0.0)

        # Legacy check: current architecture expects 8 values [W1, b1, ..., W4, b4]
        if len(self.weights) != 8:
            self.weights = None
            return ("uncertain", 0.0)

        feat = np.asarray(feat, dtype=float).reshape(1, -1)
        W1, b1, W2, b2, W3, b3, W4, b4 = self.weights
        
        z1 = feat @ W1 + b1; a1 = np.maximum(0.01 * z1, z1)
        z2 = a1 @ W2 + b2; a2 = np.maximum(0.01 * z2, z2)
        z3 = a2 @ W3 + b3; a3 = np.maximum(0.01 * z3, z3)
        z4 = a3 @ W4 + b4
        
        exp = np.exp(z4 - np.max(z4, axis=1, keepdims=True))
        probs = (exp / np.sum(exp, axis=1, keepdims=True))[0]

        idx = np.argmax(probs)
        conf = probs[idx]
        label = self.labels_map[idx]
        
        # In industrial models, we use high confidence gating.
        # If the model is "confused" between multiple classes, return unknown.
        if conf < self.threshold:
            return ("unknown", float(conf))
        
        # Entropy check: if second best is too close, it's a false positive risk
        sorted_probs = np.sort(probs)
        if len(sorted_probs) > 1 and (conf - sorted_probs[-2]) < 0.2:
            return ("uncertain", float(conf))

        return (label, float(conf))

    # ---- persistence ------------------------------------------------------- #
    def save(self, path):
        data = {
            "samples": {k: v.tolist() for k, v in self.samples.items()},
            "labels_map": self.labels_map,
            "weights": [w.tolist() for w in self.weights] if self.weights else None,
            "feature_dim": self.feature_dim
        }
        with open(path, "w") as fh:
            json.dump(data, fh)

    def load(self, path):
        if os.path.exists(path):
            try:
                with open(path) as fh:
                    data = json.load(fh)
                if isinstance(data, dict) and "samples" in data:
                    self.samples = {k: np.asarray(v, dtype=float) for k, v in data["samples"].items()}
                    self.labels_map = data.get("labels_map", [])
                    
                    # Validate dimensions
                    if self.samples:
                        first_key = next(iter(self.samples))
                        self.feature_dim = self.samples[first_key].shape[1]
                        # Target dim is 60 (42 base + 4 curls + 10 tips + 4 palm)
                        if self.feature_dim != 60:
                            print(f"[gesture_ml] wiping incompatible {self.feature_dim}-d samples (need 60-d)")
                            self.samples = {}
                            self.feature_dim = None
                    
                    w_data = data.get("weights")
                    if w_data and len(w_data) == 8:
                        self.weights = [np.asarray(w, dtype=float) for w in w_data]
                    else:
                        self.weights = None
                else:
                    # backward compatibility: old format was just the samples dict
                    self.samples = {}
                    self.weights = None
                    self.feature_dim = None
            except Exception as exc:
                print(f"[gesture_ml] could not load {path}: {exc}")
        return self
