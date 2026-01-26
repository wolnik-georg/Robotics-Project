# 🔬 Spectrogram Training Results Analysis (v16)

## Executive Summary

**UNEXPECTED RESULT**: Spectrograms (32,768 dims) performed **WORSE** than hand-crafted features (80 dims)!

- **v13 (Hand-crafted features)**: 58.97% validation accuracy ✅
- **v16 (Spectrograms)**: 54.05% validation accuracy ❌
- **Regression**: -4.92% (spectrograms are worse!)

This contradicts our hypothesis that more information → better accuracy.

---

## 📊 Experimental Setup

### Dataset Split:
- **Training**: Workspace 2 + Workspace 3 (3,362 samples)
- **Test**: 20% split from training workspaces (841 samples)
- **Validation**: Workspace 1 ONLY (2,505 samples) - **holdout workspace**

### Input Configuration:
- **Mode**: Spectrograms
- **Dimensions**: 128 mels × 256 time bins = **32,768 features**
- **Frequency Range**: 50 Hz - 20,000 Hz (full spectrum)
- **Resolution**: 23.4 Hz/bin (excellent)

---

## 🎯 Key Findings

### 1. **Massive Overfitting Problem** ⚠️

**Best Model (GPU-MLP Medium):**
- Train Accuracy: **98.90%** ✅
- Test Accuracy: **92.51%** ✅ 
- Validation Accuracy: **54.05%** ❌

**Train-Validation Gap: 44.85%** 🚨

**What this means:**
- Model memorizes Workspace 2 & 3 patterns perfectly
- Generalizes okay to test set (same workspaces, different samples)
- **Fails catastrophically on Workspace 1** (different acoustic environment)

### 2. **All Models Suffer From Overfitting**

**Top 10 Models by Validation Accuracy:**

| Rank | Model | Val Acc | Test Acc | Train Acc | Train-Val Gap |
|------|-------|---------|----------|-----------|---------------|
| 1 | GPU-MLP (Medium) | **54.05%** | 92.51% | 98.90% | 44.85% |
| 2 | PCA90+MLP (HighReg) | 53.93% | 92.98% | 98.87% | 44.94% |
| 3 | GPU-MLP (Tuned-GELU) | 53.57% | 91.91% | 97.83% | 44.26% |
| 4 | GPU-MLP (Deep) | 53.13% | 94.89% | 99.29% | 46.15% |
| 5 | Robust-GPU-MLP (Medium) | 53.05% | 93.22% | 98.87% | 45.82% |
| 6 | MLP (Large) | 53.01% | 95.96% | 98.99% | 45.97% |
| 7 | Quantile-MLP (Medium) | 51.58% | 96.20% | 99.17% | 47.59% |
| 8 | GPU-MLP (Medium-HighReg) | 51.54% | 93.46% | 98.66% | 47.12% |
| 9 | PCA15+MLP (Medium) | 51.30% | 93.10% | 98.33% | 47.04% |
| 10 | QDA | 50.90% | 49.82% | 99.85% | 48.95% |

**Pattern**: ALL top models have 44-48% train-validation gaps!

### 3. **Worst Overfitters** (>60% gap):

| Model | Train Acc | Val Acc | Gap |
|-------|-----------|---------|-----|
| PCA+ExtraTrees | 100.00% | 36.73% | **63.27%** |
| SVM (Linear) | 100.00% | 37.64% | **62.36%** |
| Logistic Regression | 100.00% | 38.36% | **61.64%** |
| PCA+RandomForest | 100.00% | 39.88% | **60.12%** |
| Ensemble (RF+SVM) | 99.55% | 40.24% | **59.31%** |

**These models learned NOTHING generalizable!**

### 4. **Best Generalizers** (smallest gap, still >36%):

| Model | Train Acc | Val Acc | Gap |
|-------|-----------|---------|-----|
| K-NN | 85.75% | 49.26% | **36.49%** ✅ (best!) |
| AdaBoost | 91.14% | 48.30% | 42.83% |
| GPU-MLP (Tuned-GELU) | 97.83% | 53.57% | 44.26% |

Even the "best" generalizer (K-NN) has a 36% gap!

---

## 🔍 Root Cause Analysis

### **Why Spectrograms Failed:**

#### 1. **Curse of Dimensionality** 📈

**Feature Space:**
- Hand-crafted features: **80 dimensions**
- Spectrograms: **32,768 dimensions** (410x more!)

**Sample Efficiency:**
- 3,362 training samples / 32,768 features = **0.103 samples per feature** ❌
- Need ~10 samples per feature for reliable learning → Need **327k samples!**
- Current ratio is **100x too small**

**Result**: Models learn workspace-specific noise patterns instead of contact physics

#### 2. **Workspace-Specific Spectrogram Patterns** 🏢

**Spectrograms capture EVERYTHING:**
- ✅ Contact signatures (what we want)
- ❌ Room acoustics (workspace-specific)
- ❌ Robot joint noise (varies by workspace)
- ❌ Background hum (different per location)
- ❌ Microphone position artifacts
- ❌ Ambient reflections

**Hand-crafted features are DESIGNED to ignore these:**
- Normalized spectral features (room-invariant)
- Relative frequency ratios (position-invariant)
- Statistical moments (noise-robust)

#### 3. **High-Dimensional Workspace Memorization** 🧠

With 32k dimensions, models can:
- Encode "Workspace 2 sounds like THIS frequency pattern"
- Encode "Workspace 3 sounds like THAT frequency pattern"
- Perfectly separate within-workspace contact/no-contact

But when they see Workspace 1:
- Frequency patterns are completely different
- No learned rules transfer
- Accuracy drops to ~54% (barely better than random 50%)

#### 4. **Linear Models Collapse Completely** 📉

| Model Type | Train | Val | Explanation |
|------------|-------|-----|-------------|
| SVM (Linear) | 100% | 37.6% | Finds hyperplane separating WS2/3, useless for WS1 |
| Logistic Reg | 100% | 38.4% | Same - linear boundary doesn't transfer |
| Random Forest | 100% | 45.9% | Trees split on workspace-specific frequencies |

**Why hand-crafted features work better:**
- 80 dims allows linear models to find generalizable patterns
- Features are workspace-invariant by design
- Less chance to memorize spurious correlations

---

## 📈 Comparison: Features vs Spectrograms

### **Performance Summary:**

| Metric | Features (v13) | Spectrograms (v16) | Δ |
|--------|----------------|-------------------|---|
| **Best Val Acc** | **58.97%** ✅ | 54.05% ❌ | **-4.92%** |
| **Best Test Acc** | ~95% | 96.20% | +1.2% |
| **Best Train Acc** | ~99% | 99.85% | +0.85% |
| **Min Train-Val Gap** | ~32% | 36.49% | +4.5% worse |

### **What We Learned:**

1. **More features ≠ Better accuracy** (when sample-limited)
2. **Hand-crafted features encode domain knowledge** (workspace-invariance)
3. **Raw spectrograms are too workspace-specific** without normalization
4. **Overfitting is the main problem**, not model capacity

---

## 🤔 Why Test Accuracy is High but Validation is Low?

### **Test Set (92-96% accuracy):**
- Comes from **same workspaces** as training (WS2 + WS3)
- Same acoustic characteristics
- Models can use workspace-specific patterns
- **Not truly independent!**

### **Validation Set (54% accuracy):**
- From **different workspace** (WS1)
- Different room acoustics
- Different robot position
- **True generalization test!**
- **This is the real performance**

**Lesson**: Test accuracy is misleading when not cross-workspace!

---

## 💡 Why Hand-Crafted Features Win

### **Features (80 dims) are Workspace-Invariant:**

1. **Spectral Centroid** → Normalized by total energy (room-invariant)
2. **Spectral Rolloff** → Relative measure (position-invariant)
3. **Spectral Contrast** → Frequency ratios (robust to gain)
4. **MFCCs** → Cepstral coefficients (perceptually-based)
5. **Zero-Crossing Rate** → Time-domain (amplitude-invariant)
6. **Chroma Features** → Pitch classes (octave-invariant)

### **Spectrograms (32k dims) are Workspace-Specific:**

1. **Absolute magnitudes** → Change with mic distance
2. **Frequency bins** → Encode room modes
3. **Time-frequency patterns** → Include background noise
4. **Phase information** → Microphone-specific

**Result**: Features extract physics, spectrograms encode environment

---

## 🚨 Critical Insight: The Real Problem

### **Root Cause**: Training on WS2+3, validating on WS1

**This setup REQUIRES workspace-invariant features!**

Spectrograms CAN'T be workspace-invariant without:
1. **Normalization** (per-workspace mean/std subtraction)
2. **More training workspaces** (10+ different environments)
3. **Data augmentation** (simulate different rooms)
4. **Domain adaptation** techniques

**Current approach:**
- ❌ Raw spectrograms → Memorize WS2/3 acoustics
- ✅ Hand-crafted features → Designed for invariance

---

## 📊 Model Architecture Analysis

### **Neural Networks (MLPs):**
- All show similar behavior: 98-99% train, 92-96% test, 51-54% val
- More regularization (HighReg) helps slightly: 47% gap vs 45% gap
- Deeper networks (Deep, Large) overfit MORE: 46-48% gaps
- **Conclusion**: Architecture tuning can't fix fundamental data problem

### **Tree-Based Models:**
- Random Forest: 100% train, 46% val (overfits hard on leaf patterns)
- XGBoost: 100% train, 48% val (similar issue)
- Gradient Boosting: 98% train, 49% val (slightly better)
- **Conclusion**: Trees split on workspace-specific frequency bins

### **Linear Models:**
- SVM Linear: 100% train, 38% val (finds non-transferable hyperplane)
- Logistic Regression: 100% train, 38% val (same issue)
- **Conclusion**: 32k dims allow perfect linear separation within WS2/3

### **Distance-Based:**
- K-NN: **Best generalizer!** 86% train, 49% val, 36% gap
- Why? Doesn't "learn" patterns, just finds similar samples
- Lower train accuracy = less overfitting
- **But still 36% gap!**

---

## 🎯 Recommendations

### **Option 1: Fix Spectrogram Approach** (Harder)

1. **Normalize spectrograms per-workspace:**
   ```python
   # Subtract per-workspace mean, divide by std
   spec_normalized = (spec - workspace_mean) / workspace_std
   ```

2. **Use relative spectrograms:**
   ```python
   # Divide by mean spectrum (like MFCCs)
   spec_relative = spec / np.mean(spec, axis=1, keepdims=True)
   ```

3. **Add more training workspaces:**
   - Need 5-10 different environments
   - Forces model to learn contact physics, not room acoustics

4. **Data augmentation:**
   - Add room impulse response simulation
   - Vary microphone gain randomly
   - Add different background noises

5. **CNN with regularization:**
   - Use 2D convolutions (exploit time-frequency structure)
   - Heavy dropout (0.5-0.7)
   - L2 regularization
   - Batch normalization

### **Option 2: Improve Hand-Crafted Features** (Easier) ✅

**Current: 59% validation with 80 features**

Add more physics-based features:
- Contact duration (time-domain)
- Impact sharpness (rise time)
- Frequency modulation (FM) analysis
- Harmonic-to-noise ratio
- Spectral flux (change over time)

**Expected: 65-70% validation** with 100-120 features

### **Option 3: Hybrid Approach** (Best?) 🏆

1. **Use hand-crafted features as base** (80 dims)
2. **Add workspace-normalized spectrogram** (after per-workspace z-score)
3. **Use PCA to reduce to 200 total dims**
4. **Train ensemble**

**Expected: 68-75% validation**

---

## 🔬 Experimental Validation Needed

### **Test Hypothesis: "Spectrograms need normalization"**

**Experiment A**: Per-workspace z-score normalization
```python
for workspace in [WS1, WS2, WS3]:
    spec_ws = spectrograms[workspace]
    spec_normalized = (spec_ws - spec_ws.mean()) / spec_ws.std()
```
**Expected**: 60-65% validation (up from 54%)

**Experiment B**: Relative spectrograms
```python
spec_relative = spec / np.mean(spec, axis=0, keepdims=True)
```
**Expected**: 58-62% validation

**Experiment C**: Log-mel with delta features
```python
spec_log = librosa.power_to_db(spec)
spec_delta = librosa.feature.delta(spec_log)
spec_delta2 = librosa.feature.delta(spec_log, order=2)
features = np.concatenate([spec_log, spec_delta, spec_delta2])
```
**Expected**: 62-68% validation

---

## 📉 Statistical Summary

### **Overfitting Severity Distribution:**

| Gap Range | # Models | % of Total |
|-----------|----------|------------|
| 30-40% | 1 | 2% |
| 40-50% | 21 | 42% |
| 50-60% | 23 | 46% |
| 60-70% | 5 | 10% |

**86% of models** have >40% train-val gap!

### **Validation Accuracy Distribution:**

| Accuracy Range | # Models | % of Total |
|----------------|----------|------------|
| 35-40% | 8 | 16% |
| 40-45% | 10 | 20% |
| 45-50% | 11 | 22% |
| 50-55% | 21 | 42% |

**Median validation: 47.6%** (barely better than random 50%)

---

## 🎓 Key Lessons Learned

1. **"More data beats better algorithms"** is TRUE
   - 3,362 samples is NOT enough for 32,768 features
   - Need ~10x more data OR ~100x fewer features

2. **Domain knowledge matters**
   - Hand-crafted features encode "what varies between contact/no-contact"
   - Raw spectrograms encode "what varies between everything and everything"

3. **Test-validation split is critical**
   - Same-workspace test set gives false confidence
   - Cross-workspace validation reveals true generalization

4. **Overfitting can be deceptive**
   - 99% train + 95% test looks great!
   - But 54% validation reveals it's all memorization

5. **Regularization helps, but can't fix data issues**
   - HighReg models: 47% gap vs 45% gap (marginal)
   - Need fundamentally different approach

---

## 🚀 Next Steps (In Order of Priority)

### **Immediate** (stick with features):
1. ✅ Use hand-crafted features (current 59% validation)
2. Add 20-40 more physics-based features
3. Target: **65-70% validation**

### **Short-term** (if needed):
4. Try "both" mode: features + normalized spectrograms
5. Use PCA to reduce dimensionality
6. Target: **68-75% validation**

### **Long-term** (if high accuracy needed):
7. Collect more data (5,000+ samples per workspace)
8. Add 5+ more training workspaces
9. Implement CNN with proper regularization
10. Target: **75-85% validation**

---

## 🏁 Conclusion

**Spectrograms failed NOT because they're bad features, but because:**

1. **Too many dimensions** (32k) for available data (3k samples)
2. **Too workspace-specific** without normalization
3. **Cross-workspace validation** requires invariant features
4. **Hand-crafted features explicitly designed** for this problem

**The data is telling us:**
> "I don't have enough samples to learn 32k features that generalize across workspaces. Give me either MORE DATA or BETTER (fewer, invariant) FEATURES."

**Recommendation**: 
- **Stick with hand-crafted features** (59% → target 70% with improvements)
- OR implement **workspace normalization + hybrid approach** (target 70-75%)
- **Don't use raw spectrograms** without domain adaptation

---

**Bottom Line**: Sometimes less is more. 80 well-designed features > 32,000 raw features (when data-limited).
