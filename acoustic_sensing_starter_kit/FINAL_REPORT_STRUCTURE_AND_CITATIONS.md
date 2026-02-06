# 📝 Final Report Structure & Citation Strategy

**Project:** Acoustic-Based Contact Detection for Robotic Manipulation  
**Author:** Georg Wolnik  
**Date:** February 4, 2026  
**Format:** IEEE Conference Paper (6-8 pages)

---

## 📋 FINAL REPORT STRUCTURE

### **PAGE ALLOCATION (6-8 Pages)**

| Section | Pages | Key Content |
|---------|-------|-------------|
| **Abstract** | 0.1 | Problem, approach, results, contribution |
| **1. Introduction** | 0.7-0.9 | Motivation, problem statement, contributions |
| **2. Related Work** | 0.4-0.5 | Wall et al., VibeCheck (compact & focused) |
| **3. Method** | 1.8-2.2 | Setup, features, ML pipeline, evaluation |
| **4. Results** | 2.5-3.0 | Proof of concept, position/object gen, discoveries |
| **5. Conclusion** | 0.4-0.6 | Summary, contributions, future work |
| **References** | 0.2-0.3 | BibTeX bibliography |
| **TOTAL** | **6.0-7.7** | |

---

## 📚 CITATION STRATEGY

### **Related Work Section (4 citations in text):**

1. ✅ **Wall (2019)** - `wall2019morphological` - PhD thesis, foundational work
2. ✅ **Wall et al. (2022)** - `wall2022passive` - Passive/active acoustic sensing (IJRR)
3. ✅ **Zöller et al. (2020)** - `zoeller2020active` - Active acoustic contact sensing (RA-L)
4. ✅ **Zhang et al. (2025)** - `zhang2025vibecheck` - VibeCheck, robot configuration entanglement

**Optional (if space permits):**
- **Xu & Chan (2023)** - `xu2023soft` - Acoustic sensors for grippers
- **Li & Stuart (2025)** - `li2025acoustac` - AcousTac resonance sensing

---

### **Method Section (cite tools, NOT in Related Work):**

- **librosa** - `mcfee2015librosa` - Cited in Section 3.2 (Feature Engineering)
- **scikit-learn** - `pedregosa2011scikit` - Cited in Section 3.3 (Classification)
- **NumPy** - `harris2020numpy` - Optional, if needed

---

### **Complete Bibliography (all references):**

**Total: 7-10 references**
- 4 core related work (Wall, Zöller, Zhang)
- 2-3 software libraries (librosa, scikit-learn, NumPy)
- 0-3 optional acoustic sensing papers

---

## 🎯 RELATED WORK SECTION (Compact Version)

### **Structure (~0.4-0.5 pages, 3 paragraphs):**

**Paragraph 1: Acoustic Sensing Foundation**
- Wall (2019, 2022) pioneered acoustic sensing for soft pneumatic actuators
- Demonstrated passive acoustic signals encode contact + morphological info
- Zöller et al. (2020) added active excitation for improved SNR

**Paragraph 2: Robot Configuration Entanglement**
- Zhang et al. (2025) VibeCheck: Robot arm config affects vibration signals
- Mechanical coupling creates entanglement between joint state and sensor
- Our work confirms in acoustic domain + provides physics-based explanation
- Shows position generalization **still works** despite entanglement

**Paragraph 3: Our Contribution (1-2 sentences)**
- First demonstration of **geometric reconstruction** (not just contact detection)
- Rigid robot manipulator (not soft actuator)
- 76% accuracy, 70% position generalization despite configuration entanglement

---

## 📊 FIGURE PRIORITIZATION

### **CRITICAL (5 figures minimum):**

| # | Figure | File | Section | Purpose |
|---|--------|------|---------|---------|
| 1 | Setup | `figure6_experimental_setup.png` | 3.1 | Hardware + workflow |
| 2 | Proof | `figure1_v4_vs_v6_main_comparison.png` (V4) | 4.1 | 76% accuracy |
| 3 | Position | `pattern_a_visual_comparison.png` | 4.2 | 70% validation |
| 4 | Object | `pattern_b_visual_comparison.png` | 4.3 | 50% failure |
| 5 | Surface | `figure10_surface_type_effect.png` | 4.4 | +15.6% discovery |

### **IMPORTANT (add if space permits):**

| # | Figure | File | Section | Purpose |
|---|--------|------|---------|---------|
| 6 | Calibration | `figure8_confidence_calibration.png` | 4.5 | Safety analysis |
| 7 | Features | `figure11_feature_dimensions.png` | 3.2 | 80-dim breakdown |

---

## 📝 SECTION DETAILS

### **Abstract (150 words)**

Problem: Geometric contact detection using acoustic sensing in robotic manipulation. Can robot-mounted microphones reconstruct contact geometry and generalize?

Approach: Hand-crafted acoustic features (MFCCs, spectral, temporal, impulse) + Random Forest classifier on Franka Panda robot.

Results: 76.2% contact detection, 70% position generalization, 50% object generalization (random chance).

Contribution: First acoustic geometric reconstruction demo + surface type discovery (+15.6%) + physics-based entanglement theory.

---

### **1. Introduction (0.7-0.9 pages)**

**1.1 Motivation (2-3 paragraphs):**
- Sensors create representations (cameras→images, LiDAR→3D, force→contact maps)
- Can acoustic sensors create representations of what they touch?
- Why acoustic? Non-contact, works with occlusions/transparent objects

**1.2 Contributions (bullet list):**
- ✅ Proof of concept: 76.2% accuracy
- ✅ Position generalization: 70% validation
- ✅ Surface type discovery: +15.6% from geometric complexity
- ✅ Physics theory: Entanglement explains object failure (50%)
- ✅ Open-source pipeline: 73+ visualizations

---

### **2. Related Work (0.4-0.5 pages)**

**Paragraph 1: Acoustic Sensing Foundation**
- Wall~\cite{wall2019morphological, wall2022passive} pioneered acoustic sensing for soft actuators
- Zöller et al.~\cite{zoeller2020active} added active excitation

**Paragraph 2: Robot Configuration Entanglement**
- Zhang et al.~\cite{zhang2025vibecheck} VibeCheck: robot config affects signals
- Our work confirms in acoustic domain + physics explanation
- Position generalization **still works** with proper features

**Paragraph 3: Our Contribution**
- First **geometric reconstruction** (not just contact detection)
- Rigid manipulator (not soft actuator)
- 76% accuracy, 70% position gen despite entanglement

---

### **3. Method (1.8-2.2 pages)**

**3.1 Experimental Setup (0.4 pages + Fig 1):**
- Franka Panda robot + contact microphone
- 4 test objects (A/B/C training, D holdout)
- 4 workspaces (WS1-3 train/val, WS4 holdout)
- Raster sweep: 1cm spacing, ~15k samples

**3.2 Feature Engineering (0.4 pages + Table 1):**
- 80-dim hand-crafted features:
  - MFCCs (39), Spectral (11), Temporal (15), Impulse (15)
- Why hand-crafted? 75% vs 51% for spectrograms
- StandardScaler normalization

**3.3 Classification Pipeline (0.3 pages):**
- Random Forest (100 trees)
- Confidence filtering (threshold=0.9)
- Cite: scikit-learn~\cite{pedregosa2011scikit}

**3.4 Evaluation Strategy (0.4 pages + Fig 2):**
- Pattern A: Train WS2+3, validate WS1 (position gen)
- Pattern B: Train WS1+2+3, holdout WS4 (object gen)

**3.5 Surface Reconstruction (0.2 pages - optional):**
- Convert predictions to 2D spatial maps

---

### **4. Experimental Results (2.5-3.0 pages)**

**4.1 Proof of Concept (0.3 pages + Fig 3):**
- 76.2% test accuracy (>> 50% random)
- <1ms inference (real-time)

**4.2 Position Generalization (0.6 pages + Fig 4):**
- Training: 96% accuracy
- Validation: **70.1% accuracy** ✅
- Well-calibrated (75.8% conf → 75.1% acc)
- Connection to VibeCheck~\cite{zhang2025vibecheck}

**4.3 Object Generalization (0.6 pages + Fig 5):**
- Training: 97% accuracy
- Holdout: **50.6% accuracy** (random chance) ❌
- Overconfident: 92% conf → 50% acc (safety concern!)

**4.4 Surface Type Effect (0.5 pages + Fig 6):**
- Cutout surfaces: **+15.6%** position gen (p<0.001)
- No effect on object gen (p>0.5)

**4.5 Confidence Calibration (0.3 pages + Fig 7 - optional):**
- Position model: Well calibrated ✅
- Object model: Overconfident ❌

**4.6 Physics Interpretation (0.4 pages - optional):**
- Same object → same eigenfrequencies (f_n formula)
- Different objects → different spectra
- Entanglement: Signal = Contact ⊗ Object

---

### **5. Conclusion (0.4-0.6 pages)**

**5.1 Summary:**
- Proof of concept: 76% ✅
- Position gen: 70% ✅
- Object gen: 50% ❌
- Surface type discovery: +15.6%

**5.2 Contributions:**
- First acoustic geometric reconstruction
- Physics-based entanglement theory
- Open-source pipeline

**5.3 Future Work:**
- Test physics predictions
- Domain adaptation for object gen
- Multi-modal fusion (acoustic + vision + force)

---

## 🎯 CONTENT SOURCES

| Section | Primary Source | Secondary Sources |
|---------|----------------|-------------------|
| Abstract | Main research doc | Presentation |
| 1. Introduction | Presentation, Main doc | - |
| 2. Related Work | **BibTeX file** | Physics doc |
| 3.1 Setup | DATA_COLLECTION_PROTOCOL.md | - |
| 3.2 Features | HANDCRAFTED_VS_SPECTROGRAM.md | - |
| 3.3 Pipeline | PIPELINE_GUIDE.md | Main doc |
| 3.4 Evaluation | DATA_SPLIT_STRATEGY_ANALYSIS.md | - |
| 4.1 Proof | Main doc Section 3.1 | Figure 1 |
| 4.2 Position | Main doc Section 3.1 | Pattern A figs |
| 4.3 Object | Main doc Section 3.2 | Pattern B figs |
| 4.4 Surface | DATA_SPLIT_STRATEGY_ANALYSIS.md | Figure 10 |
| 4.5 Calibration | CONFIDENCE_FILTERING_ANALYSIS.md | Figure 8 |
| 4.6 Physics | PHYSICS_FIRST_PRINCIPLES.md | Main doc Sec 9 |
| 5. Conclusion | Main doc Section 10 | - |

---

## ✅ READY TO WRITE!

**All content sources prepared:**
- ✅ BibTeX file updated with all citations
- ✅ Related Work structure defined (compact, 4 citations)
- ✅ Figure prioritization complete (5-7 figures)
- ✅ All experimental results documented
- ✅ Physics interpretation ready

**Estimated writing time: 10-15 hours**

---

## 🚀 NEXT STEPS

**Would you like me to:**

1. **Start drafting the LaTeX report** section by section?
2. **Create figure captions** for all 5-7 figures?
3. **Write the Related Work section first** as a template?
4. **Draft the complete Abstract**?

**Ready to start writing when you give the signal!** 📝
