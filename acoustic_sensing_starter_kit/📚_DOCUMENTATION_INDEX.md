# 📚 Documentation Index - Acoustic Contact Detection Research

**Project:** Acoustic-Based Contact Detection for Robotic Manipulation  
**Author:** Georg Wolnik  
**Date:** January 30, 2026  
**Status:** ✅ Ready for Presentation

---

## 🔄 Complete Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ACOUSTIC CONTACT DETECTION PIPELINE                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │  STAGE 1        │    │  STAGE 2        │    │  STAGE 3        │         │
│  │  Data Collection│ → │  Analysis & ML  │ → │  Reconstruction │         │
│  │  🤖 Robot Sweep │    │  📊 Training    │    │  🎯 Geometry    │         │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘         │
│         ✅                     ✅                    ✅                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

| Stage | Status | Main Document |
|-------|--------|---------------|
| **1. Data Collection** | ✅ Complete | [DATA_COLLECTION_PROTOCOL.md](#stage-1-data-collection) |
| **2. Analysis & ML** | ✅ Complete | [RESEARCH_FINDINGS_ACOUSTIC_CONTACT_DETECTION.md](#main-research-document) |
| **3. Reconstruction** | ✅ Documented | [SURFACE_RECONSTRUCTION_PIPELINE.md](#stage-3-reconstruction) |

---

## 🎯 START HERE

### For Presentation / Thesis / Paper
👉 **[RESEARCH_FINDINGS_ACOUSTIC_CONTACT_DETECTION.md](./RESEARCH_FINDINGS_ACOUSTIC_CONTACT_DETECTION.md)** (147 KB)
- **The complete research document** - Everything you need
- Executive summary with balanced findings
- 12-point insight summary for geometric reconstruction
- All experiments, results, and interpretations
- Practical recommendations and future directions

---

## 📖 Quick Navigation Guide

### What Do You Need Right Now?

| **I Need To...** | **Go To** | **Why** |
|------------------|-----------|---------|
| 🎤 **Prepare presentation slides** | [Main Document - Executive Summary](#main-research-document) | Concise overview of all findings |
| 📝 **Get quick insights for geometric reconstruction** | [Main Document - Key Insights Summary](#main-research-document) | 12 one-sentence insights (positive + negative) |
| ✅ **Understand what works (positive results)** | [Main Document - Section 4.0](#main-research-document) | Position-invariant detection (75% success) |
| ❌ **Understand what fails (limitations)** | [Main Document - Section 4.1](#main-research-document) | Object generalization failure (50% random) |
| 🔬 **Explain the entanglement problem** | [Main Document - Section 9](#main-research-document) | Contact ⊗ Object Properties theory |
| 📊 **Show data split strategy effects** | [Main Document - Section 2.5](#main-research-document) | +15.6% from geometric complexity |
| 🗺️ **Visualize predictions as surface maps** | [SURFACE_RECONSTRUCTION_PIPELINE.md](#stage-3-reconstruction) | Stage 3: 2D reconstruction outputs |
| 🧪 **Verify specific experimental claims** | [Verification Documents](#verification-documents) | All 14 experiments checked |
| 💡 **Design future experiments** | [Main Document - Section 10.3](#main-research-document) | Practical recommendations |
| 🔬 **Defend results with physics reasoning** | [PHYSICS_FIRST_PRINCIPLES_INTERPRETATION.md](#physics-interpretation) | Wave mechanics, eigenfrequencies |
| 🔬 **Statistical validation & verification** | [SCIENTIFIC_VERIFICATION_AND_DISCUSSION.md](#verification-documents) | CI, Z-tests, p-values |
| ❓ **Prepare for Q&A / thesis defense** | [SCIENTIFIC_VERIFICATION_AND_DISCUSSION.md](#verification-documents) | Anticipated questions with answers |
| 🤖 **Understand data collection setup** | [DATA_COLLECTION_PROTOCOL.md](#stage-1-data-collection) | Robot, calibration, sweep protocol |

---

## 📚 Document Hierarchy

---

### 🤖 Stage 1: Data Collection {#stage-1-data-collection}

**Purpose:** Robot control, calibration, and acoustic data acquisition

#### [DATA_COLLECTION_PROTOCOL.md](./DATA_COLLECTION_PROTOCOL.md) ⭐ PIPELINE START
- **Hardware Setup:** Franka Panda robot, microphone, end effector
- **Calibration Process:** Single-corner calibration → automatic corner computation
- **Raster Sweep Protocol:** Vertical sweep pattern, 10 points per line, 1cm spacing
- **Recording Protocol:** 5 recordings per position, 200ms settling time
- **Ground Truth Labeling:** Automatic label assignment based on position
- **Output Format:** WAV files + CSV metadata per run

**Key Parameters:**
| Parameter | Value |
|-----------|-------|
| Sweep step | 1 cm |
| Points per line | 10 |
| Recordings per position | 5 |
| Dwell time | 1.15s |
| Lift height | 4 cm |

**Source Files:**
- `calibration_v2.py` - Surface calibration
- `raster_sweep.py` - Main data collection
- `new_record.py` - Acoustic recording
- `ground_truth_hold_out_set.py` - Label assignment

**Use When:** Understanding how data was collected, reproducing experiments

---

### 🗺️ Stage 3: Surface Reconstruction {#stage-3-reconstruction}

**Purpose:** Transform trained ML models into visual 2D surface maps

#### [SURFACE_RECONSTRUCTION_PIPELINE.md](./docs/SURFACE_RECONSTRUCTION_PIPELINE.md) ⭐ PIPELINE END
- **Input:** Trained model from Stage 2 + sweep.csv with spatial coordinates
- **Process:** Feature extraction → Model prediction → Spatial mapping
- **Output:** 6 visualization types (ground truth, predictions, confidence, errors)
- **Dependencies:** Requires `multi_dataset_training` experiment
- **Configuration:** `sweep_dataset` path in experiment_config.yml

**Key Output Files:**
| Visualization | Filename | Description |
|--------------|----------|-------------|
| Ground Truth | `ground_truth_surface.png` | Actual class distribution |
| Predictions | `predicted_surface.png` | Model-predicted map |
| Comparison | `comparison_surface_maps.png` | Side-by-side GT vs Pred |
| Confidence | `confidence_map.png` | Prediction certainty heatmap |
| Errors | `error_map.png` | Misclassification locations |
| Model Grid | `model_comparison_maps.png` | All models compared |

**Source Files:**
- `src/acoustic_sensing/experiments/surface_reconstruction.py` - Main experiment
- `src/acoustic_sensing/models/geometric_reconstruction.py` - Feature modes
- `src/acoustic_sensing/experiments/orchestrator.py` - Execution controller

**Use When:** Visualizing predictions as spatial maps, generating final outputs

---

### 🏆 Stage 2: Analysis & ML (Main Research Document) {#main-research-document}

**[RESEARCH_FINDINGS_ACOUSTIC_CONTACT_DETECTION.md](./RESEARCH_FINDINGS_ACOUSTIC_CONTACT_DETECTION.md)** (147 KB)

**Contains:**
- ✅ Executive Summary (balanced positive/negative)
- ✅ **12 Key Insights for Geometric Reconstruction** (NEW - one sentence each)
- ✅ Section 2.5: Data Split Strategy Effects (+15.6% from cutout surfaces)
- ✅ Section 3.1: Position Generalization SUCCESS (75% accuracy)
- ✅ Section 3.2: Object Generalization FAILURE (50% random)
- ✅ Section 4.0: What Works - Position-Invariant Detection
- ✅ Section 9: The Entanglement Problem (3,500+ words)
- ✅ Section 10: Conclusions (balanced positive/negative)
- ✅ Complete experimental design, results, recommendations

**Quick Sections for Presentation:**

| Section | Content | Use For |
|---------|---------|---------|
| **Executive Summary** | 1-page overview | Opening slide |
| **Key Insights Summary** | 12 one-sentence insights | Quick takeaways slide |
| **Section 3.1 (V4)** | Position generalization: 75% ✅ | Success story |
| **Section 3.2 (V6)** | Object generalization: 50% ❌ | Limitation discussion |
| **Section 4.0** | What works - practical use cases | Positive applications |
| **Section 9** | Entanglement theory | Scientific contribution |
| **Section 10.3** | Recommendations | Future work slide |
| **PHYSICS_FIRST_PRINCIPLES** | Eigenfrequencies, wave physics | "Why does this happen?" |

---

### 🔬 Physics Interpretation {#physics-interpretation}

**Purpose:** First-principles physics explanations for all observations

#### [PHYSICS_FIRST_PRINCIPLES_INTERPRETATION.md](./PHYSICS_FIRST_PRINCIPLES_INTERPRETATION.md) ⭐ NEW - DEEP PHYSICS
- **Core Equations:** Eigenfrequency f_n = (1/2π)√(k_n/m_n), wave propagation, damped oscillations
- **Why Position Generalization Works:** Same object = same eigenfrequencies, amplitude changes preserved
- **Why Object Generalization Fails:** Different objects = completely different eigenfrequency spectra
- **Why Cutout Surfaces Help (+15.6%):** Geometric complexity F(x,y) forces position-invariant learning
- **Entanglement Explained:** Coupled oscillator physics, contact stiffness k_contact vs object stiffness k_object
- **Material Dominance:** ρ, E, G dependencies in acoustic signature
- **Testable Predictions:** 5 physics-based predictions for future experiments

**Use When:**
- Explaining WHY physics makes these results inevitable
- Defending methodology from first principles
- Showing the acoustic problem is fundamentally well-posed (for position) but ill-posed (for object)
- Preparing for physics-based thesis defense questions

---

### �🔍 Verification Documents

**Purpose:** Prove all claims are backed by experimental data

#### [SCIENTIFIC_VERIFICATION_AND_DISCUSSION.md](./SCIENTIFIC_VERIFICATION_AND_DISCUSSION.md) ⭐ NEW - CRITICAL FOR DEFENSE
- **Physics-Based Verification:** All claims verified against raw experimental data
- **First-Principles Analysis:** Acoustic physics explains why results occur
- **Statistical Validation:** Confidence intervals, Z-tests, p-values for all claims
- **Argument Preparation:** What you CAN argue vs should argue carefully vs should avoid
- **Q&A Preparation:** Anticipated questions with physics-based answers
- ✅ Position: 76.19% ± 1.7% (95% CI), Z=16.28, p<0.0001
- ✅ Object: 50.46% (95% CI includes 50%, random chance)
- ✅ Surface effect: 15.6% gap, p<0.0001

**Use When:** 
- Defending results in presentation or thesis defense
- Answering "how do you know this is true?" 
- Explaining results from physics first principles
- Preparing for technical questions

#### [COMPREHENSIVE_EXPERIMENTAL_VERIFICATION.md](./COMPREHENSIVE_EXPERIMENTAL_VERIFICATION.md) (21 KB)
- ✅ All 14 experiments systematically verified
- ✅ Zero contradictions found in main document
- Position generalization: 4 experiments (71.9%-76.2%)
- Object generalization: 9 experiments (all ~50%)
- 3 new findings discovered (CNN 3-way, pure surfaces, data splits)

**Use When:** Reviewer asks "How do you know this is true?"

#### [VERIFICATION_SUMMARY.md](./VERIFICATION_SUMMARY.md) (6.7 KB)
- Quick reference checklist
- Which experiments support which claims
- Summary of verification process

**Use When:** Quick lookup needed

#### [COMPLETE_VERIFICATION_CHECKLIST.md](./COMPLETE_VERIFICATION_CHECKLIST.md) (9.8 KB)
- Detailed step-by-step verification
- Exact file locations for each experiment
- Re-verification procedure

**Use When:** Need to update or re-verify claims

---

### 📊 Analysis Documents

**Purpose:** Deep dives into specific discoveries

#### [DATA_SPLIT_STRATEGY_ANALYSIS.md](./DATA_SPLIT_STRATEGY_ANALYSIS.md) (18 KB) ⭐ NEW DISCOVERY
- **Key Finding:** Surface type affects position (✅ +15.6%) but NOT object (❌ 0%) generalization
- Why cutout surfaces help: Geometric complexity forces position-invariant learning
- Statistical validation: p<0.001 for position, p>0.5 for object
- Design principles for future experiments

**Use When:** Explaining why experimental design matters

**Status:** ✅ Integrated into main document Section 2.5

#### [CONFIDENCE_FILTERING_ANALYSIS.md](./CONFIDENCE_FILTERING_ANALYSIS.md) (13 KB)
- V4 (position): 75.8% confidence → 75.1% accuracy ✅ Well calibrated
- V6 (object): 92.2% confidence → 50.5% accuracy ❌ Overconfident
- Why filtering can't fix object generalization
- Deployment safety implications

**Use When:** Discussing model calibration and uncertainty

---

### 🛠️ Implementation Documents

**Purpose:** Technical details for reproducing work

#### [PIPELINE_GUIDE.md](./PIPELINE_GUIDE.md) (6.0 KB)
- How to run experiments
- File structure and configurations
- Step-by-step execution guide

**Use When:** Running new experiments or reproducing results

#### [CONFIDENCE_FILTERING_IMPLEMENTATION.md](./CONFIDENCE_FILTERING_IMPLEMENTATION.md) (9.1 KB)
- Implementation details
- Thresholds and modes (reject vs default)
- Code examples

**Use When:** Understanding or modifying confidence filtering

#### [NORMALIZATION_IMPLEMENTATION.md](./NORMALIZATION_IMPLEMENTATION.md) (4.0 KB)
- Feature normalization strategy
- StandardScaler usage
- Train/validation split handling

**Use When:** Understanding preprocessing pipeline

---

### 🔬 Specialized Analysis

#### [SPECTROGRAM_RESULTS_ANALYSIS.md](./SPECTROGRAM_RESULTS_ANALYSIS.md) (14 KB)
- Why mel-spectrograms failed (51% accuracy)
- Why hand-crafted features work better (75% accuracy)
- Feature engineering insights

**Use When:** Justifying feature engineering choices

#### [SPECTROGRAM_PARAMETERS_ANALYSIS.md](./SPECTROGRAM_PARAMETERS_ANALYSIS.md) (8.4 KB)
- Parameter tuning experiments
- n_mels, hop_length, window size optimization

**Use When:** Discussing spectrogram alternatives

#### [frequency_band_analysis_results/frequency_band_analysis_report.md](./frequency_band_analysis_results/frequency_band_analysis_report.md)
- Which frequency bands matter most
- Spectral contribution analysis

**Use When:** Explaining acoustic signature characteristics

---

## 🎯 Presentation Preparation Checklist

### Essential Slides You Can Build From Main Document:

- [ ] **Slide 1: Title & Context**  
  → Main Doc: Section 1.1-1.2 (Problem & Why Acoustic Sensing)

- [ ] **Slide 2: Research Question**  
  → Main Doc: Section 1.2 (Research Question box)

- [ ] **Slide 3: Experimental Design**  
  → Main Doc: Section 2.1 (V4 vs V6 comparison table)

- [ ] **Slide 4: Key Results Overview**  
  → Main Doc: Executive Summary or Key Insights Summary

- [ ] **Slide 5: SUCCESS - Position Generalization (75%)**  
  → Main Doc: Section 3.1 & Section 4.0

- [ ] **Slide 6: FAILURE - Object Generalization (50%)**  
  → Main Doc: Section 3.2

- [ ] **Slide 7: Why It Fails - Entanglement**  
  → Main Doc: Section 9 (simplified)

- [ ] **Slide 8: Data Split Discovery (+15.6%)**  
  → Main Doc: Section 2.5 or DATA_SPLIT_STRATEGY_ANALYSIS.md

- [ ] **Slide 9: Practical Recommendations**  
  → Main Doc: Section 10.3

- [ ] **Slide 10: Future Directions**  
  → Main Doc: Section 10.3 (Research Directions)

---

## 🔑 Key Numbers to Remember

**Position Generalization (Same Objects, Different Positions):**
- ✅ **75.1% accuracy** (V4 experiment)
- ✅ **71.9%-76.2% range** across 4 experiments
- ✅ **+15.6% improvement** with geometric complexity (cutout surfaces)
- Confidence: 75.8% (well calibrated)

**Object Generalization (New Object):**
- ❌ **50.5% accuracy** (random chance)
- ❌ **All 9 experiments ~50%** (consistent failure)
- ❌ **92.2% confidence** (severe overconfidence)
- ❌ **0% effect** from surface type diversity

**Dataset:**
- ~15,000 total samples
- 80 hand-crafted acoustic features
- 3 workspaces (positions 1,2,3) + 1 hold-out (position 4)
- Objects A,B,C (training) + Object D (testing)

**Performance Comparison:**
- Hand-crafted features: 75% ✅
- Mel-spectrograms: 51% ❌
- Random Forest best performer
- Real-time capable: <1ms inference

---

## 📌 Quick Links to Key Sections

### Main Document Sections (RESEARCH_FINDINGS_ACOUSTIC_CONTACT_DETECTION.md)

**For Presentation:**
- [Executive Summary](./RESEARCH_FINDINGS_ACOUSTIC_CONTACT_DETECTION.md#executive-summary) - Page 1
- [Key Insights Summary](./RESEARCH_FINDINGS_ACOUSTIC_CONTACT_DETECTION.md#key-insights-summary-acoustic-tactile-sensing-for-geometric-reconstruction) - After executive summary
- [Section 2.1: Experimental Design](./RESEARCH_FINDINGS_ACOUSTIC_CONTACT_DETECTION.md#21-two-critical-experiments) - V4 vs V6
- [Section 2.5: Data Split Effects](./RESEARCH_FINDINGS_ACOUSTIC_CONTACT_DETECTION.md#25-data-split-strategy-effects-why-surface-geometry-matters) - +15.6% discovery
- [Section 3.1: Position Success](./RESEARCH_FINDINGS_ACOUSTIC_CONTACT_DETECTION.md#31-experiment-v4-position-generalization-same-object-) - 75% accuracy
- [Section 3.2: Object Failure](./RESEARCH_FINDINGS_ACOUSTIC_CONTACT_DETECTION.md#32-experiment-v6-object-generalization-different-object) - 50% failure
- [Section 4.0: What Works](./RESEARCH_FINDINGS_ACOUSTIC_CONTACT_DETECTION.md#40-what-works-position-invariant-acoustic-contact-detection-) - Positive findings
- [Section 9: Entanglement](./RESEARCH_FINDINGS_ACOUSTIC_CONTACT_DETECTION.md#9-the-entanglement-problem) - Theory
- [Section 10: Conclusions](./RESEARCH_FINDINGS_ACOUSTIC_CONTACT_DETECTION.md#10-conclusions) - Summary
- [Section 10.3: Recommendations](./RESEARCH_FINDINGS_ACOUSTIC_CONTACT_DETECTION.md#103-practical-recommendations) - Future work

---

## 💡 Tips for Using These Documents

### For Your Presentation:
1. **Start with:** Executive Summary (1 slide)
2. **Add:** 12 Key Insights Summary (1 slide with bullets)
3. **Show success:** Section 3.1 / 4.0 (position generalization 75%)
4. **Show failure:** Section 3.2 (object generalization 50%)
5. **Explain why:** Section 9 (entanglement - simplified)
6. **End with:** Section 10.3 (recommendations)

### For Your Thesis:
- **Main document is ready** - Use as Chapter 4 (Results & Analysis)
- Copy directly or adapt sections as needed
- All claims verified and backed by data
- Figures and tables included

### For Questions:
- **"How do you know?"** → Point to COMPREHENSIVE_EXPERIMENTAL_VERIFICATION.md
- **"What's new?"** → Data split discovery (+15.6%), Entanglement theory
- **"What works?"** → Position generalization (75%)
- **"What doesn't?"** → Object generalization (50%)
- **"What's next?"** → Section 10.3 recommendations

---

## ✅ Document Quality Assurance

All documents have been:
- ✅ **Verified** against experimental data (14 experiments checked)
- ✅ **Cross-referenced** for consistency
- ✅ **Balanced** with positive and negative findings
- ✅ **Updated** with latest discoveries (data split effects, geometric reconstruction insights)
- ✅ **Ready** for presentation, thesis, or publication

**Last Updated:** January 30, 2026  
**Status:** 🎉 **PRESENTATION READY**

---

## 📞 Document Maintenance

**If you need to update findings:**
1. Update main document first: `RESEARCH_FINDINGS_ACOUSTIC_CONTACT_DETECTION.md`
2. Re-verify using: `COMPLETE_VERIFICATION_CHECKLIST.md`
3. Update verification docs if needed
4. Update this index if new documents added

**If you run new experiments:**
1. Follow: `PIPELINE_GUIDE.md`
2. Add results to main document
3. Verify using checklist
4. Update this index

---

**Ready to present? Start here:** 👉 [RESEARCH_FINDINGS_ACOUSTIC_CONTACT_DETECTION.md](./RESEARCH_FINDINGS_ACOUSTIC_CONTACT_DETECTION.md)
