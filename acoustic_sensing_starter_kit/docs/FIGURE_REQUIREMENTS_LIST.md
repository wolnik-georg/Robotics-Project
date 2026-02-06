# Figure Requirements List for Final IEEE Report
**Project:** Acoustic-Based Contact Detection for Robotic Manipulation  
**Author:** Georg Wolnik  
**Date:** February 5, 2026  
**Purpose:** Strategic figure selection BEFORE adding to final_report.tex

---

## Overview

This document lists **all figures needed** for the IEEE conference paper, mapping each to:
- **Section placement**
- **Scientific claim/result it supports**
- **Why it's critical**
- **Source file from existing 73+ visualizations**

**Target:** 5-7 figures for 6-8 page IEEE conference paper (standard practice)

---

## Required Figures (Priority Order)

### FIGURE 1: Main Result - V4 vs V6 Comparison ⭐ CRITICAL
**Section:** IV-A (Proof of Concept) or spanning IV-A through IV-C  
**Source File:** `ml_analysis_figures/figure1_v4_vs_v6_main_comparison.png`  

**What It Shows:**
- Side-by-side bar chart comparing V4 (position generalization) vs V6 (object generalization)
- Three bars per experiment: Training (100%), Test (99.9%), Validation (75% vs 50%)
- Visual badges: "SUCCESS ✅" for V4, "FAILURE ❌" for V6
- Statistical annotations: Z=16.28, p<0.001 for V4

**Claims It Supports:**
1. RQ1: Proof of concept works (76.2% > 50% random)
2. RQ2: Position generalization SUCCESS (75%)
3. RQ3: Object generalization FAILURE (50%)
4. Abstract claim: "75% position, 50% object"
5. Fundamental asymmetry between two generalization types

**Why It's Critical:**
- **Single most important figure** - shows all three research questions
- Immediately communicates success vs failure dichotomy
- Provides visual evidence for main contribution
- Readers can grasp entire paper from this one figure

**Caption:**
"Generalization performance comparison. **V4 (Position Generalization)**: Training on WS2+3, validating on WS1 with same objects A,B,C achieves 75.1% accuracy (SUCCESS, Z=16.28, p<0.001). **V6 (Object Generalization)**: Training on WS1+2+3 with objects A,B,C, validating on WS4 with novel object D achieves 50.5% accuracy (FAILURE, random chance). Both experiments achieve near-perfect in-distribution performance (99.9% test), but only position generalization succeeds."

---

### FIGURE 2: Experimental Setup and Workflow
**Section:** III-D (Evaluation Strategy)  
**Source File:** `ml_analysis_figures/figure6_experimental_setup.png`  

**What It Shows:**
- Two workflow diagrams side-by-side
- Left: V4 experimental design (WS2+3 → WS1, same objects)
- Right: V6 experimental design (WS1+2+3 → WS4, new object)
- Visual distinction between position-only vs position+object changes
- Icons showing objects A,B,C vs object D

**Claims It Supports:**
1. Clear experimental design methodology
2. Section III-D: "V4 changes only position... V6 changes both object and position"
3. Justification for two-experiment approach
4. Visual explanation of generalization test design

**Why It's Critical:**
- Readers need to understand experimental design to evaluate results
- Shows systematic approach (not just random experiments)
- Explains why V4 and V6 test different hypotheses
- Critical for methods section completeness

**Caption:**
"Experimental evaluation strategy. **V4 (Position Generalization)**: Tests position-invariance by training on Workspaces 2+3 and validating on Workspace 1, using the same three objects (A,B,C) across all positions. **V6 (Object Generalization)**: Tests object-agnostic detection by training on objects A,B,C across Workspaces 1+2+3, then validating on novel object D in Workspace 4. V4 isolates position changes; V6 compounds position and object changes."

---

### FIGURE 3: Surface Reconstruction - Visual Proof of Concept ⭐ CRITICAL
**Section:** IV-A (Proof of Concept)  
**Source File:** `pattern_a_summary/pattern_a_visual_comparison.png`  

**What It Shows:**
- Side-by-side surface maps: Ground Truth vs Predicted
- Color-coded contact patterns (red=contact, blue=no-contact)
- Spatial 2D visualization showing geometric reconstruction
- Accuracy metrics overlaid or in caption

**Claims It Supports:**
1. RQ1: "geometric reconstruction capability"
2. Abstract: "spatial surface mapping capabilities"
3. Section IV-A: "mapping predictions onto 2D spatial coordinates"
4. Visual proof that 76.2% accuracy translates to recognizable shapes
5. Demonstrates "moving beyond binary detection to spatial understanding"

**Why It's Critical:**
- **Core innovation claim**: first acoustic-based GEOMETRIC reconstruction
- Visual evidence is more compelling than numbers alone
- Shows practical utility: operator can see contact patterns
- Distinguishes this work from simple binary contact detection
- Directly addresses title promise: "Geometric Reconstruction"

**Caption:**
"Acoustic-based geometric reconstruction (Experiment V4, Workspace 1 validation). **Left**: Ground truth contact patterns for objects A (cutout), B (empty), and C (full contact). **Right**: Model predictions from acoustic features alone, achieving 75.1% accuracy. Color indicates contact state (red) vs no-contact (blue). This demonstrates that acoustic sensing enables spatial surface mapping, not just binary contact detection—the first such demonstration for rigid manipulators."

---

### FIGURE 4: Feature Engineering Architecture
**Section:** III-B (Feature Engineering)  
**Source File:** `ml_analysis_figures/figure11_feature_dimensions.png`  

**What It Shows:**
- Visual breakdown of 80-dimensional feature vector
- Four categories with dimension counts:
  - Spectral features (11 dims)
  - MFCCs + derivatives (39 dims)
  - Temporal features (15 dims)
  - Impulse response (15 dims)
- Possibly pie chart or horizontal bar chart showing proportions

**Claims It Supports:**
1. Section III-B: "80-dimensional hand-crafted feature vector"
2. Justification for feature categories chosen
3. Abstract: "hand-crafted acoustic features (MFCCs, spectral, temporal, and impulse response features)"
4. Comparison: 80 dims vs 10,240 mel-spectrogram dims
5. Supports "enables feature importance analysis and real-time processing"

**Why It's Critical:**
- Methods section needs visual support
- Shows engineered approach (not black-box deep learning)
- Justifies design choices for feature extraction
- Helps readers understand WHAT acoustic features mean
- Supports claim of <1ms inference time (small feature space)

**Caption:**
"Hand-crafted acoustic feature architecture. We extract an 80-dimensional feature vector from each 50ms acoustic clip, comprising: 11 spectral features (centroid, rolloff, bandwidth, flatness, contrast), 39 MFCCs with first and second derivatives, 15 temporal features (zero-crossing rate, RMS energy, statistical moments), and 15 impulse response characteristics. This compact representation achieves 75% validation accuracy compared to 51% for 10,240-dimensional mel-spectrograms, enabling real-time processing (<1ms inference)."

---

### FIGURE 5: Surface Geometry Effect (+15.6% Discovery) ⭐ NEW DISCOVERY
**Section:** IV-D (Surface Geometry Effects on Generalization)  
**Source File:** `ml_analysis_figures/figure10_surface_type_effect.png`  

**What It Shows:**
- Bar chart or split comparison showing:
  - Position generalization: WITH cutout surfaces (76.2%) vs WITHOUT (60.6%) → +15.6%
  - Object generalization: WITH cutout surfaces (~50%) vs WITHOUT (~50%) → 0% effect
- Statistical significance annotations (p<0.001 for position, p>0.5 for object)
- Visual asymmetry highlighting different mechanisms

**Claims It Supports:**
1. Abstract: "surface geometric complexity improves position generalization by +15.6% (p<0.001) but has no effect on object generalization"
2. Section IV-D: "+15.6 percentage points (60.6% to 76.2%, p<0.001)"
3. Section IV-D: "zero effect on object generalization (all variants achieve ~50%, p>0.5)"
4. Contribution #3: "Discovery of surface geometry effects on learning"
5. Design principle: "geometric complexity aids position generalization"

**Why It's Critical:**
- **Original research contribution** - not found in prior work
- Demonstrates systematic experimental variation
- Provides actionable design guidelines for future experiments
- Shows asymmetric mechanisms (position vs object learning)
- Statistical rigor (p-values prove it's not random)

**Caption:**
"Asymmetric effect of surface geometric complexity on generalization. Including cutout surfaces (Object A) in training improves **position generalization** by +15.6 percentage points (60.6%→76.2%, p<0.001, blue bars) but has **zero effect on object generalization** (50.5%→50.3%, p>0.5, red bars). This asymmetry reveals different learning mechanisms: geometric complexity forces position-invariant feature learning (beneficial for V4), but cannot overcome instance-level object memorization when training on only 2-3 objects (ineffective for V6)."

---

### FIGURE 6: Confidence Calibration Analysis ⭐ SAFETY CRITICAL
**Section:** IV-B (Position Generalization) and IV-C (Object Generalization)  
**Source File:** `ml_analysis_figures/figure8_confidence_calibration.png`  

**What It Shows:**
- Two calibration curves or confidence histograms:
  - V4: Mean confidence 75.8% → accuracy 75.1% (well-calibrated)
  - V6: Mean confidence 92.2% → accuracy 50.5% (OVERCONFIDENT)
- Possibly reliability diagram showing predicted vs actual accuracy
- Visual highlighting of dangerous overconfidence in V6

**Claims It Supports:**
1. Section IV-B: "mean confidence of 75.8% closely matches the 75.1% accuracy"
2. Section IV-B: "well-calibrated predictions"
3. Section IV-C: "model exhibits severe overconfidence: 57.2% of predictions exceed 95% confidence despite only 50.5% accuracy"
4. Section IV-C: "Mean confidence reaches 92.2%... while actual performance is far worse"
5. Practical implications: "significant safety concerns for real-world deployment"
6. Section V-B: "dangerous overconfidence (92% confidence at 50% accuracy)"

**Why It's Critical:**
- **Deployment safety** - shows V6 is not just wrong, but confidently wrong
- Explains why confidence filtering cannot fix object generalization
- Demonstrates model cannot recognize out-of-distribution data
- Supports discussion of practical implications
- Critical for arguing against deploying V6 approach

**Caption:**
"Confidence calibration analysis reveals safety-critical overconfidence. **V4 (Position Generalization)**: Mean confidence 75.8% matches accuracy 75.1%, indicating well-calibrated uncertainty (left). The model appropriately recognizes when predictions are uncertain. **V6 (Object Generalization)**: Mean confidence 92.2% dramatically exceeds 50.5% accuracy, with 57.2% of predictions exceeding 95% confidence despite random-chance performance (right). This inverse relationship (higher confidence, lower accuracy) indicates the model cannot recognize novel objects as out-of-distribution, presenting deployment safety risks."

---

### OPTIONAL FIGURE 7: All Classifiers Fail on V6 (Supports Generality)
**Section:** IV-C (Object Generalization)  
**Source File:** `ml_analysis_figures/figure2_all_classifiers_comparison.png`  

**What It Shows:**
- Bar chart showing 5 classifiers on V6:
  - Random Forest: 50.5%
  - k-NN: 49.8%
  - MLP: 50.1%
  - GPU-MLP: 50.3%
  - Ensemble: 50.2%
- All clustered around 50% random baseline
- Visual emphasis on classifier-agnostic failure

**Claims It Supports:**
1. Section IV-C: "We tested five different classifier families... observed identical failure"
2. Section IV-C: "all achieve 49.8%--50.5% accuracy on object D with less than 1% variance"
3. Section IV-C: "This classifier-agnostic failure confirms the problem lies not in the learning algorithm but in the feature representation itself"
4. Strengthens argument that failure is fundamental, not algorithmic

**Why It's Potentially Critical:**
- Proves failure is not due to poor classifier choice
- Strengthens scientific rigor (tested multiple approaches)
- Deflects potential criticism: "Did you try other classifiers?"
- Shows systematic exploration

**Why It Might Be Skipped:**
- Page limits (already have 6 strong figures)
- Main point already made by Figure 1
- Can be described in text without visual

**Caption (if included):**
"Classifier-agnostic failure on object generalization. Testing five different classifier families on Experiment V6 yields identical performance: all achieve 49.8-50.5% accuracy (random chance) with <1% variance. This consistency across Random Forest, k-NN, MLP, GPU-accelerated MLP, and ensemble methods confirms the failure stems from feature representation (instance-specific signatures) rather than learning algorithm choice."

---

## Figure Placement Strategy

### IEEE Conference Format Constraints:
- **Target:** 5-7 figures for 6-8 page paper
- **Column width:** Figures can be single-column (3.5") or double-column (7.16")
- **Placement:** Float to top/bottom of pages near first reference

### Recommended Inclusion:

**MUST INCLUDE (6 figures):**
1. ✅ Figure 1: V4 vs V6 Comparison (main result)
2. ✅ Figure 2: Experimental Setup (methods clarity)
3. ✅ Figure 3: Surface Reconstruction (proof of concept)
4. ✅ Figure 4: Feature Engineering (methods support)
5. ✅ Figure 5: Surface Geometry Effect (+15.6% discovery)
6. ✅ Figure 6: Confidence Calibration (safety critical)

**OPTIONAL (if space allows):**
7. ⚠️ Figure 7: All Classifiers Fail (scientific rigor)

---

## Section-by-Section Figure Mapping

| Section | Figures Needed | Purpose |
|---------|---------------|---------|
| **I. Introduction** | None | Text-only, sets up RQs |
| **II. Related Work** | None | Literature review |
| **III. Method** | Fig 2 (setup), Fig 4 (features) | Explain methodology |
| **IV. Results** | Fig 1 (main), Fig 3 (reconstruction), Fig 5 (geometry), Fig 6 (confidence) | Support all claims |
| **V. Conclusion** | None | Summarizes findings |

---

## Figure Quality Checklist

Before adding each figure, verify:
- ✅ High resolution (300 DPI minimum for publication)
- ✅ Readable text at column width (test at 3.5" width)
- ✅ Color-blind friendly palette (if using color)
- ✅ Clear axis labels and legends
- ✅ Caption explains figure WITHOUT referring to main text
- ✅ First mention in text BEFORE figure placement
- ✅ Figure number referenced in text: "Fig.~\ref{fig:label}"

---

## Next Steps

1. **Get user approval** on this figure list
2. **Add \includegraphics** commands to final_report.tex
3. **Write detailed captions** with all necessary context
4. **Adjust text** to reference figures appropriately
5. **Compile and check** figure placement and readability
6. **Verify page count** stays within 6-8 pages

---

## Summary Statistics

| Metric | Value |
|--------|-------|
| **Required Figures** | 6 (core) + 1 (optional) |
| **Sections with Figures** | III (2 figs), IV (4 figs) |
| **Research Questions Supported** | RQ1 (2 figs), RQ2 (3 figs), RQ3 (3 figs) |
| **New Discoveries Visualized** | +15.6% geometry effect, overconfidence problem |
| **Safety-Critical Figures** | 1 (confidence calibration) |
| **Proof-of-Concept Figures** | 2 (reconstruction + main result) |

---

**Status:** ✅ Ready for user review  
**Action Required:** Confirm figure selection before LaTeX integration
