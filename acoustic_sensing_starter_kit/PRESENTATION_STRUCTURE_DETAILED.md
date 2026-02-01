# 🎯 Presentation Structure - Detailed Breakdown
## Acoustic Contact Detection for Robotic Surface Reconstruction
**15 Minutes | 12 Slides | Two-Goal Narrative**

---

## 📐 Overall Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PRESENTATION FLOW                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PART 1: HOOK & CONTEXT           PART 2: PROOF OF CONCEPT                  │
│  ┌──────┐ ┌──────┐ ┌──────┐      ┌──────┐ ┌──────┐ ┌──────┐                │
│  │ S1   │→│ S2   │→│ S3   │  →   │ S4   │→│ S5⭐ │→│ S6   │                │
│  │Hook  │ │Setup │ │Goals │      │Method│ │Result│ │Visual│                │
│  └──────┘ └──────┘ └──────┘      └──────┘ └──────┘ └──────┘                │
│     1min   1.5min    1min          1.5min   2min    1.5min   = 8.5 min     │
│                                                                             │
│  PART 3: GENERALIZATION           PART 4: WRAP-UP                          │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐     ┌──────┐ ┌──────┐                 │
│  │ S7   │→│ S8   │→│ S9   │→│ S10  │  →  │ S11  │→│ S12  │                 │
│  │Trans │ │Pos/Obj│ │Why  │ │Disco │     │Concl │ │ Q&A  │                 │
│  └──────┘ └──────┘ └──────┘ └──────┘     └──────┘ └──────┘                 │
│    0.5min   1.5min   1min    1min          1min    1.5min   = 6.5 min      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

# PART 1: HOOK & CONTEXT (3.5 min)

## Slide 1: Hook / Title (1 min)

### Purpose
- Capture attention immediately
- Show the end result first (surface reconstruction)
- Create curiosity about "how"

### Content Structure
| Element | Content |
|---------|---------|
| **Title** | "Can a Robot HEAR What It Touches?" |
| **Subtitle** | Acoustic-Based Contact Detection for Surface Reconstruction |
| **Visual** | Animated ground truth reconstruction |

### Visual Assets
| Asset | File | Notes |
|-------|------|-------|
| PRIMARY | `presentation_animations/ground_truth_3_shapes_blink.gif` | 3 shapes, blinking outlines |
| BACKUP | `presentation_animations/ground_truth_complete_sweep.gif` | Real data sweep animation |

### Key Topics (bullet points)
- Surface geometry detection through sound
- Binary contact classification: contact vs no-contact
- Spatial reconstruction from acoustic signals

### Narrative Arc
> "What if robots could sense geometry just by listening to touch?"

---

## Slide 2: Physical Setup (1.5 min)

### Purpose
- Ground the abstract concept in real hardware
- Show this is practical, not theoretical
- Establish credibility through real experimental setup

### Content Structure
| Element | Content |
|---------|---------|
| **Main Visual** | Photo grid: Robot + Objects + Finger + Video |
| **Data Point 1** | 10×10 raster sweep pattern |
| **Data Point 2** | 5 recordings per position |
| **Data Point 3** | 48kHz audio sampling |

### Visual Assets
| Asset | Source | Notes |
|-------|--------|-------|
| Robot photo | USER PROVIDED | Franka Panda with end effector |
| Surface objects | USER PROVIDED | Objects A, B, C, D with cutouts |
| Acoustic finger | USER PROVIDED | Microphone close-up |
| Sweep video | USER PROVIDED | Data collection in action |

### Key Topics (bullet points)
- Franka Panda robot arm
- Custom acoustic finger (microphone-based)
- Test surfaces with geometric cutouts (square, circle, triangle)
- Systematic raster sweep data collection
- Ground truth from known geometry

### Technical Details Available
| Parameter | Value | Source |
|-----------|-------|--------|
| Sweep step | 1 cm | DATA_COLLECTION_PROTOCOL.md |
| Points per line | 10 | DATA_COLLECTION_PROTOCOL.md |
| Recordings/position | 5 | DATA_COLLECTION_PROTOCOL.md |
| Dwell time | 1.15s | DATA_COLLECTION_PROTOCOL.md |
| Sample rate | 48 kHz | Feature extraction config |

---

## Slide 3: Project Goals (1 min)

### Purpose
- Set clear expectations
- Frame the two-goal narrative
- Prepare audience for what success looks like

### Content Structure
| Element | Content |
|---------|---------|
| **Goal 1 Box** | PROOF OF CONCEPT - "Is acoustic contact detection possible?" |
| **Goal 2 Box** | GENERALIZATION - "How far can we push it?" |
| **Success Metrics** | Accuracy above 50% (random chance) |

### Visual Assets
| Asset | File | Notes |
|-------|------|-------|
| Optional diagram | Text-based or simple graphic | Goals 1 & 2 boxes |

### Key Topics (bullet points)
- **Goal 1:** Prove acoustic sensing can detect contact
  - Binary classification: contact vs no-contact
  - Enable surface geometry reconstruction
  - Success = accuracy > 50%
  
- **Goal 2:** Explore generalization capabilities
  - Position generalization: same object, new positions
  - Object generalization: new object entirely
  - Research direction, not success/failure

### Transition Setup
> "Let me show you how we approached this..."

---

# PART 2: PROOF OF CONCEPT (5 min)

## Slide 4: Method / Pipeline (1.5 min)

### Purpose
- Explain the technical approach briefly
- Justify feature engineering choice
- Show the ML pipeline

### Content Structure
| Element | Content |
|---------|---------|
| **Pipeline Diagram** | Audio → Features → Classifier → Prediction |
| **Feature Breakdown** | 80 dimensions (65 hand-crafted + 15 impulse) |
| **Classifier** | Random Forest (best performer) |
| **Key Insight** | Hand-crafted > Mel-spectrograms |

### Visual Assets
| Asset | File | Notes |
|-------|------|-------|
| PRIMARY | `ml_analysis_figures/figure11_feature_dimensions.png` | 80-dim breakdown |
| ALTERNATIVE | `ml_analysis_figures/figure6_experimental_setup.png` | V4/V6 workflow |

### Key Topics (bullet points)
- **Feature extraction:**
  - 13 MFCCs + deltas = 39 dimensions
  - Spectral features (centroid, bandwidth, etc.) = 11 dimensions
  - Temporal features (ZCR, RMS, etc.) = 15 dimensions
  - Impulse response features = 15 dimensions
  
- **Classifier comparison:**
  - Random Forest: 76% ✅ (best)
  - SVM: 74%
  - Neural Network: 71%
  - Mel-spectrograms: 51% ❌ (fails)
  
- **Why hand-crafted beats deep learning:**
  - Domain-relevant features
  - Limited training data
  - Acoustic-specific patterns

### Technical Details Available
| Feature Type | Count | Source |
|--------------|-------|--------|
| MFCCs | 39 | feature_extraction.py |
| Spectral | 11 | feature_extraction.py |
| Temporal | 15 | feature_extraction.py |
| Impulse | 15 | feature_extraction.py |

---

## Slide 5: PROOF OF CONCEPT RESULT ⭐ (2 min)

### Purpose
- **THE KEY SLIDE - Main achievement**
- Prove the concept works with data
- Celebrate the success

### Content Structure
| Element | Content |
|---------|---------|
| **Main Number** | 76.2% Accuracy (LARGE) |
| **Context** | Random chance = 50% |
| **Statement** | "Acoustic contact detection IS POSSIBLE" |
| **Confidence** | Well-calibrated (75.8%) |

### Visual Assets
| Asset | File | Notes |
|-------|------|-------|
| PRIMARY | `ml_analysis_figures/figure1_v4_vs_v6_main_comparison.png` | V4 bar (76%) highlighted |
| SUPPORTING | `ml_analysis_figures/figure5_key_metrics_summary.png` | Metrics table |

### Key Topics (bullet points)
- **Primary result:** 76.2% classification accuracy
- **Significance:** 26 percentage points above random chance
- **Consistency:** 71.9% - 76.2% across multiple experiments
- **Calibration:** Model confidence matches actual performance
- **Conclusion:** Proof of concept ACHIEVED ✅

### Key Numbers to Emphasize
| Metric | Value | Significance |
|--------|-------|--------------|
| Accuracy | 76.2% | Main result |
| Random chance | 50% | Baseline |
| Improvement | +26% | Above baseline |
| Confidence | 75.8% | Well calibrated |

### Narrative Arc
> "This is the core result: We proved acoustic sensing can detect contact vs no-contact with 76% accuracy. The concept WORKS."

---

## Slide 6: Visual Proof - Surface Reconstruction (1.5 min)

### Purpose
- Show the concept working visually
- Demonstrate practical application
- Reinforce proof of concept with images

### Content Structure
| Element | Content |
|---------|---------|
| **Layout** | Ground Truth (left) vs Predicted (right) |
| **Surfaces** | 3 surfaces: squares, pure_contact, pure_no_contact |
| **Accuracy** | 70.1% validation accuracy |
| **Color Code** | Green = contact, Red = no contact |

### Visual Assets
| Asset | File | Notes |
|-------|------|-------|
| PRIMARY | `pattern_a_summary/pattern_a_visual_comparison.png` | Side-by-side comparison |
| DETAILED | `pattern_a_consistent_reconstruction/VAL_WS1_*/comparison.png` | Individual surfaces |

### Key Topics (bullet points)
- **What we see:**
  - Left: actual ground truth (known geometry)
  - Right: model's reconstruction (predictions)
  - Visual similarity demonstrates concept works
  
- **Pattern A results (Position Generalization):**
  - Train on WS2+WS3, validate on WS1
  - 70.1% accuracy on unseen positions
  - Geometry is reconstructed correctly
  
- **Visual interpretation:**
  - Green = robot touched surface (contact)
  - Red = robot touched cutout (no contact)
  - Shape boundaries are detected

### Transition
> "With proof of concept established, we asked: how far can this generalize?"

---

# PART 3: GENERALIZATION RESEARCH (4 min)

## Slide 7: Transition - Generalization Question (0.5 min)

### Purpose
- Shift narrative from "achieved" to "explored"
- Frame generalization as research direction
- Set up the two generalization types

### Content Structure
| Element | Content |
|---------|---------|
| **Question** | "How far can this approach generalize?" |
| **Type 1** | Position Generalization |
| **Type 2** | Object Generalization |

### Visual Assets
| Asset | File | Notes |
|-------|------|-------|
| Simple diagram | Text/diagram | Two arrows: position vs object |

### Key Topics (bullet points)
- **Research questions:**
  - Can model work on new positions (same object)?
  - Can model work on entirely new objects?
  
- **Experimental design:**
  - Position: Train WS2+3, test WS1
  - Object: Train A+B+C, test D (holdout)

---

## Slide 8: Generalization Results (1.5 min)

### Purpose
- Present both generalization results
- Show position works, object doesn't
- Frame as research insights, not failures

### Content Structure
| Element | Content |
|---------|---------|
| **Position Result** | 70% - Promising ✅ |
| **Object Result** | 50% - Random chance 🔬 |
| **Visual Comparison** | Two reconstruction grids side by side |

### Visual Assets
| Asset | File | Notes |
|-------|------|-------|
| POSITION | `pattern_a_summary/pattern_a_visual_comparison.png` | 70% (works) |
| OBJECT | `pattern_b_summary/pattern_b_visual_comparison.png` | 50% (fails) |
| BAR CHART | `pattern_a_summary/pattern_a_test_vs_validation_summary.png` | Accuracy comparison |
| BAR CHART | `pattern_b_summary/pattern_b_test_vs_holdout_summary.png` | Accuracy comparison |

### Key Topics (bullet points)
- **Position Generalization (Pattern A):**
  - 70.1% on new positions (WS1)
  - ~85% on training positions (WS2+3)
  - 15% drop but still well above chance
  - Indicates practical scalability possible
  
- **Object Generalization (Pattern B):**
  - 50.5% on new object (WS4/Object D)
  - ~85% on training objects (A+B+C)
  - Complete failure to generalize
  - Interesting scientific finding

### Key Numbers
| Test | Train Acc | Generalization Acc | Drop |
|------|-----------|-------------------|------|
| Position | ~85% | 70% | -15% |
| Object | ~85% | 50% | -35% |

---

## Slide 9: Why Object Generalization Fails (1 min)

### Purpose
- Explain the physics behind the failure
- Turn "failure" into "scientific insight"
- Show understanding of the problem

### Content Structure
| Element | Content |
|---------|---------|
| **Concept** | Signal Entanglement |
| **Equation** | Signal = Contact ⊗ Object Properties |
| **Explanation** | Model learns object-specific patterns |

### Visual Assets
| Asset | File | Notes |
|-------|------|-------|
| PRIMARY | `ml_analysis_figures/figure7_entanglement_concept.png` | Entanglement diagram |

### Key Topics (bullet points)
- **The Entanglement Problem:**
  - Acoustic signal contains BOTH contact state AND object identity
  - Cannot separate: Signal = Contact ⊗ Object
  - Model learned "Object A touching" not "touching in general"
  
- **Physics explanation:**
  - Each object has unique resonance frequencies
  - Material, size, geometry affect acoustic response
  - Eigenfrequencies depend on object properties
  
- **Scientific value:**
  - Not a failure, but a discovery
  - Understanding guides future solutions
  - Documented in PHYSICS_FIRST_PRINCIPLES_INTERPRETATION.md

---

## Slide 10: Key Discovery - Surface Type Effect (1 min)

### Purpose
- Highlight the +15.6% improvement finding
- Show actionable insight
- Demonstrate research contribution

### Content Structure
| Element | Content |
|---------|---------|
| **Finding** | Cutout surfaces: +15.6% accuracy |
| **Reason** | Geometric complexity creates distinctive signals |
| **Implication** | Surface design matters for this approach |

### Visual Assets
| Asset | File | Notes |
|-------|------|-------|
| PRIMARY | `ml_analysis_figures/figure10_surface_type_effect.png` | Cutout vs pure comparison |

### Key Topics (bullet points)
- **Discovery:**
  - Cutout surfaces (with shapes): 76% accuracy
  - Pure surfaces (solid): 60% accuracy
  - Difference: +15.6%
  
- **Why this matters:**
  - Complex geometry = more distinctive acoustic signals
  - Practical implication for system design
  - Guides future hardware/surface design
  
- **Actionable insight:**
  - Design surfaces with geometric features
  - Use multiple contact types per surface
  - Acoustic diversity improves classification

---

# PART 4: WRAP-UP (2.5 min)

## Slide 11: Conclusions (1 min)

### Purpose
- Summarize both goals and outcomes
- Clear takeaway messages
- Set up future directions briefly

### Content Structure
| Element | Content |
|---------|---------|
| **Goal 1 Status** | ✅ ACHIEVED - Proof of Concept |
| **Goal 2 Status** | 🔬 EXPLORED - Research Direction |
| **Key Numbers** | 76%, 70%, 50% |
| **Future Work** | Multi-modal, object-specific models |

### Visual Assets
| Asset | File | Notes |
|-------|------|-------|
| PRIMARY | `ml_analysis_figures/figure12_complete_summary.png` | All findings summary |

### Key Topics (bullet points)
- **Goal 1: Proof of Concept** ✅
  - 76% accuracy proves concept works
  - Surface reconstruction demonstrated
  - Real-time capable (<1ms inference)
  
- **Goal 2: Generalization Research** 🔬
  - Position: 70% → promising, worth pursuing
  - Object: 50% → challenge identified, future work
  - Entanglement problem understood
  
- **Contributions:**
  - Proved acoustic-tactile sensing works
  - Identified generalization boundaries
  - Discovered surface type effect (+15.6%)
  - Physics-based understanding of limitations

### Future Directions (brief)
| Direction | Approach |
|-----------|----------|
| Position scaling | More training positions |
| Object generalization | Multi-modal sensing (vision + acoustic) |
| Object-specific | Per-object calibration models |

---

## Slide 12: Q&A / Thank You (1.5 min)

### Purpose
- Close professionally
- Invite questions
- Provide contact/resources

### Content Structure
| Element | Content |
|---------|---------|
| **Title** | "Thank You" or "Questions?" |
| **Summary Numbers** | 76% | 70% | 50% |
| **Contact** | Email / GitHub |

### Visual Assets
| Asset | File | Notes |
|-------|------|-------|
| Optional | Any memorable figure from presentation | e.g., ground_truth_3_shapes_blink.gif |

### Anticipated Questions & Answers
| Question | Short Answer | Detailed Source |
|----------|--------------|-----------------|
| Why not deep learning? | Mel-spectrograms: 51%, hand-crafted: 76% | Section 2.2 main doc |
| Is 76% good enough? | Above chance, proof of concept | SCIENTIFIC_VERIFICATION.md |
| How to improve object gen? | Multi-modal sensing, more training | Section 10.3 main doc |
| Real-time capable? | Yes, <1ms inference | Section 10.3 main doc |
| Dataset size? | ~15,000 samples | Section 2.1 main doc |

---

# 📦 Asset Checklist

## Ready ✅
| Asset | Location | Status |
|-------|----------|--------|
| Hook animation | `presentation_animations/ground_truth_3_shapes_blink.gif` | ✅ |
| 12 ML figures | `ml_analysis_figures/` | ✅ |
| Pattern A summary | `pattern_a_summary/` | ✅ |
| Pattern B summary | `pattern_b_summary/` | ✅ |

## User Provides 📷
| Asset | Type | Use |
|-------|------|-----|
| Robot setup photo | Photo | Slide 2 |
| Surface objects photo | Photo | Slide 2 |
| Acoustic finger photo | Photo | Slide 2 |
| Data collection video | Video | Slide 2 |

---

# 📊 Figure Usage Map

| Figure | Slide | Purpose |
|--------|-------|---------|
| `ground_truth_3_shapes_blink.gif` | 1 | Hook animation |
| User photos/video | 2 | Physical setup |
| `figure11_feature_dimensions.png` | 4 | Feature breakdown |
| `figure1_v4_vs_v6_main_comparison.png` | 5 | **Main result** |
| `pattern_a_visual_comparison.png` | 6 | Surface reconstruction |
| `pattern_a_visual_comparison.png` | 8 | Position gen (70%) |
| `pattern_b_visual_comparison.png` | 8 | Object gen (50%) |
| `figure7_entanglement_concept.png` | 9 | Why object fails |
| `figure10_surface_type_effect.png` | 10 | Key discovery |
| `figure12_complete_summary.png` | 11 | Conclusions |

---

# 🔢 Key Numbers Reference

| Number | Meaning | Slide |
|--------|---------|-------|
| **76.2%** | Proof of concept accuracy | 5 |
| **50%** | Random chance baseline | 5 |
| **70.1%** | Position generalization | 6, 8 |
| **50.5%** | Object generalization | 8 |
| **+15.6%** | Cutout surface improvement | 10 |
| **80** | Feature dimensions | 4 |
| **<1ms** | Inference time | 11 |

---

# 🎯 Narrative Checkpoints

| After Slide | Audience Should Understand |
|-------------|---------------------------|
| 3 | Two goals: proof of concept + generalization |
| 5 | **Core message: IT WORKS (76%)** |
| 6 | Visual proof of surface reconstruction |
| 8 | Position works (70%), object doesn't (50%) |
| 9 | Why object fails (entanglement) |
| 11 | What was achieved, what's next |

---

# 📚 Backup Materials

## If More Detail Needed
| Topic | Document | Section |
|-------|----------|---------|
| Full results | RESEARCH_FINDINGS_ACOUSTIC_CONTACT_DETECTION.md | All |
| Physics explanation | PHYSICS_FIRST_PRINCIPLES_INTERPRETATION.md | All |
| Statistical validation | SCIENTIFIC_VERIFICATION_AND_DISCUSSION.md | All |
| Data collection | DATA_COLLECTION_PROTOCOL.md | All |
| Reconstruction pipeline | SURFACE_RECONSTRUCTION_PIPELINE.md | All |

## Extra Figures (if Q&A needs them)
| Figure | Use For |
|--------|---------|
| `figure2_all_classifiers_comparison.png` | All classifiers fail on V6 |
| `figure3_generalization_gap.png` | Gap visualization |
| `figure4_sample_distribution.png` | Dataset splits |
| `figure8_confidence_calibration.png` | Confidence analysis |
| `figure9_per_class_performance.png` | Per-class breakdown |
