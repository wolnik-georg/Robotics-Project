# Experimental Validation Required

**Last Updated:** February 8, 2026 (Updated: Item #1 COMPLETE ✅)  
**Report Status:** 3-Class Framework Complete ✅  
**Figure Status:** All 3 reconstruction figures verified ✅  
**Validation Status:** Item #1 (Features vs Spectrograms) COMPLETE ✅

**Document Purpose:** This document tracks all claims, design choices, and methodological decisions in the final report that currently lack empirical validation. Each item represents an experiment needed to rigorously justify statements made in the paper.

**Priority Levels:**
- 🔴 **CRITICAL**: Core claims affecting paper validity - must complete before submission
- 🟡 **HIGH**: Significant methodological choices - strongly recommended for rigor
- 🟢 **MEDIUM**: Supporting details that enhance credibility
- ⚪ **LOW**: Nice-to-have validation for comprehensive reporting
- ✅ **COMPLETE**: Already validated or no longer applicable

---

## 📊 Current State Summary (Feb 8, 2026)

### ✅ Completed Updates:
1. **3-Class Framework:** Complete transition from binary to 3-class (contact, no-contact, edge)
2. **All Figures Updated:** Proof of concept, position generalization, object generalization reconstructions
3. **All Metrics Updated:** 77% CV, 60% avg validation, 33.3% random baseline throughout
4. **Binary Comparison Added:** Table comparing 3-class vs binary with normalized performance
5. **Edge Cases:** Now INCLUDED explicitly as third class (not excluded)
6. **✅ Item #1 COMPLETE:** Hand-crafted vs spectrograms experimental validation added to Section III.B

### 🎯 Next Steps:
- Phase 0: NO CRITICAL TEXT FIXES NEEDED ✅ (all done in 3-class update)
- Phase 1: Critical experimental validations (4-8 hours remaining - Item #2 only)
- Phase 2: High-priority enhancements (2-3 hours)

---

## 🔴 CRITICAL Experiments (Must Complete Before Submission)

### ✅ 1. Hand-Crafted Features vs. Spectrograms - COMPLETE
**Claim (Section III.B):** 
> "our compact 80-dimensional representation significantly outperforms spectrograms (75% vs. 51% validation accuracy)"

**Status:** ✅ **COMPLETE - ADDED TO REPORT (Feb 8, 2026)**

**What Was Added:**
- **Table 5:** Classifier-by-classifier comparison (Random Forest, K-NN, MLP, GPU-MLP, Ensemble)
  - Hand-crafted features win 4/5 classifiers
  - Random Forest: 80.9% (features) vs 17.8% (spectrograms) = +63.1% advantage
  - Best overall: 80.9% (hand-crafted RF) vs 66.3% (spectrogram GPU-MLP)
- **Figure 3:** Side-by-side classifier performance comparison showing CV/validation gaps
  - Hand-crafted: Small gaps (good generalization)
  - Spectrograms: Large gaps (severe overfitting)
- **Paragraph in Section III.B:** Full experimental description and interpretation

**Evidence Location:**
- Section III.B (Feature Engineering)
- Table 5: Hand-Crafted Features vs. Spectrograms Comparison
- Figure 3: Classifier performance comparison

**Experimental Results (Rotation 1: Train WS1+WS3, Validate WS2):**
| Classifier | Features (80D) | Spectrograms (10,240D) | Advantage |
|------------|----------------|------------------------|-----------|
| Random Forest | 80.9% | 17.8% | +63.1% |
| K-NN | 45.9% | 42.9% | +3.0% |
| MLP (Medium) | 43.1% | 32.2% | +10.9% |
| GPU-MLP (Medium) | 63.6% | 66.3% | -2.7% |
| Ensemble (Top3-MLP) | 43.2% | 32.6% | +10.6% |

**Key Finding:** Hand-crafted features avoid workspace-specific overfitting (19.1% gap) compared to spectrograms (82.2% gap for RF).

**Time Invested:** ~3 hours (experiments + report integration)

---

### 2. Multi-Sample Recording Necessity (Motion Artifact Elimination) ⚠️ CRITICAL
|--------------|----------------|---------|----------------------|----------------------|----------------------|---------|
| Mel-spectrogram | 10,240 | ? | ~51% expected | ? | ? | ~51% |
| Hand-crafted | 80 | 77.0% | 84.9% | 60.4% | 34.9% | 60.0% |

**Where to Add:** Section III.B (Feature Engineering) or Section IV as ablation study

**Estimated Time:** 2-4 hours (re-run pipeline with spectrogram features)

**Impact:** HIGH - Validates core feature engineering choice

---

### 2. Multi-Sample Recording Necessity (Motion Artifact Elimination)
**Claim (Section III.A):**
> "recording 5--10 acoustic samples per position with 150~ms mechanical settling time between recordings"

**Implicit Claim:** Single-sample recording fails due to robot motion contamination.

**Current Status:** ❌ Design choice explained but NOT VALIDATED.

**Why Critical:** This affects data collection protocol and validates whether settling time is actually necessary.

**Required Experiment:**
- **Baseline:** 1 sample per position, no settling time (immediate recording during/after motion)
- **Current:** 5-10 samples per position with 150ms settling time
- Compare model accuracy for both protocols on 3-class workspace rotations

**Expected Results (3-Class Framework):**
| Protocol | Samples/Position | Settling Time | CV Acc. | Val Acc. (Avg) | Interpretation |
|----------|------------------|---------------|---------|----------------|----------------|
| Baseline | 1 | 0ms | ~40-50% expected | ~33% (random) | Motion artifacts dominate |
| Current | 5-10 | 150ms | 77.0% | 60.0% | Clean acoustic signals |

**Hypothesis:** Baseline should show near-random performance (~33%) or severe overfitting, proving that motion artifact elimination is essential.

**Where to Add:** Section III.A or create new Section IV subsection "Motion Artifact Validation"

**Estimated Time:** 4-8 hours (may need early experimental data OR re-collect small test set)

**Impact:** HIGH - Validates fundamental data collection protocol

---

### ✅ 3. Edge Case Inclusion Benefit (NOW COMPLETE)
**Original Claim (Binary Version):**
> "All edge cases where the contact finger partially overlaps object boundaries are excluded to maintain clean binary labels."

**NEW STATUS:** ✅ **EXPERIMENT COMPLETE - NOW PART OF CORE 3-CLASS FRAMEWORK**

**What Changed:**
- Edge cases are now INCLUDED as explicit third class
- Binary comparison shows 3-class (1.80× over random) outperforms binary (1.15× over random) by 56%
- Table 5 in report provides empirical evidence

**Evidence in Report:**
- **Table 5:** 3-Class vs Binary Classification Comparison
  - Binary (exclude edge): 57.6% val, 1.15× over random
  - 3-Class (include edge): 60.0% val, 1.80× over random
- **Section IV.D:** Full analysis of why edge inclusion improves normalized performance

**Action:** ✅ NO FURTHER VALIDATION NEEDED - This is now a core contribution of the paper

---

## 🟡 HIGH Priority Experiments (Strongly Recommended for Rigor)

## 🟡 HIGH Priority Experiments (Strongly Recommended for Rigor)

### ✅ 4. StandardScaler vs. Alternative Normalization - SKIPPED

**Original Claim (Section III.B):**
> "We selected StandardScaler over alternatives after experimental validation showed that per-sample normalization reduced accuracy by 5.8%"

**Status:** ✅ **SKIPPED - Claim removed from report (Feb 8, 2026)**

**Reason for Skipping:** 
- StandardScaler is standard ML practice for feature normalization
- Specific "5.8%" claim was unsupported and removed from report
- Low priority compared to critical validations (Item #2)
- Current text simply states normalization method without claiming superiority

**Action Taken:**
- Removed "5.8%" claim from Section III.B
- Simplified to: "All features are normalized using StandardScaler (zero mean, unit variance) fitted exclusively on training data and applied consistently to validation sets, ensuring zero data leakage across train/validation splits."
- No experimental validation needed

**Time Saved:** 1-2 hours (redirected to higher-priority items)

---

### 5. Random Forest 100 Trees - Parameter Sweep
**Current Claim (Section III.C):**
> "We use 100 trees as a standard default configuration, as preliminary experiments showed all top-performing models achieved comparable performance (77% ± 2% cross-validation accuracy for 3-class classification)"

**Current Status:** ⚠️ Reasonable explanation, but could be strengthened.

**Why High Priority:** Quick win, demonstrates hyperparameter selection rigor.

**Possible Enhancement:**
- Show hyperparameter sweep: 10, 25, 50, 100, 200, 500 trees
- Plot accuracy vs. tree count (diminishing returns curve)

**Expected Finding:** 100 trees likely in plateau region where additional trees yield <0.5% improvement.

**Where to Add:** Supplementary material or brief mention

**Estimated Time:** 1 hour (quick parameter sweep)

**Impact:** MEDIUM - Nice-to-have validation of standard choice

---

### ✅ 6. Ground Truth Automatic Labeling Logic (PARTIALLY DOCUMENTED)
**Claim (Section III.A):**
> "Ground truth labels (contact, no-contact, or edge) are assigned automatically based on spatial position relative to object geometry"

**Current Status:** ⚠️ Claimed but logic not fully described.

**What's Documented:**
- Edge cases: "explicitly labeled when the contact finger partially overlaps object boundaries"
- Contact/no-contact: Based on spatial position

**What's Missing:**
- Exact algorithm for determining edge boundaries
- How "partial overlap" is computed (geometric calculation)
- Validation that automatic labeling is accurate

**Recommended Enhancement:**
- Manually verify subset of labels (e.g., 100 random samples)
- Compute inter-annotator agreement (automatic vs. manual)
- Expected: >95% agreement for 3-class labels

**Where to Document:** Technical documentation or supplementary methods

**Estimated Time:** 2 hours (manual verification) + documentation

**Impact:** MEDIUM - Improves reproducibility

---

## � HIGH Priority Experiments (Strongly Recommended for Rigor)

### 6. Class Balance Investigation - Data Quality Validation ⚠️ CRITICAL
**Claim (Section III.D):**
> "Dataset construction ensures balanced representation across all three classes"
> "This 33/33/33 split ensures the model cannot exploit class imbalance"

**Current Status:** ❌ **CLAIMED BUT NOT VERIFIED**

**Why Critical:** Class imbalance can invalidate your entire experimental setup and results.

**Required Investigation:**
1. **Count actual samples per class** across all datasets:
   - Rotation 1 (Train WS1+WS3): Contact, No-Contact, Edge counts
   - Rotation 2 (Train WS2+WS3): Contact, No-Contact, Edge counts
   - Rotation 3 (Train WS1+WS2): Contact, No-Contact, Edge counts
   - Validation sets (WS2, WS1, WS3): Individual class counts

2. **Verify balanced split claim:**
   - Expected: 33/33/33 split (±2% tolerance)
   - Actual: ? / ? / ? (needs investigation)
   - If imbalanced: Results may be invalid (model exploits majority class)

3. **Check workspace-specific imbalances:**
   - Does WS3 have fewer edge cases? (Could explain 34.9% failure)
   - Does WS1 have more contact samples? (Could inflate 85% accuracy)
   - Are certain objects over-represented in specific workspaces?

**Expected Findings:**
| Dataset | Contact | No-Contact | Edge | Balance Check |
|---------|---------|------------|------|---------------|
| Rotation 1 Train | ~33% | ~33% | ~33% | ✓ Balanced |
| Rotation 1 Val (WS2) | ? | ? | ? | **Verify** |
| Rotation 2 Train | ~33% | ~33% | ~33% | ✓ Balanced |
| Rotation 2 Val (WS1) | ? | ? | ? | **Verify** |
| Rotation 3 Train | ~33% | ~33% | ~33% | ✓ Balanced |
| Rotation 3 Val (WS3) | ? | ? | ? | **Verify** |

**Critical Questions:**
- Are edge cases underrepresented in WS3? (explains low 34.9% accuracy)
- Is WS1 somehow easier due to class distribution? (explains high 85% accuracy)
- Are you actually achieving 33/33/33 or is it more like 40/40/20?

**Where to Add:** 
- Section III.D (Dataset Construction) - Add actual counts
- Supplementary material - Full class distribution tables

**Estimated Time:** 30-60 minutes (run counts on existing datasets)

**Impact:** 🔴 **CRITICAL** - If imbalanced, entire experimental setup needs revision

**Priority:** 🔴 **MUST DO IMMEDIATELY** - This could invalidate your results

---

## 🟢 MEDIUM Priority Experiments (Nice to Have, Strengthen Arguments)

### 7. Confidence Threshold Selection - Full Sweep
**Claim (Section III.C):**
> "We evaluated a range of threshold values (0.60, 0.70, 0.80, 0.90, 0.95) and selected 0.80 as providing optimal balance between accuracy and coverage"

**Current Status:** ⚠️ States that sweep was performed, but NO RESULTS shown.

**Possible Enhancement:**
- Show table of coverage vs. accuracy for each threshold
- Plot Pareto frontier (coverage vs. accuracy trade-off)

**Expected Results (3-Class Framework):**
| Threshold | Coverage (Rot 1) | Accuracy (Rot 1) | Coverage (Rot 2) | Accuracy (Rot 2) | Coverage (Rot 3) | Accuracy (Rot 3) |
|-----------|------------------|------------------|------------------|------------------|------------------|------------------|
| 0.60 | ~95% | ~80% | ~95% | ~57% | ~95% | ~32% |
| 0.70 | ~85% | ~82% | ~85% | ~58% | ~85% | ~33% |
| 0.80 | ~60% | ~85% | ~60% | ~60% | ~60% | ~35% |
| 0.90 | ~20% | ~87% | ~20% | ~62% | ~20% | ~36% |
| 0.95 | ~10% | ~88% | ~10% | ~63% | ~10% | ~36% |

**Where to Add:** Supplementary material or brief figure

**Estimated Time:** 30 minutes (already have predictions, just apply different thresholds)

**Impact:** LOW-MEDIUM - Nice visualization of trade-off

---

### ✅ 8. Why Objects A/B/C Used for 3-Class Framework (NOW DOCUMENTED)
**Original Question (Binary Version):**
> "Why are objects A and C grouped as same 'contact' class?"

**NEW STATUS:** ✅ **NOW CLEARLY EXPLAINED IN 3-CLASS FRAMEWORK**

**Current Documentation:**
- Object A (cutout): Provides all 3 classes (contact, no-contact, edge)
- Object B (empty): Provides pure no-contact
- Object C (full): Provides contact and edge
- **Dataset construction:** "balanced representation across all three classes" (Section III.D)
- **33/33/33 split** ensures no class imbalance

**Evidence in Report:**
- Section III.A: Clear descriptions of each object type
- Section III.D: "contact samples come from objects A (cutout) and C (full contact), no-contact samples come from object B (empty workspace) and positions where the acoustic finger enters cutout regions without touching surfaces, and edge samples come from positions where the contact finger partially overlaps object boundaries"

**Action:** ✅ NO FURTHER VALIDATION NEEDED - Methodology is clearly explained

---

### 9. Feature Variance Analysis - "Natural Data Augmentation" Claim
**Claim (Section IV.E - Physics Interpretation):**
> "Different workspace cutout patterns create workspace-specific vibration damping and reflection patterns, especially for edge cases"

**Current Status:** ⚠️ Compelling explanation, but "natural augmentation" claim is interpretive.

**Possible Validation:**
- Compute feature diversity metrics (variance, range) for edge cases across workspaces
- Show that WS3 edge cases have different acoustic signatures than WS1/WS2
- Measure acoustic variation: Var(features | WS3 edges) vs Var(features | WS1/WS2 edges)

**Expected Finding:** WS3 should show fundamentally different edge signature distributions, explaining 34.9% validation failure.

**Scientific Value:** Would strengthen the mechanistic explanation for workspace dependence.

**Estimated Time:** 1-2 hours (feature variance analysis on existing data)

**Impact:** MEDIUM - Strengthens physics-based interpretation

---

## ⚪ LOW Priority Experiments (Optional Engineering Details)

### 11. Why Exactly 150ms Settling Time
**Claim (Section III.A):**
> "150~ms mechanical settling time between recordings"

**Current Status:** Empirically chosen, but not rigorously tested.

**Possible Validation:**
- Test 50ms, 100ms, 150ms, 200ms settling times
- Measure vibration decay with accelerometer to find optimal value

**Recommendation:** Engineering parameter, not critical to validate unless reviewers question it.

---

### 12. Why 5-10 Samples (Not Fixed Count)
**Claim (Section III.A):**
> "5--10 acoustic samples per position"

**Question:** Why variable count? Why not always 5 or always 10?

**Possible Answer:** Practical data collection flexibility, or different experiments used different counts.

**Recommendation:** Clarify in text if needed, but doesn't require validation.

---

## 🔴 ADDITIONAL CRITICAL VALIDATIONS (From Comprehensive Report Review - Feb 6, 2026)

### 13. ~~Random Forest "95.2% Average Accuracy"~~ ✅ RESOLVED
**Status:** RESOLVED - Changed to "best in-distribution performance on test data"

**Action Taken:** Removed confusing "95.2%" number and replaced with general statement about in-distribution performance.

---

### 21. Sample Count Discrepancy - "15,749 Labeled Samples"
**Claim (Abstract):** 
> "15,749 labeled samples"

**Issue:** This is V4 total only (10,639 + 2,660 + 2,450 = 15,749). V6 has different total (14,819).

**Severity:** 🟢 MEDIUM - Minor accuracy issue

**Recommendation:** Change abstract to "approximately 15,000 samples" OR clarify "up to 15,749 samples across experiments"

**Where to Fix:** Abstract

**Estimated Time:** 5 minutes

---

### 22. ❌ "Non-Contact Regime" - FACTUALLY INCORRECT
**Claim (Section I):** 
> "First, it operates in a *non-contact* regime, detecting impending contact before force is applied"

**CRITICAL ISSUE:** Your system uses a **contact microphone** that REQUIRES physical contact. It does NOT detect "impending contact."

**Severity:** 🔴 **CRITICAL FACTUAL ERROR**

**Required Action:** **DELETE THIS CLAIM ENTIRELY** or rephrase completely

**Suggested Fix:** 
- DELETE the claim, OR
- Replace with: "First, it can detect contact through vibration sensing, requiring minimal applied force compared to traditional force sensors"

**Where to Fix:** Section I (Introduction), paragraph 3

**Estimated Time:** 10 minutes

**PRIORITY:** MUST FIX IMMEDIATELY

---

### 23. "Material Properties" - Overclaiming System Capabilities
**Claim (Section I):**
> "acoustic signals encode rich temporal and spectral information about contact events, **material properties**, and surface geometry"

**Issue:** You never classify materials or extract material properties. System only does binary contact detection.

**Severity:** 🟡 HIGH - Overclaiming capabilities

**Recommendation:** Remove "material properties" and replace with "object-specific acoustic signatures"

**Suggested Fix:** "encode rich temporal and spectral information about contact events and surface geometry through object-specific vibration patterns"

**Where to Fix:** Section I (Introduction), paragraph 3

**Estimated Time:** 5 minutes

---

### 24. "Comparable or Superior Information Density" - Unsupported
**Claim (Section I):**
> "providing **comparable or superior** information density"

**Issue:** No quantitative comparison of information density provided in paper.

**Severity:** 🟡 HIGH - Strong claim without evidence

**Recommendation:** Remove "or superior" → "providing information density suitable for contact detection tasks"

**Where to Fix:** Section I (Introduction), paragraph 3

**Estimated Time:** 5 minutes

---

### 25. "Approximately 500 Positions Per Workspace" - MATH ERROR
**Claim (Section III.A):**
> "Each workspace yields approximately 500 positions, producing ~2,500 samples per workspace"

**CRITICAL ISSUE:** 10cm × 10cm surface at 1cm resolution = **100 positions maximum**, NOT 500!

**Possible Explanation:** Multiple objects (A, B, C) scanned per workspace → 3 × ~170 positions each?

**Severity:** 🔴 **CRITICAL - Math doesn't add up**

**Required Action:** **CLARIFY IMMEDIATELY** - Investigate actual dataset and explain methodology

**Questions to Answer:**
- Are all 3 objects (A, B, C) scanned at different positions?
- Is it actually ~500 total positions across all objects in a workspace?
- Or is it ~100-170 positions per object?

**Where to Fix:** Section III.A (Experimental Setup)

**Estimated Time:** 1-2 hours (requires dataset investigation)

**PRIORITY:** MUST CLARIFY BEFORE SUBMISSION

---

### 26. Five Classifier Comparison Table Missing
**Claim (Section IV.C):**
> "We tested five different classifier families (Random Forest, k-NN, MLP, GPU-MLP, ensemble methods) and observed identical failure: all achieve 49.8\%--50.5\% accuracy on object D"

**Issue:** This is a CRITICAL claim for explaining object generalization failure, but NO table/figure shows this data.

**Severity:** 🔴 **CRITICAL - Must show evidence**

**Required Action:** Add table showing all 5 classifier results on V6 experiment

**Expected Table:**
| Classifier | Train Acc. | Test Acc. (WS1+2+3) | Val Acc. (WS4/Obj D) |
|------------|------------|---------------------|----------------------|
| Random Forest | 100% | 99.9% | 50.5% |
| k-NN | ? | ? | ~50% |
| MLP | ? | ? | ~50% |
| GPU-MLP | ? | ? | ~50% |
| Ensemble | ? | ? | ~50% |

**Where to Add:** Section IV.C (Object Generalization)

**Estimated Time:** 30 minutes (if data exists) OR 1-2 hours (if need to re-run)

**PRIORITY:** MUST ADD BEFORE SUBMISSION

---

### 27. "75.1% Accuracy" Should Be "76.2%" - TYPO
**Claim (Section V.A - Conclusion):**
> "Models trained at specific robot configurations successfully generalize to new positions with **75.1% accuracy**"

**Issue:** Actual validation accuracy is **76.2%** (from Section IV.A, Table 2)

**Severity:** 🔴 **CRITICAL ERROR - Inconsistent numbers**

**Required Action:** Change 75.1% → 76.2% throughout conclusion

**Where to Fix:** Section V.A (Summary of Findings)

**Estimated Time:** 2 minutes (find & replace)

**PRIORITY:** MUST FIX IMMEDIATELY

---

### 28. "10+ Diverse Objects" - Number Without Justification
**Claim (Section IV.D):**
> "object generalization requires training on 10+ diverse objects to force abstraction beyond instance-specific patterns"

**Issue:** Where does "10+" come from? No citation or derivation provided.

**Severity:** 🟡 HIGH - Specific number without justification

**Recommendation:** Add hedge: "likely requires training on substantially more diverse objects (estimated 10+ based on machine learning best practices for category-level learning)" OR cite comparable work

**Where to Fix:** Section IV.D (Surface Geometry Effects)

**Estimated Time:** 10 minutes

---

### 29. "Impossible" Too Strong for Object-Agnostic Detection
**Claim (Section IV.E):**
> "making object-agnostic contact detection impossible without sufficient object diversity"

**Issue:** You showed it fails with 3 training objects, but "impossible" is too absolute.

**Severity:** 🟡 HIGH - Overclaiming limitation

**Recommendation:** Change to: "making object-agnostic contact detection **extremely challenging** without sufficient object diversity"

**Where to Fix:** Section IV.E (Physics-Based Interpretation)

**Estimated Time:** 5 minutes

---

### 30. "p>0.5" Without Statistical Test Description
**Claim (Section IV.D):**
> "zero effect on object generalization (all variants achieve $\sim$50\%, p$>$0.5)"

**Issue:** p-value cited but no statistical test described.

**Severity:** 🟡 HIGH - Incomplete statistical reporting

**Recommendation:** Add test description: "p>0.5, two-sample t-test" OR remove p-value if no formal test was conducted

**Where to Fix:** Section IV.D (Surface Geometry Effects)

**Estimated Time:** 5 minutes (if test exists) OR 30 minutes (to run test)

---

### 14. ~~"Inference Time <1ms"~~ ✅ REMOVED
**Status:** REMOVED per user request

**Action Taken:** All mentions of inference time removed from report (Section IV.A and Conclusion).

---

### 15. ~~"High-Frequency Contact Transients up to 20kHz"~~ ✅ REMOVED
**Status:** REMOVED per user request

**Action Taken:** Removed claim about "reliably capture high-frequency contact transients up to 20kHz" from Section III.A. Now simply states "48kHz sampling rate in mono (16-bit PCM)" without technical justification.

---

### 16. ~~"Well-Calibrated Confidence Estimates"~~ ✅ SIMPLIFIED
**Status:** SIMPLIFIED - Removed calibration claims

**Action Taken:** 
- Removed "well-calibrated confidence estimates" from Section III.C
- Removed detailed calibration analysis from Section IV.B
- Simplified confidence filtering description to just state tested threshold values (0.60-0.95) and selected 0.90
- Updated Figure 8 caption to remove "well-calibrated" and "appropriately recognizes" language
- Kept factual confidence statistics (mean confidence 75.8% vs 76.2% accuracy) without interpretation

---

### 17. "Approximately 500 Positions Per Workspace" - Dataset Size Validation
**Claim (Section III.A):**
> "Each workspace yields approximately 500 positions, producing ~2,500 samples per workspace"

**Implicit Math:** 500 positions × 5 samples/position = 2,500 samples ✓

**Questions:**
- Is it actually ~500 positions, or is this rounded? (476? 523?)
- Why does this produce 2,500 samples if 5-10 samples per position? (Should be 2,500-5,000)
- Are all 4 objects scanned at all positions, or different positions per object?

**Required Clarification:**
- State actual position counts (or confirm 500 is accurate average)
- Clarify sampling protocol: "500 positions × 5 samples/position average across contact/no-contact regions"

**Recommendation:** Minor clarification in text, not a critical validation experiment.

---

### 18. "Automatic Ground Truth Labeling" - Logic Validation
**Claim (Section III.A):**
> "Ground truth labels (contact vs. no-contact) are assigned automatically based on spatial position relative to object geometry"

**Current Status:** Claimed but logic not described or validated.

**Required Documentation:**
- How is "spatial position relative to object geometry" computed?
- What is the ground truth labeling algorithm?
- How accurate is automatic labeling vs. manual verification?

**Validation Needed:**
- Manually verify subset of labels (e.g., 100 random samples)
- Compute inter-annotator agreement (automatic vs. manual)
- Expected: >95% agreement for clean binary labels

**Critical for Reproducibility:** Other researchers cannot replicate without knowing the labeling logic.

**Where to Document:** Should be in technical documentation or methods section.

**Estimated Time:** 2 hours (manual verification) + documentation

---

### 19. "Avoid Overfitting to Training-Specific Acoustic Patterns" - Spectrograms
**Claim (Section III.B):**
> "our compact 80-dimensional representation significantly outperforms spectrograms (75% vs. 51% validation accuracy) by avoiding overfitting to training-specific acoustic patterns"

**Current Status:** Explanation given, but overfitting claim not proven.

**To Rigorously Prove "Overfitting" Claim:**
Need to show spectrograms achieve:
- High train accuracy (~90%+)
- High test accuracy on WS2+3 (~90%+)  
- **Low validation accuracy on WS1 (~51%)**

This pattern = overfitting (memorizes training workspace acoustics).

**Contrast with hand-crafted features:**
- High train accuracy (100%)
- High test accuracy (99.9%)
- **Decent validation accuracy (76.2%)**

This pattern = better generalization.

**Required Data (part of Experiment #1):**
Show train/test/val for both feature types to prove overfitting claim.

---

### 20. "Surface Geometry Complexity as Natural Data Augmentation"
**Claim (Section IV.D):**
> "The geometric complexity acts as natural data augmentation, improving robustness to robot configuration changes."

**Current Status:** Compelling explanation, but "data augmentation" claim is interpretive.

**To Validate:**
- Compute feature diversity metrics (variance, range) for Object A vs. Objects B+C
- Show that Object A creates more diverse acoustic patterns across positions
- Measure acoustic variation: Var(features | Object A) > Var(features | Objects B+C)

**Expected Finding:** Cutout surfaces should show higher intra-object feature variance, supporting "natural augmentation" interpretation.

**Scientific Value:** Would strengthen the mechanistic explanation for why geometric complexity helps.

**Estimated Time:** 1-2 hours (feature variance analysis on existing data)

---

## 📊 Summary Statistics (Updated Feb 8, 2026 - Item #1 Complete)

### Total Validation Items: 11 Active (down from 30 original, 2 newly completed/skipped, 1 newly added)

**Status Breakdown:**
- ✅ **COMPLETE/SKIPPED:** 11 items (9 from 3-class transition + Item #1 features vs spectrograms + Item #4 normalization skipped)
- 🔴 **CRITICAL:** 1 item (Item 2: Multi-sample necessity)
- 🟡 **HIGH:** 2 items (Item 5: RF trees sweep, Item 6: Class balance investigation)
- 🟢 **MEDIUM:** 2 items (Items 7, 9: Supporting analyses)
- ⚪ **LOW:** 6 items (Engineering details - skipped or optional)

---

### ✅ COMPLETED Items (11 Total)

**Phase 0 Text Fixes - ALL COMPLETE (9 items):**
1. ✅ Item 3: Edge case exclusion → NOW CORE 3-CLASS CONTRIBUTION
2. ✅ Item 8: Object grouping → NOW CLEARLY EXPLAINED in 3-class framework
3. ✅ Item 13: "95.2%" claim → Removed
4. ✅ Item 14: Inference time → Removed
5. ✅ Item 15: High-frequency transients → Removed
6. ✅ Item 16: Calibration claims → Simplified
7. ✅ Item 22-30: All overclaiming/inconsistencies → Fixed in 3-class version

**Experimental Validations - COMPLETE (1 item):**
8. ✅ **Item 1: Hand-Crafted vs Spectrograms (Feb 8, 2026)**
   - Added Table 5: Classifier comparison (5 classifiers)
   - Added Figure 3: Side-by-side performance plots
   - Added paragraph in Section III.B with full experimental description
   - Result: Hand-crafted wins 4/5 classifiers (+63.1% for Random Forest)
   - Time: ~3 hours

**Methodological Decisions - SKIPPED (7 items):**
9. ✅ Item 4: Normalization comparison → Claim removed (Feb 8), standard practice
10. ✅ Item 5: 1cm spatial resolution → Practical choice, well-justified
11. ✅ Item 6: 48kHz sampling rate → Standard audio engineering practice
12. ✅ Item 10: 80/20 vs CV → Using 5-fold CV in 3-class framework
13. ✅ Item 11: 150ms settling time → Engineering parameter
14. ✅ Item 12: 5-10 samples count → Practical flexibility
15. ✅ Item 17: Dataset size clarification → Documented in methods

---

### 🔴 CRITICAL Experiments Remaining (1 item, 4-8 hours)

**Item 2: Multi-Sample Recording Necessity** (4-8 hours)  
- **Status:** ❌ Design choice not validated
- **Priority:** CRITICAL - Fundamental data collection protocol
- **Impact:** Proves motion artifact elimination is essential
- **Next Action:** Compare 1-sample (no settling) vs 5-10 samples (150ms settling)

**Total Critical Time Remaining:** 4-8 hours (unchanged)

---

### 🟡 HIGH Priority Experiments Remaining (2 items, 1.5-2 hours)

**Item 5: Random Forest 100 Trees Sweep** (1 hour)
- **Status:** ⚠️ Reasonable but could strengthen
- **Priority:** HIGH - Quick win for rigor
- **Impact:** Shows diminishing returns curve

**Item 6: Class Balance Investigation** (30-60 min) ⚠️ **NEWLY ADDED - CRITICAL**
- **Status:** ❌ Claimed but not verified
- **Priority:** HIGH - Data quality validation
- **Impact:** Could invalidate results if severely imbalanced

**Total High-Priority Time:** 1.5-2 hours (up from 1 hour)

---

### 🟢 MEDIUM Priority Experiments Remaining (2 items, 1-2.5 hours)

**Item 7: Confidence Threshold Sweep Visualization** (30 min)
- **Status:** ⚠️ Sweep done but not shown
- **Priority:** MEDIUM - Nice supporting visualization
- **Impact:** Shows accuracy/coverage trade-off

**Item 9: Feature Variance Analysis** (1-2 hours)
- **Status:** ⚠️ Interpretive claim about workspace dependence
- **Priority:** MEDIUM - Strengthens physics explanation
- **Impact:** Quantifies why WS3 fails (edge signature differences)

**Total Medium-Priority Time:** 1-2.5 hours (unchanged)

---

## 📋 Updated Action Plan (Streamlined - Items #1 Complete, #4 Skipped)

### ✅ Phase 0: Critical Text Fixes - COMPLETE
**Status:** ALL 15 text fixes completed during 3-class transition + normalization claim removed

**What Was Fixed:**
- Edge cases now INCLUDED as third class (core contribution)
- All overclaiming language removed
- All numerical inconsistencies resolved
- Normalization "5.8%" claim removed (Feb 8, 2026)
- All missing tables added (Table 3: 5 classifiers, Table 5: binary comparison)
- All factual errors corrected

**Time Invested:** ~10-15 hours (entire 3-class transition)

---

### ✅ Item #1: Hand-Crafted vs Spectrograms - COMPLETE (Feb 8, 2026)

**Status:** ✅ COMPLETE - Added to report Section III.B

**What Was Added:**
1. **Table 5:** Full classifier comparison (5 classifiers × 2 feature types)
2. **Figure 3:** Side-by-side performance plots (hand-crafted vs spectrograms)
3. **Paragraph:** Experimental description and interpretation

**Key Results:**
- Random Forest: 80.9% (features) vs 17.8% (spectrograms) = +63.1%
- Win count: Hand-crafted wins 4/5 classifiers
- Overfitting: Spectrograms show 82.2% gap vs 19.1% for hand-crafted

**Time Invested:** ~3 hours (experiments + report integration)

---

### 🎯 Phase 1: Critical Experimental Validation (4-8 hours remaining)

**REMAINING CRITICAL ITEM:**

1. **Multi-sample Recording Necessity** (Item 2, 4-8h) ⚠️ CRITICAL
   - Test 1-sample vs 5-10-sample protocols
   - Show that 1-sample achieves ~random performance (motion artifacts)
   - **Validates:** Fundamental data collection protocol
   - **Addresses:** Supervisor feedback about moving arm challenge

**Deliverables:**
- 1 new table showing 1-sample vs multi-sample comparison
- 1 paragraph added to Section III.A or new Section IV subsection

---

### 🌟 Phase 2: High-Priority Enhancements (2-3 hours)

**RECOMMENDED NEXT ACTION (QUICK WIN):**

3. **Normalization Comparison** (Item 4, 1-2h) ⚠️ **USER PRIORITY - DO THIS FIRST**
   - Compare StandardScaler, MinMaxScaler, RobustScaler, per-sample, none
   - Show "5.8%" degradation with per-sample normalization
   - **Why first:** User specifically requested, can complete in 1-2 hours
   - **Validates:** Specific numerical claim

4. **Random Forest Tree Count Sweep** (Item 5, 1h)
   - Test 10, 25, 50, 100, 200, 500 trees
   - Plot diminishing returns curve
   - **Validates:** Hyperparameter selection rationale

**Deliverables:**
- 2 new tables or 1 figure
- Brief text additions to Section III

---

### ⭐ Phase 3: Optional Polishing (1-2.5 hours)

**NICE TO HAVE for comprehensive reporting:**

5. **Confidence Threshold Sweep** (Item 7, 30min)
   - Show accuracy/coverage trade-off for 0.60-0.95 thresholds
   - Create Pareto frontier visualization
   - **Impact:** Justifies 0.80 threshold selection

6. **Feature Variance Analysis** (Item 9, 1-2h)
   - Compute edge case feature variance across workspaces
   - Show WS3 has fundamentally different signatures
   - **Impact:** Quantifies physics-based explanation

**Deliverables:**
- 1-2 new figures
- Supplementary material or brief text

---

## ⏱️ Time Estimates (Revised - Item #1 Complete)

### If Doing ALL Remaining Validation:
- Phase 1 (Critical): 4-8 hours (Item 2 only)
- Phase 2 (High-priority): 2-3 hours (Items 4-5)
- Phase 3 (Optional): 1-2.5 hours (Items 7, 9)
- **Grand Total:** 7-13.5 hours (reduced from 9-17.5 hours)

### Recommended Minimum (Submission-Ready):
- Phase 1 (Item 2): 4-8 hours
- Phase 2 Item 4 (User priority): 1-2 hours
- **Minimum Total:** 5-10 hours (reduced from 7-14 hours)

### Quick Win Path (Strong Foundation):
- ✅ Item 1 (Spectrograms): COMPLETE ✅
- Item 4 (Normalization): 1-2 hours
- Item 5 (RF trees): 1 hour
- **Quick Path Total:** 2-3 hours (reduced from 4-7 hours)

---

## 🎯 Next Steps Recommendation (Updated Feb 8, 2026)

**✅ COMPLETED (Feb 8):**
1. ✅ Review 3-class report one final time
2. ✅ Update validation tracking document
3. ✅ **Item 1 COMPLETE:** Hand-crafted vs Spectrograms (2-4h) - Added to Section III.B

**IMMEDIATE (Next Priority):**
4. ➡️ **Item 4:** Normalization comparison (1-2h) - **USER PRIORITY**, quick win
   - Compare StandardScaler, MinMaxScaler, RobustScaler, per-sample, none
   - Validate "5.8%" claim for per-sample normalization
   - Can complete in 1-2 hours using existing features

**This Week (If Time):**
5. Item 2: Multi-sample necessity (4-8h) - Addresses supervisor feedback
6. Item 5: RF trees sweep (1h) - Quick validation

**Optional (Nice to Have):**
7. Items 7, 9: Supporting analyses (1-2.5h)

**Current Status:** 
- ✅ 1 CRITICAL item complete (Item #1)
- ⏳ 1 CRITICAL item remaining (Item #2)
- 🎯 Next recommended: Item #4 (user priority, 1-2h quick win)

**Target:** 5-10 hours minimum to reach submission-ready state (down from 7-14 hours)

---

## 📁 Implementation Notes
3. **Edge case exclusion** (Item 3, 2-3h) - Dataset construction justification

**Total Phase 1 Time:** 8-15 hours

---

### Phase 2: High Priority Enhancements (For Strong Paper)
4. **Normalization comparison** (Item 4, 1-2h) - Already cited specific number ⚠️ **USER PRIORITY**
   - Also resolves Item 35 (removes "5.8% reduction" unsupported claim)
5. **Random Forest trees** (Item 7, 1h) - Quick parameter sweep
6. **Ground truth logic** (Item 18, 2h) - Document automatic labeling

**Total Phase 2 Time:** 4-5 hours

---

### Phase 3: Optional Polishing
7. **Confidence threshold sweep** (Item 8, 30min) - Easy win, good figure
8. **3-class geometry detection** (Item 9, 1-2h) - Interesting scientific question
9. **Feature variance analysis** (Item 20, 1-2h) - "Natural augmentation" claim

**Total Phase 3 Time:** 3-5 hours

---

### Phase 4: NEW CRITICAL EXPERIMENTS (User Requested - Session 14)

**Data Quality & Design Justification Experiments:**

10. **Workspace 1 Data Coverage Investigation** (HIGHEST PRIORITY, 1-2h)
    - Count samples: WS1 vs WS2 vs WS3
    - Visualize spatial coverage for each workspace
    - Check if WS1 has incomplete data → potentially inflated 76.2% accuracy
    - **WHY URGENT:** May invalidate core result if WS1 easier than WS2/WS3

11. **Data Split Rotations** (1-2h)
    - Test all 3 workspace rotation combinations:
      - Current: Train WS2+WS3 → Val WS1 (76.2%)
      - Rotation 2: Train WS1+WS3 → Val WS2 (?)
      - Rotation 3: Train WS1+WS2 → Val WS3 (?)
    - Report: mean ± std across rotations for statistical rigor
    - **Tests if 76.2% is cherry-picked**

12. **Why Multiple Workspaces Needed** (2-3h)
    - Train on 1 workspace vs 2 workspaces, validate on third
    - Show progressive improvement proves necessity

| Training Workspaces | Val Accuracy | Interpretation |
|---------------------|--------------|----------------|
| WS2 only | ~60% expected | Insufficient |
| WS2 + WS3 | 76.2% current | Necessary |

13. **Object Diversity Within Workspace** (2-3h)
    - Train on 1 object vs 2 vs 3 objects
    - Show diversity improves generalization

| Training Objects | Val Accuracy | Interpretation |
|------------------|--------------|----------------|
| A only | ~60% expected | Insufficient |
| A + B + C | 76.2% current | Necessary |

**Total Phase 4 Time:** 6-10 hours

---

## OVERALL TIME ESTIMATE

**If doing ALL validation:**
- Phase 0 (Critical fixes): 2-4 hours ⚠️ **DO FIRST**
- Phase 1 (Original critical): 8-15 hours
- Phase 2 (High priority): 4-5 hours
- Phase 3 (Optional): 3-5 hours  
- Phase 4 (NEW critical): 6-10 hours
- **Grand Total: 23-39 hours**

**Recommended Minimum (Strong Paper):**
- Phase 0 (Critical fixes): 2-4 hours ⚠️ **MUST DO**
- Phase 1 (Experiments 1-3): 8-15 hours
- Phase 2 (Item 4 normalization): 1-2 hours
- Phase 4 (Experiments 10-11): 2-4 hours
- **Minimum Total: 13-25 hours**

---

## 📁 Implementation Notes (Updated Feb 8, 2026)

### Data Requirements
- **Existing data sufficient:** Items 1, 4, 5, 7, 9 (can use current 3-class dataset)
- **May need early experimental data:** Item 2 (if 1-sample protocol data exists)
- **Requires new data collection:** Item 2 (if no early data available - likely needed)

### Code Locations
- Feature extraction: `/acoustic_sensing_starter_kit/dataprocessing/`
- Model training: `/acoustic_sensing_starter_kit/run_modular_experiments.py`
- Results analysis: `/acoustic_sensing_starter_kit/modular_analysis_results_v*/`
- 3-class datasets: `/acoustic_sensing_starter_kit/balanced_collected_data_2025_12_16_v3_*/`

### Report Integration Plan
- **New tables needed:** 2 tables (Items 1, 4)
- **New figures possible:** 2-3 figures (spectrogram comparison, confidence sweep, tree count curve)
- **Text additions:** 1-2 paragraphs per experiment in Section III or IV
- **Page budget:** Report currently fits well in format, minor additions acceptable

---

## 🎯 Quick Reference: What's Left vs What's Done

### ✅ DONE (3-Class Framework Complete):
- All content updated to 3-class (contact, no-contact, edge)
- All figures regenerated (proof of concept, position, object generalization)
- All metrics consistent (77% CV, 60% avg val, 33.3% baseline)
- Edge cases INCLUDED as core contribution (Table 5)
- Binary comparison added showing 56% normalized improvement
- All text fixes completed (15 items)
- All factual errors corrected
- All overclaiming language removed

### ⏳ REMAINING (Experimental Validation):
- **CRITICAL:** Items 1-2 (hand-crafted vs spectrograms, multi-sample necessity) - 6-12 hours
- **HIGH:** Items 4-5 (normalization comparison, RF trees) - 2-3 hours
- **MEDIUM:** Items 7, 9 (confidence sweep, feature variance) - 1.5-2.5 hours
- **TOTAL:** 9-17.5 hours for complete validation

### 🎯 Minimum Viable Validation (7-14 hours):
- Item 1: Spectrograms (2-4h) ← Highest impact
- Item 2: Multi-sample (4-8h) ← Fundamental protocol
- Item 4: Normalization (1-2h) ← User priority

---

**Document Version:** 2.0 (3-Class Update)  
**Last Updated:** February 8, 2026  
**Status:** Report content complete ✅ | Validation experiments pending ⏳  
**Next Action:** Begin Item 1 (Hand-crafted vs Spectrograms)

---

## 📊 Tracking Progress Table (Updated Feb 8, 2026 - Item #1 COMPLETE)

### 🎯 ACTIVE EXPERIMENTS REMAINING

| # | Experiment | Status | Priority | Time Est. | Type | Notes |
|---|------------|--------|----------|-----------|------|-------|
| **1** | **Hand-crafted vs. Spectrograms** | ✅ **COMPLETE** | 🔴 **CRITICAL** | **~3h** | **New Exp** | **Added Table 5 + Figure 3 to Section III.B** |
| 2 | Multi-sample necessity | ⏳ Pending | 🔴 CRITICAL | 4-8h | New Exp | Motion artifacts validation |
| **4** | **Normalization comparison** | ✅ **SKIPPED** | 🟡 ~~HIGH~~ | ~~1-2h~~ | ~~Rerun~~ | **Claim removed (Feb 8) - standard practice** |
| 5 | Random Forest 100 trees | ⏳ Pending | 🟡 HIGH | 1h | Sweep | Quick win - diminishing returns |
| **6** | **Class balance investigation** | ⏳ **Pending** | 🟡 **HIGH** | **30-60min** | **Data Check** | **NEWLY ADDED - Verify 33/33/33 claim** |
| 7 | Confidence threshold sweep | ⏳ Pending | 🟢 MEDIUM | 30min | Analysis | Trade-off visualization |
| 9 | Feature variance analysis | ⏳ Pending | 🟢 MEDIUM | 1-2h | Analysis | Explains WS3 failure |

**Active Items:** 5 experiments remaining (6-12 hours total, up from 5.5-11.5h)

---

### ✅ COMPLETED ITEMS (3-Class Transition + Experimental Validation)

| # | Experiment | Status | Category | Notes |
|---|------------|--------|----------|-------|
| **1** | **Hand-crafted vs. Spectrograms** | ✅ **COMPLETE (Feb 8)** | **Experimental** | **Table 5 + Figure 3 in Section III.B** |
| **3** | **Edge case inclusion** | ✅ **COMPLETE** | **Core Contribution** | **Now Table 5 - 3-class vs binary** |
| **4** | **Normalization comparison** | ✅ **SKIPPED (Feb 8)** | **Methodological** | **Claim removed - standard practice** |
| **8** | **Object A/B/C grouping** | ✅ **COMPLETE** | **Methodology** | **Explained in Section III.D** |
| 6 | Sampling rate / high-freq | ✅ Removed | Text Fix | Removed from 3-class report |
| 13 | "95.2%" claim | ✅ Removed | Text Fix | Changed to general statement |
| 14 | Inference time <1ms | ✅ Removed | Text Fix | All references removed |
| 15 | High-frequency transients | ✅ Removed | Text Fix | Claim removed |
| 16 | Calibration claims | ✅ Simplified | Text Fix | Interpretation removed |
| 22 | "Non-contact regime" | ✅ Never existed | Text Fix | 3-class version never had this |
| 23 | "Material properties" | ✅ Never claimed | Text Fix | 3-class version never had this |
| 24 | "Superior information" | ✅ Never claimed | Text Fix | 3-class version never had this |
| 25 | "500 positions" math | ✅ Clarified | Text Fix | Documented in 3-class methods |
| 26 | 5-classifier table | ✅ Added | Evidence | **Now Table 3 in report** |
| 27 | "75.1% typo" | ✅ Corrected | Text Fix | Changed to 60.0% avg (3-class) |
| 28 | "10+ objects" | ✅ Never claimed | Text Fix | 3-class version never had this |
| 29 | "Impossible" language | ✅ Never used | Text Fix | 3-class uses hedged language |
| 30 | p-value tests | ✅ Removed | Text Fix | Not in 3-class version |

**Completed Items:** 18 items (all Phase 0 text fixes + Item #1 experiment + Item #4 skipped)

---

### ⏸️ SKIPPED / NOT APPLICABLE

| # | Experiment | Status | Reason |
|---|------------|--------|--------|
| 4 | Normalization comparison | ✅ Skip | Claim removed (Feb 8) - standard ML practice |
| 5 | 1cm spatial resolution | ⏸️ Skip | Engineering choice, well-justified |
| 10 | 80/20 vs 5-fold CV | ⏸️ Deferred | Using 5-fold CV in 3-class framework |
| 11 | 150ms settling time | ⏸️ Skip | Engineering parameter, not critical |
| 12 | 5-10 samples variance | ⏸️ Skip | Practical flexibility, clarified in methods |
| 17 | Dataset size precision | ⏸️ Skip | "~15K samples" sufficient, not critical |
| 18 | Ground truth logic | ⏸️ Partial | Edge logic documented, full detail optional |
| 19 | Overfitting proof | ⏸️ Part of #1 | Included in spectrograms experiment |
| 20 | "Natural augmentation" | → Item 9 | Renamed to "Feature variance analysis" |

**Skipped Items:** 9 items (engineering details, merged, or optional)

---

### 📊 Summary by Status

| Status | Count | Time Estimate | Items |
|--------|-------|---------------|-------|
| ⏳ **ACTIVE PENDING** | **5** | **6-12 hours** | 2, 5, 6, 7, 9 |
| ✅ **COMPLETE/SKIPPED** | **18** | - | 1, 3, 4, 8, 13-16, 22-30 |
| ⏸️ **SKIP/DEFER** | **8** | - | 5, 10-12, 17-20 |
| **TOTAL TRACKED** | **31** | - | All original + 1 new item |

---

### 🎯 Recommended Execution Order

**✅ Phase 1a - CRITICAL (COMPLETE):**
1. ✅ **Item 1: Hand-crafted vs Spectrograms (3h)** ← COMPLETE Feb 8, 2026

**⏳ Phase 1b - CRITICAL REMAINING (4-8 hours):**
2. ⏳ Item 2: Multi-sample necessity (4-8h)

**🎯 Phase 2 - HIGH PRIORITY (1.5-2 hours) - IMPORTANT:**
3. ✅ **Item 4: Normalization comparison** ← **SKIPPED** (Claim removed Feb 8)
4. ⏳ Item 5: RF tree count sweep (1h) ← Quick win
5. ⏳ **Item 6: Class balance investigation (30-60min)** ← **NEWLY ADDED - DO FIRST**

**⭐ Phase 3 - OPTIONAL POLISH (1.5-2.5 hours):**
6. ⏳ Item 7: Confidence threshold sweep (30min)
7. ⏳ Item 9: Feature variance analysis (1-2h)

**Legend:**
- ✅ COMPLETE - Finished (Item #1 complete Feb 8, 2026)
- ✅ SKIPPED - Not needed (Item #4 skipped Feb 8, 2026)
- ⏳ Pending - Not started
- ⏸️ Skip/Defer - Not doing

---

**Document Version:** 2.3 (Item #1 Complete, Item #4 Skipped, Item #6 Added)  
**Last Updated:** February 8, 2026  
**Status:** Report content complete ✅ | Item #1 complete ✅ | Item #4 skipped ✅ | 5 experiments remaining ⏳  
**Next Action:** Item #6 Class Balance Investigation (30-60min CRITICAL) THEN Item #2 Multi-sample (4-8h)

