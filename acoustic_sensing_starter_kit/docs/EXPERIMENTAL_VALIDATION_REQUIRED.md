# Experimental Validation Required

**Document Purpose:** This document tracks all claims, design choices, and methodological decisions in the final report that currently lack empirical validation. Each item represents an experiment needed to rigorously justify statements made in the paper.

**Priority Levels:**
- 🔴 **CRITICAL**: Core claims affecting paper validity - must complete before submission
- 🟡 **HIGH**: Significant methodological choices - strongly recommended for rigor
- 🟢 **MEDIUM**: Supporting details that enhance credibility
- ⚪ **LOW**: Nice-to-have validation for comprehensive reporting

---

## 🔴 CRITICAL Experiments (Must Complete)

### 1. Hand-Crafted Features vs. Spectrograms
**Claim (Section III.B):** 
> "our compact 80-dimensional representation significantly outperforms spectrograms (75% vs. 51% validation accuracy)"

**Current Status:** Claim stated but no evidence provided in paper.

**Required Experiment:**
- Train Random Forest on mel-spectrogram features (10,240 dimensions)
- Compare performance metrics for both V4 and V6 experiments
- Report train/test/validation accuracy for both feature types

**Expected Table:**
| Feature Type | Dimensionality | Train Acc. | Test Acc. | Val Acc. (V4) | Val Acc. (V6) |
|--------------|----------------|------------|-----------|---------------|---------------|
| Mel-spectrogram | 10,240 | ? | ? | ~51% | ? |
| Hand-crafted | 80 | 100% | 99.9% | 76.2% | 50.5% |

**Where to Add:** Section III.B (Feature Engineering) or Section IV as ablation study

**Estimated Time:** 2-4 hours (re-run pipeline with spectrogram features)

---

### 2. Multi-Sample Recording Necessity (Motion Artifact Elimination)
**Claim (Section III.A):**
> "recording 5--10 acoustic samples per position with 150~ms mechanical settling time between recordings"

**Implicit Claim:** Single-sample recording fails due to robot motion contamination.

**Current Status:** Design choice explained but not validated.

**Required Experiment:**
- **Baseline:** 1 sample per position, no settling time (immediate recording during/after motion)
- **Current:** 5-10 samples per position with 150ms settling time
- Compare model accuracy for both protocols on V4 experiment

**Expected Results:**
| Protocol | Samples/Position | Settling Time | Train Acc. | Test Acc. | Val Acc. (V4) |
|----------|------------------|---------------|------------|-----------|---------------|
| Baseline | 1 | 0ms | ? | ? | ~50% (expected) |
| Current | 5-10 | 150ms | 100% | 99.9% | 76.2% |

**Hypothesis:** Baseline should show random-chance performance (~50%) or severe overfitting, proving that motion artifact elimination is essential.

**Where to Add:** Section III.A or create new Section IV subsection "Motion Artifact Validation"

**Estimated Time:** 4-8 hours (may need to re-collect data with 1-sample protocol, or use early experimental data if available)

---

### 3. Edge Case Exclusion Justification
**Claim (Section III.A & III.D):**
> "All edge cases where the contact finger partially overlaps object boundaries are excluded to maintain clean binary labels."

**Current Status:** Design choice stated but benefit not proven.

**Required Experiment:**
- **Model A:** Trained with edge cases included
- **Model B:** Trained with edge cases excluded (current)
- Compare accuracy on clean validation data (non-edge cases only)

**Expected Results:**
| Dataset | Train Samples | Val Acc. (V4) | Interpretation |
|---------|---------------|---------------|----------------|
| With edges | ~12,000 | ? | Edge cases add noise? |
| Without edges | ~10,639 | 76.2% | Current baseline |

**Hypothesis:** Including edge cases should reduce accuracy OR have minimal effect (either outcome is scientifically interesting).

**Where to Add:** Section III.D (Evaluation Strategy) with 1-2 sentence summary

**Estimated Time:** 2-3 hours (re-run pipeline with edge cases included in dataset)

---

## 🟡 HIGH Priority Experiments (Strongly Recommended)

### 4. StandardScaler vs. Alternative Normalization ⚠️ NEEDS EXPERIMENT
**Claim (Section III.B):**
> "We selected StandardScaler over alternatives after experimental validation showed that per-sample normalization reduced accuracy by 5.8%"

**Current Status:** Specific number cited (5.8%) but no details provided.

**Required Experiment:**
- Compare normalization methods: StandardScaler, MinMaxScaler, RobustScaler, per-sample normalization, no normalization
- Report V4 validation accuracy for each

**Expected Table:**
| Normalization | Val Acc. (V4) | Δ from Baseline |
|---------------|---------------|-----------------|
| StandardScaler (current) | 76.2% | - |
| Per-sample | 70.4% | -5.8% |
| MinMaxScaler | ? | ? |
| RobustScaler | ? | ? |
| None | ? | ? |

**Where to Add:** Section III.B or supplementary material

**Estimated Time:** 1-2 hours (already have features, just re-normalize and re-train)

**Priority:** HIGH - Specific claim made that requires validation

---

### 5. 1cm Spatial Resolution Justification
**Claim (Section III.A):**
> "1~cm spatial resolution, chosen to match the acoustic finger's contact area (approximately 1~cm $\times$ 0.25~cm)"

**Current Status:** Pragmatic choice, but not rigorously optimized.

**Possible Validation (Optional):**
- Test 0.5cm, 1cm, 2cm resolutions
- Compare geometric reconstruction quality and data collection time

**Recommendation:** This is a practical engineering choice with reasonable justification (matches sensor footprint). **Low priority for validation** unless reviewers specifically question it.

**Estimated Time:** N/A (would require full data re-collection, not recommended)

---

### 6. 48kHz Sampling Rate Justification
**Claim (Section III.A):**
> "Acoustic signals are captured at 48~kHz sampling rate, providing a Nyquist frequency of 24~kHz with 4~kHz anti-aliasing margin"

**Current Status:** Standard audio engineering practice, but choice not validated for this specific application.

**Possible Validation:**
- Downsample to 16kHz, 24kHz, 32kHz and compare accuracy
- Test if high-frequency content (>10kHz) actually matters for contact detection

**Recommendation:** This is based on audio engineering standards (Nyquist theorem). **Medium priority** - could be interesting to show frequency spectrum analysis proving high frequencies matter.

**Estimated Time:** 2-3 hours (downsample existing audio, re-extract features, re-train)

---

### 7. Random Forest 100 Trees - More Rigorous Justification
**Current Claim (Section III.C):**
> "We use 100 trees as a standard default configuration, as preliminary experiments showed all top-performing models (Random Forest, MLP, ensemble methods) achieved comparable validation performance (76% ± 1%)"

**Current Status:** Reasonable explanation, but could be strengthened.

**Possible Enhancement:**
- Show hyperparameter sweep: 10, 25, 50, 100, 200, 500 trees
- Plot accuracy vs. tree count (diminishing returns curve)

**Expected Finding:** 100 trees likely in plateau region where additional trees yield <0.5% improvement.

**Where to Add:** Supplementary material or brief mention

**Estimated Time:** 1 hour (quick parameter sweep)

---

## 🟢 MEDIUM Priority Experiments (Nice to Have)

### 8. Confidence Threshold Selection - Full Sweep
**Claim (Section III.C):**
> "selected through empirical evaluation of a range of threshold values (0.60, 0.70, 0.80, 0.90, 0.95)"

**Current Status:** States that sweep was performed, but no results shown.

**Possible Enhancement:**
- Show table of coverage vs. accuracy for each threshold
- Plot Pareto frontier (coverage vs. accuracy trade-off)

**Expected Table:**
| Threshold | Coverage (V4) | Accuracy (V4) | Coverage (V6) | Accuracy (V6) |
|-----------|---------------|---------------|---------------|---------------|
| 0.60 | ~95% | ~73% | ~98% | ~50% |
| 0.70 | ~85% | ~74% | ~95% | ~50% |
| 0.80 | ~60% | ~75% | ~85% | ~50% |
| 0.90 | ~20% | ~76% | ~60% | ~50% |
| 0.95 | ~10% | ~77% | ~57% | ~50% |

**Where to Add:** Supplementary material or brief figure

**Estimated Time:** 30 minutes (already have predictions, just apply different thresholds)

---

### 9. Why Objects A/C Grouped as Same "Contact" Class
**Claim (Section III.D):**
> "contact samples come from objects A (cutout) and C (full contact)"

**Current Status:** Implicit assumption that both represent "contact" despite different geometries.

**Possible Validation:**
- Could object A and C be distinguished acoustically?
- Test 3-class classification: {no-contact, cutout-contact, full-contact}

**Scientific Question:** Does acoustic sensing capture surface geometry differences, or only binary contact?

**Estimated Time:** 1-2 hours (re-train with 3 classes)

---

### 10. 80/20 Split vs. Cross-Validation
**Claim (Section III.C):**
> "Training follows an 80/20 train/test split within each training workspace"

**Current Status:** Standard practice, but cross-validation would be more rigorous.

**Action:** User mentioned this will be corrected in implementation. **Leave for now, update later.**

---

## ⚪ LOW Priority Experiments (Optional)

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

## Summary Statistics

**Total Validation Items:** 30 (20 original + 10 new from comprehensive review)

### By Priority:
- 🔴 **CRITICAL:** 9 items requiring immediate action
  - **Original Items 1-3:** Core experimental validations (8-15 hours)
  - **NEW Items 22, 25, 26, 27:** Factual errors & missing critical data (2-4 hours)
  - ✅ Items 13-16: RESOLVED (text revisions applied)
  
- 🟡 **HIGH:** 7 items (strongly recommended for rigor)
  - Item 4: Normalization comparison ⚠️ **User wants this shown**
  - Item 7: Random Forest trees  
  - Item 18: Ground truth logic
  - **NEW Items 23, 24, 28, 29, 30:** Overclaiming, unjustified numbers (1-2 hours)
  
- 🟢 **MEDIUM:** 6 items (nice to have, strengthen arguments)
  - Items 8-9, 19-20: Original medium priority
  - **NEW Item 21:** Sample count precision
  
- ⚪ **LOW:** 3 items (optional engineering details)
  - Items 11-12, 17

**Completed/Removed:** 6 items
- ✅ Item 5: Spatial resolution (skipped - not needed)
- ✅ Item 6: Sampling rate / high-freq transients (removed from report)
- ✅ Item 10: Cross-validation (deferred to future)
- ✅ Item 13: 95.2% claim (revised to general statement)
- ✅ Item 14: Inference time (removed from report)
- ✅ Item 16: Calibration claims (simplified, removed interpretation)

**Estimated Time Investment:**
- **CRITICAL text fixes (NEW):** 2-4 hours (items 22, 25, 26, 27)
- **Critical experiments (original):** 8-15 hours (items 1-3)
- **High priority experiments:** 3-5 hours (items 4, 7, 18)
- **High priority text fixes (NEW):** 1-2 hours (items 23, 24, 28, 29, 30)
- **Medium priority enhancements:** 3-5 hours (items 8-9, 19-21)
- **Total for rigorous paper:** 17-31 hours

### NEW CRITICAL ISSUES FROM COMPREHENSIVE REVIEW:

**Must Fix IMMEDIATELY (Before Any Submission):**
1. ❌ Item 22: Delete "non-contact regime" claim (FACTUALLY WRONG)
2. ❌ Item 25: Explain "500 positions" math error (10min)
3. ❌ Item 26: Add 5-classifier comparison table (30min-2h)
4. ❌ Item 27: Fix 75.1% → 76.2% typo (2min)

**Should Fix Before Submission:**
5. Item 23: Remove "material properties" overclaim (5min)
6. Item 24: Remove "or superior" unsupported claim (5min)
7. Item 28: Justify "10+ objects" or add hedge (10min)
8. Item 29: Change "impossible" to "extremely challenging" (5min)
9. Item 30: Add statistical test details for p-values (5-30min)

**Quick Wins (High Impact, Low Effort):**
1. ✅ ~~Remove "95.2%" claim~~ DONE
2. ✅ ~~Remove inference time claims~~ DONE  
3. ✅ ~~Remove calibration interpretation~~ DONE
4. **Normalization comparison** (1-2 hours) ⚠️ **USER PRIORITY**
5. **Confidence threshold sweep table** (30 min) - optional
6. **Random Forest tree count sweep** (1 hour)

---

## Recommended Action Plan

### ⚠️ IMMEDIATE PRIORITY: Critical Text Fixes (2-4 hours)

**Phase 0: MUST FIX BEFORE ANYTHING ELSE**
These are factual errors and inconsistencies that must be corrected immediately:

1. **Item 22:** Delete "non-contact regime" claim (FACTUALLY INCORRECT) - 10 min
2. **Item 27:** Fix 75.1% → 76.2% typo in conclusion - 2 min
3. **Item 23:** Remove "material properties" overclaim - 5 min
4. **Item 24:** Remove "or superior information density" - 5 min
5. **Item 25:** Investigate and explain "500 positions" math - 1-2 hours
6. **Item 26:** Add 5-classifier comparison table for V6 - 30min-2h (if data exists)
7. **Item 28:** Add hedge to "10+ objects" recommendation - 10 min
8. **Item 29:** Change "impossible" to "extremely challenging" - 5 min
9. **Item 30:** Add statistical test details for p>0.5 - 5-30 min

**Total Phase 0 Time:** 2-4 hours

---

### Phase 1: Critical Experimental Validation (Before Submission)
1. **Hand-crafted vs. Spectrograms** (Item 1, 2-4h) - Core methodological choice
   - Also resolves Items 33, 66 (removes "75% vs 51%" unsupported claim)
2. **Multi-sample recording necessity** (Item 2, 4-8h) - Fundamental design validation
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

## Implementation Notes

### Data Requirements
- **Existing data sufficient:** Experiments 1, 3, 4, 7, 8, 9 (can use current dataset)
- **May need early experimental data:** Experiment 2 (if 1-sample protocol data exists)
- **Requires new data collection:** Experiment 2 (if no early data available)

### Code Locations
- Feature extraction: `/acoustic_sensing_starter_kit/dataprocessing/`
- Model training: `/acoustic_sensing_starter_kit/run_modular_experiments.py`
- Results analysis: `/acoustic_sensing_starter_kit/modular_analysis_results_v*/`

### Figure/Table Planning
- **New tables needed:** 5 tables (experiments 1, 2, 3, 4, 8)
- **New figures possible:** 2-3 figures (confidence sweep, tree count curve, frequency analysis)
- **Page budget:** Need to ensure still fits in 9 pages

---

## Tracking Progress

| # | Experiment | Status | Priority | Time Est. | Type | Notes |
|---|------------|--------|----------|-----------|------|-------|
| 1 | Handcrafted vs. Spectrograms | ⏳ Pending | 🔴 CRITICAL | 2-4h | New Exp | Core claim |
| 2 | Multi-sample necessity | ⏳ Pending | 🔴 CRITICAL | 4-8h | New Exp | Motion artifacts |
| 3 | Edge case exclusion | ⏳ Pending | 🔴 CRITICAL | 2-3h | New Exp | Dataset design |
| **4** | **Normalization comparison** | ⚠️ **PRIORITY** | 🟡 **HIGH** | **1-2h** | **Rerun** | **User wants shown** |
| 5 | Spatial resolution | ⏸️ Skip | - | N/A | - | Not needed |
| 6 | Sampling rate / high-freq | ✅ Removed | - | N/A | - | Removed from report |
| 7 | Random Forest trees | ⏳ Pending | 🟡 HIGH | 1h | Sweep | Optional |
| 8 | Confidence threshold | ⏳ Pending | 🟢 MEDIUM | 30min | Analysis | Optional |
| 9 | 3-class geometry | ⏳ Pending | 🟢 MEDIUM | 1-2h | New Exp | Optional |
| 10 | Cross-validation | ⏸️ Deferred | - | N/A | - | Future work |
| 11 | Settling time | ⏸️ Skip | ⚪ LOW | N/A | - | Not needed |
| 12 | Sample count variance | ⏸️ Skip | ⚪ LOW | N/A | - | Clarify only |
| **13** | **95.2% claim** | ✅ **DONE** | - | - | **Text Edit** | **Revised to general** |
| **14** | **Inference time <1ms** | ✅ **DONE** | - | - | **Removed** | **All refs removed** |
| **15** | **High-freq transients** | ✅ **DONE** | - | - | **Removed** | **Claim removed** |
| **16** | **Calibration claims** | ✅ **DONE** | - | - | **Simplified** | **Interpretation removed** |
| 17 | Dataset size clarification | ⏸️ Skip | ⚪ LOW | 10min | Text | Not critical |
| 18 | Ground truth logic | ⏳ Pending | � HIGH | 2h | Validate | Document logic |
| 19 | Overfitting proof | ⏳ Pending | 🟢 MEDIUM | 0h | Part of #1 | Included in #1 |
| 20 | Feature variance analysis | ⏳ Pending | 🟢 MEDIUM | 1-2h | Analysis | Optional |

**Legend:**
- ⏳ Pending - Not started
- ⚠️ PRIORITY - User specifically requested
- ✅ DONE - Completed and applied
- ⏸️ Skip/Deferred - Not doing

---

## Immediate Next Steps

### ✅ Completed This Session (Text Revisions):
1. ✅ Removed "95.2% average accuracy" → Changed to "best in-distribution performance"
2. ✅ Removed all "inference time <1ms" claims (Section IV.A, Conclusion)
3. ✅ Removed "high-frequency contact transients up to 20kHz" claim
4. ✅ Removed "well-calibrated" interpretation from confidence analysis
5. ✅ Simplified confidence filtering to just threshold selection (0.60-0.95)
6. ✅ Updated Figure 8 caption to remove calibration language

### ⚠️ User Priority Experiments:
1. **Normalization comparison** (Item 4) - User wants to show difference
   - Compare: StandardScaler vs. per-sample vs. others
   - Validate the "5.8% reduction" claim
   - Time: 1-2 hours

2. **Data augmentation effects** (mentioned by user)
   - Not currently in validation list
   - Should add as new item if user wants validation

### 🔴 Still Critical (Must Do Before Submission):
1. **Handcrafted vs. Spectrograms** (Item 1) - Core feature choice
2. **Multi-sample recording** (Item 2) - Motion artifact claim
3. **Edge case exclusion** (Item 3) - Dataset construction

---

**Document Version:** 1.0  
**Last Updated:** February 6, 2026  
**Next Review:** After completing critical experiments

---

## 📊 UPDATED COMPREHENSIVE TRACKING TABLE (After Feb 6 Review)

### ⚠️ CRITICAL PATH - DO THESE FIRST (4-8 hours total)

**Phase 0A: Immediate Text Fixes (45 minutes):**
- Item 22: Delete "non-contact regime" claim → FACTUAL ERROR
- Item 27: Fix 75.1% → 76.2% typo → WRONG NUMBER  
- Item 23: Remove "material properties" → OVERCLAIM
- Item 24: Remove "or superior" → UNSUPPORTED
- Item 28: Hedge "10+ objects" → UNJUSTIFIED
- Item 29: Change "impossible" → "extremely challenging"
- Item 30: Add statistical test details → INCOMPLETE

**Phase 0B: Critical Investigations (3-7 hours):**
- Item 25: Investigate "500 positions" math (1-2h) → MATH ERROR
- Item 26: Add 5-classifier table (30min-2h) → MISSING EVIDENCE
- Item 31: WS1 coverage investigation (1-2h) → **MAY INVALIDATE 76.2%**
- Item 32: Data split rotations (1-2h) → **TEST IF CHERRY-PICKED**

**TOTAL PHASE 0: 4-8 hours ← MUST COMPLETE BEFORE OTHER WORK**

---

### All Validation Items by Status

**🔴 CRITICAL - URGENT (16 items, 18-33 hours):**
- Text fixes: Items 22, 25, 26, 27 (2-5h)
- Original experiments: Items 1, 2, 3 (8-15h)
- User-requested: Items 31, 32, 33, 34 (6-10h)
- Missing evidence: Item 26 (30min-2h)

**🟡 HIGH PRIORITY (10 items, 5-9 hours):**
- Text fixes: Items 23, 24, 28, 29, 30 (30min-1h)
- Experiments: Items 4, 7, 18 (4-5h)
- User wants: Item 4 (normalization) (1-2h)

**🟢 MEDIUM PRIORITY (5 items, 3-6 hours):**
- Items 8, 9, 19, 20, 21 (3-6h)

**✅ COMPLETED (10 items):**
- Items 5, 6, 10, 13, 14, 15, 16

**⏸️ SKIPPED/DEFERRED (3 items):**
- Items 11, 12, 17

**GRAND TOTAL: 34 items tracked**

---

## 🚨 MOST URGENT ACTIONS (Do in this order)

1. **DELETE "non-contact regime"** (10 min) → Section I, paragraph 3
2. **FIX 75.1% → 76.2%** (2 min) → Section V.A conclusion
3. **Remove overclaims** (25 min) → Items 23, 24, 28, 29, 30
4. **Investigate "500 positions"** (1-2h) → Explain math in Section III.A
5. **Check WS1 data coverage** (1-2h) → **HIGHEST PRIORITY - May invalidate core result**
6. **Add 5-classifier table** (30min-2h) → Section IV.C
7. **Run split rotations** (1-2h) → Test if 76.2% cherry-picked

**After these 7 items (4-8 hours), proceed with original experiments (Items 1-3)**

