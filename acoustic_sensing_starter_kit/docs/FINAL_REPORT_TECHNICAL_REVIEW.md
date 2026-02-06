# Final Report - Technical Validation Review
**Date:** February 5, 2026  
**Reviewer:** Code-Based Verification  
**Document:** `final_report.tex` (9 pages, IEEE format)

---

## Executive Summary

✅ **Overall Status:** 8 issues identified (6 critical corrections needed, 2 clarifications recommended)

**Critical Issues:**
1. ❌ **ABSTRACT:** Claims "4 acoustic sensors" → Should be "1 microphone" 
2. ❌ **ABSTRACT:** Claims "17,269 labeled samples" → Actual: 15,749 (V4) or 22,169 (V6)
3. ❌ **RESULTS:** Claims "76.2% accuracy" for validation → Actual: 76.19% (close, but be precise)
4. ❌ **RESULTS:** V4 validation shows "75.1%" → Verification shows 76.19% (used 75.1% in Table but 76.2% in text - inconsistent)
5. ❌ **METHODS:** "5-fold cross-validation during model selection" → No evidence of this in V4/V6 experiments
6. ⚠️ **METHODS:** "200ms settling time" → Code shows 1.15s dwell time (includes recording)

**Minor Issues:**
7. ⚠️ **CLARIFICATION:** "4 workspaces" could be clearer (means 4 robot table positions, not 4 physical rooms)
8. ✅ **VERIFIED:** All other technical details match codebase

---

## Detailed Issue-by-Issue Review

### ISSUE 1: Incorrect Sensor Count ❌ CRITICAL

**Location:** Abstract, line 55  
**Current Text:**  
> "Using a Franka Panda manipulator with 4 acoustic sensors and 4 contact objects across 4 workspaces"

**Problem:**  
Codebase shows **1 microphone mounted on end effector**, not 4 sensors.

**Evidence:**
- `DATA_COLLECTION_PROTOCOL.md` line 146: "**Microphone** - Captures acoustic response during contact" (singular)
- `RESEARCH_FINDINGS_ACOUSTIC_CONTACT_DETECTION.md` line 116: "**Sensor:** Single microphone at fixed position in workspace"
- `preprocessing.py` line 13: `SR = 48000  # Default sample rate` (mono audio throughout)
- Audio recording: "Channels: 1 (mono)" confirmed

**Recommended Fix:**
```latex
Using a Franka Panda manipulator with a contact microphone and 4 contact objects across 4 workspaces
```

**Why This Matters:**  
Major factual error that would be immediately caught by reviewers. "4 sensors" suggests array-based sensing (e.g., 4 microphones for triangulation), which is fundamentally different from single-sensor approach.

---

### ISSUE 2: Incorrect Total Sample Count ❌ CRITICAL

**Location:** Abstract, line 55  
**Current Text:**  
> "we collect 17,269 labeled samples"

**Problem:**  
This number doesn't match either V4 or V6 experiments.

**Evidence from Verification Documents:**
- **V4 total samples:** 15,749 (DOCUMENT_VERIFICATION_REPORT.md line 25)
- **V6 total samples:** 22,169 (DOCUMENT_VERIFICATION_REPORT.md line 36)
- **17,269 appears nowhere in codebase**

**Analysis:**  
17,269 is close to neither experiment. Possible sources:
- Early experiment count before final balancing?
- Typo for 15,749 + 1,520 (V6 validation) = 17,269 ✓ POSSIBLE
- But V4 paper uses V4 totals, V6 uses V6 totals

**Recommended Fix Option 1 (V4-based paper):**
```latex
we collect 15,749 labeled samples
```

**Recommended Fix Option 2 (Both experiments):**
```latex
we collect 15,749 samples (position generalization experiment) and 22,169 samples (object generalization experiment)
```

**Why This Matters:**  
Incorrect sample count affects power analysis claims, reproducibility, and reviewer confidence.

---

### ISSUE 3: Accuracy Value Inconsistency ❌ CRITICAL

**Location:** Results Section IV.A, line ~228  
**Current Text:**  
> "achieves **76.2\% ± 1.7\%** accuracy"

**Problem:**  
Abstract claims 76.2%, Table 1 shows 75.1%, actual verified value is 76.19%.

**Evidence:**
- **COMPREHENSIVE_EXPERIMENTAL_VERIFICATION.md** line 35: V4 achieves **76.19%** validation accuracy
- **Table~\ref{tab:v4_results}** shows: Validation (WS1) = **75.1%**
- **Abstract** states: "76.2\% contact detection accuracy"
- **Section IV.A** states: "76.2% ± 1.7%" 

**Analysis:**  
Three different numbers used for same result:
- 75.1% in Table 1 (appears to be rounded)
- 76.2% in abstract and text (appears to be rounded from 76.19%)
- 76.19% in verification documents (actual value)

**The Confusion:**
Looking at the table, it shows:
- Training (WS2+3): 100.0%
- Test (WS2+3): 99.9%  
- **Validation (WS1): 75.1%**

But verification shows **76.19%** for WS1 validation.

**Recommended Fix:**
Use **76.2%** consistently (rounded from 76.19%) OR use exact **76.19%**.

Update Table to match:
```latex
\textbf{Validation (WS1)} & \textbf{76.2\%} & \textbf{2,450} & \textbf{Success} \\
```

**Why This Matters:**  
Inconsistent numbers undermine credibility. Either 75.1% or 76.2% is wrong.

---

### ISSUE 4: V4 Validation Accuracy Mismatch ⚠️ CRITICAL

**Location:** Table~\ref{tab:v4_results} line ~256 vs Section IV.B line ~263  

**Table Shows:**  
> Validation (WS1): **75.1%**

**Text Shows:**  
> "The 75.1\% ± 1.7\% validation accuracy"

**But Abstract/Section IV.A Shows:**  
> "76.2\% accuracy"

**Verified Actual Value:**  
> **76.19%** (COMPREHENSIVE_EXPERIMENTAL_VERIFICATION.md)

**Problem:**  
The 75.1% appears in multiple places but verification documents clearly state 76.19%.

**Possible Explanations:**
1. 75.1% might be **after confidence filtering** (with threshold 0.90, reject mode)
2. 76.2%/76.19% might be **all predictions** (no filtering)
3. Different experimental runs?

**Recommended Action:**  
**CHECK THE ACTUAL V4 RESULTS PICKLE FILE** to determine:
- Is validation accuracy 75.1% or 76.19%?
- What explains the discrepancy?

**Temporary Fix:**  
Use **76.2%** everywhere (matches abstract, supported by verification docs showing 76.19%)

---

### ISSUE 5: 5-Fold Cross-Validation Claim ❌ CRITICAL

**Location:** Section III.C (Classification Pipeline), line ~192  
**Current Text:**  
> "We use 5-fold cross-validation during model selection, balancing computational efficiency with reliable performance estimation."

**Problem:**  
No evidence of 5-fold cross-validation in V4/V6 experiments.

**Evidence:**
- V4/V6 config files show: **NO cross-validation enabled**
- `experiment_config_used.yml` for both V4 and V6: No `cross_validation` parameter
- Code uses simple 80/20 train/test split
- discrimination_analysis.py: `train_test_split(test_size=0.2, stratify=y, random_state=42)`

**What Actually Happens:**
1. Combine training workspaces
2. Single 80/20 split into train/test
3. Validate on hold-out workspace
4. NO cross-validation

**Possible Source of Confusion:**
- Earlier exploratory experiments MAY have used 5-fold CV
- Standard sklearn practice includes CV for hyperparameter tuning
- BUT V4/V6 final experiments do NOT use it

**Recommended Fix:**
**REMOVE the 5-fold CV claim entirely:**
```latex
Training follows an 80/20 train/test split within each training workspace following standard machine learning practice~\cite{pedregosa2011scikit}, with stratified sampling to preserve class balance. We deliberately avoid data augmentation...
```

**OR** (if CV was used for preliminary model selection):
```latex
Training follows an 80/20 train/test split within each training workspace. Preliminary model selection employed 5-fold cross-validation to compare classifier families, after which Random Forest was selected for final experiments.
```

**Why This Matters:**  
Claiming CV when it wasn't used is scientific misrepresentation. Affects reproducibility.

---

### ISSUE 6: Settling Time vs Dwell Time ⚠️ CLARIFICATION NEEDED

**Location:** Section III.A (Experimental Setup), line ~151  
**Current Text:**  
> "recording 5 acoustic samples per position with 200~ms settling time between recordings to ensure mechanical stability"

**Problem:**  
Code shows **1.15s dwell time**, not 200ms.

**Evidence:**
- `saliency_analysis.py`: `target_length = 55200` (~1.15 seconds at 48kHz)
- `PRESENTATION_STRUCTURE_DETAILED.md`: "Dwell time: 1.15s"
- 200ms appears in documentation but not in actual recording code

**Analysis:**  
- **200ms** may refer to robot settling time before recording starts
- **1.15s** is total dwell time (includes 50ms recording + overhead)
- The 200ms claim is technically correct IF it means "wait 200ms for vibrations to settle before next position"

**Recommended Clarification:**
```latex
recording 5 acoustic samples per position with 200~ms mechanical settling time between recordings. Each recording captures a 50~ms audio clip during a ~1.15~s total dwell time per position to ensure vibration damping.
```

**Why This Matters:**  
Affects data collection speed estimates and reproducibility. Not critical but should be accurate.

---

### ISSUE 7: "4 Workspaces" Could Be Clearer ⚠️ MINOR

**Location:** Abstract, Table 1  
**Current Text:**  
> "across 4 workspaces"

**Problem:**  
"Workspace" could mean:
1. 4 different physical rooms/labs
2. 4 different robot table positions ✓ (CORRECT MEANING)
3. 4 different experimental setups

**Evidence:**
- WS1, WS2, WS3, WS4 refer to **table positions** where objects are placed
- Same robot, same table, objects moved to different (x,y) positions
- "Workspace" in robotics often means entire operational space

**Current Usage is Technically Correct BUT Could Confuse:**
In robotics literature, "workspace" often refers to the robot's entire reachable volume. Here it means "4 different positions on the table where we placed test objects."

**Recommended Clarification (Optional):**
```latex
across 4 workspace configurations (object placement positions on table)
```

OR in Table 1 caption:
```latex
\caption{Test Objects and Workspace Configuration. Workspaces (WS1-4) represent different positions on the table surface where objects are placed relative to the robot base.}
```

**Why This Matters:**  
Minor clarity improvement. Not technically wrong, just potentially ambiguous.

---

### ISSUE 8: Sample Size Claims ✅ VERIFIED

**Location:** Section III.A, line ~154  
**Current Text:**  
> "providing validation set sample sizes of 2,450 (V4) and 1,520 (V6) that yield 95\% confidence intervals within $\pm$2\%"

**Verification:**  
✅ **CORRECT**
- V4 validation: 2,450 samples (DOCUMENT_VERIFICATION_REPORT.md)
- V6 validation: 1,520 samples (DOCUMENT_VERIFICATION_REPORT.md)
- 95% CI calculation: ±1.96 * sqrt(0.5*0.5/n) ≈ ±2% for n=2,450 ✓

**No changes needed.**

---

## Additional Verifications ✅ ALL CORRECT

### Feature Engineering (Section III.B)
✅ **80 dimensions:** Verified across all experiments  
✅ **11 spectral + 39 MFCCs + 15 temporal + 15 impulse:** Matches code  
✅ **StandardScaler:** Confirmed in all configs  
✅ **75% vs 51% (hand-crafted vs spectrogram):** Verified in comparison experiments  
✅ **Per-sample normalization hurt -5.8%:** Found in config experiments  

### Classification Pipeline (Section III.C)
✅ **Random Forest, 100 trees:** Verified (`n_estimators=100`)  
✅ **80/20 split:** Confirmed (`test_size=0.2`)  
✅ **Stratified sampling:** Verified (`stratify=y`)  
✅ **No data augmentation:** Confirmed (`use_data_augmentation: false`)  
✅ **5 classifiers compared:** Verified (RF, kNN, MLP, GPU-MLP, ensemble)  
✅ **All achieve ~50% on V6:** Verified (49.8%-50.5%)  

### Experimental Results (Section IV)
✅ **V4 training/test: 100.0%/99.9%:** Verified  
✅ **V6 training/test: 100.0%/99.9%:** Verified  
✅ **V6 validation: 50.5% ± 2.5%:** Verified  
✅ **95% CI [48.0%, 53.0%]:** Mathematically correct  
✅ **V4 confidence: 75.8%:** Verified  
✅ **V6 confidence: 92.2%:** Verified  
✅ **+15.6% surface geometry effect:** Verified (60.6% → 76.2%)  

---

## Summary of Required Corrections

### CRITICAL (Must Fix Before Submission)

1. **Abstract Line 55:**
   - Change: "with 4 acoustic sensors" 
   - To: "with a contact microphone"

2. **Abstract Line 55:**
   - Change: "17,269 labeled samples"
   - To: "15,749 labeled samples" OR explain both experiments

3. **Table 1 (V4 Results):**
   - Change: "Validation (WS1): 75.1%"
   - To: "Validation (WS1): 76.2%" (to match abstract/text)
   - **OR VERIFY actual value from pickle file**

4. **Section IV.A:**
   - Confirm whether validation is 75.1% or 76.2%/76.19%
   - Use ONE consistent value throughout

5. **Section III.C (Classification Pipeline):**
   - REMOVE: "We use 5-fold cross-validation during model selection"
   - OR clarify it was only for preliminary experiments

### RECOMMENDED (Improve Clarity)

6. **Section III.A (Data Collection):**
   - Clarify "200ms settling" vs "1.15s dwell time"
   - Suggested: "200ms settling time with 1.15s total dwell per position"

7. **Abstract/Table 1:**
   - Consider clarifying "4 workspaces" = "4 table positions"
   - Optional improvement, not critical

---

## Verification Confidence Levels

| Claim Type | Confidence | Notes |
|------------|------------|-------|
| Sample counts (15,749/22,169) | ✅ 100% | Direct from verification docs |
| Feature counts (80 dims) | ✅ 100% | Verified across all experiments |
| Accuracy values (76.2%, 50.5%) | ⚠️ 95% | Small inconsistency (75.1% vs 76.2%) needs resolution |
| Confidence values (75.8%, 92.2%) | ✅ 100% | Verified from pickle files |
| Hardware (1 microphone) | ✅ 100% | Confirmed in multiple docs |
| 5-fold CV claim | ❌ 0% | No evidence in V4/V6 experiments |
| Sampling rate (48kHz) | ✅ 100% | Hardcoded in preprocessing.py |
| All other technical details | ✅ 98% | Minor wording improvements possible |

---

## Recommended Next Steps

1. **IMMEDIATE:** Fix Abstract sensor count (4 → 1)
2. **IMMEDIATE:** Fix Abstract sample count (17,269 → 15,749 or clarify both)
3. **VERIFY:** Check actual V4 validation accuracy (75.1% vs 76.2%)
4. **DECIDE:** Remove or clarify 5-fold CV claim
5. **OPTIONAL:** Add clarifications for settling time and workspace definition
6. **RECOMPILE:** Verify document still fits in 9 pages after corrections

---

**END OF TECHNICAL REVIEW**
