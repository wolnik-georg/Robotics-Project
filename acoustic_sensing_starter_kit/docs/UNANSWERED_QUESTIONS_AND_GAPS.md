# Unanswered Questions & Gaps in Final Report

**Document:** final_report.tex  
**Review Date:** February 6, 2026  
**Purpose:** Identify questions likely to arise from reviewers/examiners that are not yet addressed in the report

---

## 📊 PRIORITY RANKING SYSTEM

- 🔴 **CRITICAL** - Likely to be asked by any reviewer, fundamental to understanding
- 🟡 **HIGH** - Commonly asked, strengthens scientific rigor
- 🟢 **MEDIUM** - Nice to have, demonstrates thoroughness
- ⚪ **LOW** - Optional, mostly for completeness

---

## 🔴 CRITICAL GAPS (Must Address)

### 1. Why Only ONE Object in V6 Validation? 🔴
**Question:** "You train on 3 objects (A,B,C) but validate on only 1 object (D). Why not use multiple novel objects to test object generalization? Isn't this a single-point evaluation?"

**Current Gap:** Paper doesn't justify why object D alone is sufficient for testing object generalization. Single object = single data point.

**Likely Follow-up:** "How do you know the 50% result generalizes to OTHER novel objects? Maybe object D is just particularly difficult?"

**Suggested Answer:**
- Resource constraints (time, materials)
- Object D represents a different cutout pattern (structural variation)
- All 5 classifiers fail identically → suggests systematic failure, not object-specific
- BUT: Acknowledge this is a limitation (need 5-10 novel objects for robust conclusion)

**Where to Add:** Section III.D (Evaluation Strategy) - add 1 sentence justifying single holdout object + acknowledge limitation in Section V.C (if you add Limitations subsection)

---

### 2. Why 100 Trees in Random Forest? 🔴
**Question:** "You state 'we did not conduct exhaustive hyperparameter tuning for the number of trees' - why not? How do you know 100 is appropriate?"

**Current Gap:** Paper mentions no tuning but doesn't justify why that's acceptable.

**Current Justification (in paper):** "All five classifiers achieve identical performance (~50%) on object generalization tasks, indicating the performance bottleneck lies in feature representation rather than model capacity"

**Issue:** This justifies not tuning for V6, but what about V4 where you DO get good performance?

**Suggested Answer:**
- 100 trees is a common default (cite sklearn documentation)
- Preliminary experiments showed diminishing returns beyond 50-100 trees
- Learning curves (if you have them) show convergence
- Main point: V6 failure is classifier-agnostic → proves it's not about hyperparameters

**Where to Add:** Section III.C (Classification Pipeline) - expand by 1-2 sentences

---

### 3. How Were Confidence Thresholds (0.90, 0.95) Selected? 🔴
**Question:** "Why did you choose 90% and 95% confidence thresholds? Did you test other values? What's the justification?"

**Current Gap:** Paper mentions these thresholds but doesn't explain selection process.

**Suggested Answer:**
- Common industry standards for high-reliability applications
- 90% provides balance between coverage and accuracy (cite your results)
- 95% represents very high confidence bar
- Evaluated range [0.5, 0.99] in steps of 0.05 (if true)

**Where to Add:** Section III.C (Classification Pipeline) - expand confidence filtering paragraph

---

### 4. Why Exactly 50ms Recording Window? 🔴
**Question:** "You mention 'typical 10-30ms contact transients' - why use 50ms window? Why not 30ms or 100ms? How was this chosen?"

**Current Justification (in paper):** "captures contact transients (typically 10-30ms duration) while providing 20Hz frequency resolution"

**Gap:** Doesn't explain HOW you determined 50ms is optimal. Was there testing?

**Suggested Answer:**
- Empirical observation of transient duration in pilot studies
- 50ms = 2× longest transient (10-30ms) + margin
- Trade-off: Longer window = better frequency resolution (20Hz) but more noise
- Shorter window = risk of truncating transient
- 50ms is common in acoustic event detection literature (cite if true)

**Where to Add:** Section III.A (Experimental Setup) - expand the 50ms justification

---

### 5. What is "Contact Microphone" Exactly? 🔴
**Question:** "What type of contact microphone? What are its specifications? Frequency response? Sensitivity? Model/manufacturer?"

**Current Gap:** Paper says "contact microphone" but no specifications.

**Impact:** Reproducibility issue - someone cannot replicate this without knowing the sensor.

**Suggested Answer:** (You need to provide actual specs)
- Model: [Specific model number]
- Frequency response: [XX Hz - XX kHz]
- Sensitivity: [XX mV/Pa]
- Mounting: [adhesive/magnetic/mechanical coupling]
- Cost: [~$XX USD] (demonstrates accessibility)

**Where to Add:** Section III.A (Experimental Setup) - expand "contact microphone" to full specification

---

### 6. How Do You KNOW Transients Are 10-30ms? 🔴
**Question:** "You state contact transients are 'typically 10-30ms duration' - where does this number come from? Did you measure this?"

**Current Gap:** Presented as fact but no citation or measurement shown.

**Suggested Answer:**
- Measured from pilot recordings (show histogram if you have data)
- Or: "Consistent with impact transient literature [CITE]"
- Or: Visual inspection of waveforms (Fig. X shows examples)

**Where to Add:** Section III.A - either add figure showing waveforms with transient duration, or cite literature

---

### 7. Why Is Object B "Empty" Used for No-Contact? 🔴
**Question:** "Why does 'empty workspace' (object B) provide no-contact samples? Shouldn't the robot still contact the wooden surface?"

**Current Gap:** Confusing terminology - "empty" vs "no-contact"

**Clarification Needed:**
- Is Object B a plain wooden board that IS contacted (but has no shapes)?
- Or is Object B literally empty space where robot doesn't contact anything?
- If it's a plain board, why is that "no-contact"?

**Likely Intent:** Object B = plain wooden surface with no shapes → "no-contact" means "no contact with SHAPES" (but still contacts surface)

**Problem:** This is confusing. "Contact vs no-contact" suggests binary contact detection, but you're actually detecting "shape contact vs plain surface contact"

**Suggested Fix:**
- Clarify that "no-contact" = "plain surface" (not literally no contact)
- Or redefine problem as "shape detection" rather than "contact detection"
- Explain why plain surface = negative class

**Where to Add:** Section III.D (Evaluation Strategy) - clarify what "no-contact" means physically

---

## 🟡 HIGH PRIORITY (Strongly Recommended)

### 8. Why 80/20 Train/Test Split? 🟡
**Question:** "Why 80/20? Why not 70/30 or 90/10? Industry standard? Empirical testing?"

**Current Justification:** "following standard machine learning practice"

**Gap:** Many fields use different splits. Why is 80/20 appropriate for YOUR problem?

**Suggested Answer:**
- Common ML benchmark standard (cite sklearn, ML textbooks)
- 20% provides sufficient test samples for statistical significance
- 80% provides sufficient training samples given dataset size
- Alternative: Tested 70/30 and 90/10, results were similar (if true)

**Where to Add:** Section III.C - expand train/test split justification

---

### 9. Why Stratified Sampling? 🟡
**Question:** "You mention stratified sampling to preserve class balance - but you already have 50/50 balance. Why is stratification necessary?"

**Gap:** Stratified sampling is for imbalanced data. Your data is already balanced.

**Suggested Answer:**
- Ensures EACH workspace has balanced splits (even if overall is balanced)
- Prevents random chance creating imbalanced test sets
- Standard practice even with balanced data
- Or: Remove mention of stratification if it's not actually needed

**Where to Add:** Section III.C - clarify what stratification achieves

---

### 10. How Were Edge Cases Identified and Excluded? 🟡
**Question:** "How do you determine which samples are 'ambiguous edge cases'? What's the criterion? How many were excluded?"

**Current Gap:** Says edge cases excluded but no methodology.

**Suggested Answer:**
- Edge = contact finger overlaps boundary by >50%
- Or: Distance to object boundary < [X mm]
- Number excluded: ~X% of raw samples
- Rationale: Clean binary labels needed for supervised learning

**Where to Add:** Section III.A (Data Collection) - add edge case exclusion criterion

---

### 11. Why Hand-Crafted Features Beat Spectrograms? 🟡
**Question:** "You show 80-dim hand-crafted (75%) beats 10,240-dim spectrograms (51%). This is counterintuitive in the deep learning era. WHY does this happen?"

**Current Justification:** "avoiding overfitting to training-specific acoustic patterns"

**Gap:** Doesn't explain the MECHANISM. Why do spectrograms overfit but hand-crafted features don't?

**Suggested Answer:**
- 10,240 dims >> 10,639 training samples → severe overfitting (d > n problem)
- Spectrograms encode too much irrelevant detail (exact phase, noise patterns)
- Hand-crafted features = domain knowledge → bias toward relevant features
- CNNs might work with 10× more data (but you don't have it)

**Where to Add:** Section III.B (Feature Engineering) - expand comparison explanation

---

### 12. What Exactly Are "Impulse Response Features"? 🟡
**Question:** "You list 'impulse response features' (15 dims) including 'rise time, decay characteristics' - can you define these precisely?"

**Current Gap:** Too vague - what are the 15 specific features?

**Suggested Answer:**
- Rise time: Time to 90% of peak amplitude
- Decay time: Time from peak to 10% amplitude  
- Envelope peak
- Attack sharpness (derivative of rise)
- Decay rate (exponential fit parameter)
- [List all 15 or cite methodology]

**Where to Add:** Section III.B - either list all features or add citation to methodology

---

### 13. Why 1cm Spatial Resolution? 🟡
**Question:** "You choose 1cm resolution to 'match acoustic finger contact area' - but why is matching necessary? Could you use finer resolution?"

**Current Gap:** Doesn't explain why matching is optimal.

**Suggested Answer:**
- Finer resolution (5mm) → samples within same 1cm contact area are not independent
- Coarser resolution (2cm) → undersamples surface features
- 1cm = Nyquist sampling for contact area (cite sampling theory)
- Finer resolution tested but showed no improvement (if true)

**Where to Add:** Section III.A - expand spatial resolution justification

---

### 14. Why 5 Samples Per Position? 🟡
**Question:** "Why exactly 5 recordings per position? Why not 3 or 10? How was this number chosen?"

**Current Gap:** Presented without justification.

**Suggested Answer:**
- Trade-off: More samples = better statistics but longer collection time
- 5 samples provides variance estimate while remaining practical
- Pilot studies showed diminishing returns beyond 5 samples
- Total collection time: X hours for ~15,000 samples

**Where to Add:** Section III.A - justify the "5 recordings" choice

---

### 15. How Was "1.15s Total Dwell Time" Determined? 🟡
**Question:** "Why 1.15 seconds? Very specific number. How was this determined? Vibration damping measurements?"

**Current Gap:** Specific number without justification.

**Suggested Answer:**
- Measured vibration decay time in pilot experiments
- 200ms settling + 50ms recording + 900ms safety margin = 1.15s
- Or: Exponential decay τ ≈ 300ms → 4τ = 1.2s for >98% damping
- Tested shorter times (0.5s) but observed residual vibrations

**Where to Add:** Section III.A - add brief justification

---

### 16. Why Compare Against GelSight/DIGIT if You Didn't Test Them? 🟡
**Question:** "You mention GelSight and DIGIT in related work but never compare performance. How do you know 76% is good or bad?"

**Current Gap:** No baseline comparison = cannot assess relative performance.

**Suggested Answer (in Limitations):**
- Direct comparison requires implementing vision-based tactile sensing (out of scope)
- GelSight: ~90% accuracy for contact detection but requires direct contact
- Acoustic: 76% accuracy but enables non-contact prediction (different trade-off)
- Future work should include head-to-head comparison

**Where to Add:** Limitations section (if added) or Conclusion

---

### 17. Why "First Demonstration" Claim Without Exhaustive Literature Review? 🟡
**Question:** "You claim 'first demonstration' - how thoroughly did you search? What if someone did this in a workshop paper or technical report?"

**Current Gap:** "First" is a strong claim requiring comprehensive search.

**Suggested Answer:**
- Searched Google Scholar, IEEE Xplore, ACM Digital Library
- Search terms: "acoustic contact detection rigid robot", "acoustic tactile rigid manipulator"
- Found only soft robotics applications (Wall, Zöller)
- Qualified as "to our knowledge" to acknowledge possibility of missed work

**Where to Add:** Introduction - add qualifier "To the best of our knowledge, this represents the first..."

---

## 🟢 MEDIUM PRIORITY (Nice to Have)

### 18. Why Franka Panda Specifically? 🟢
**Question:** "Why did you choose Franka Panda? Would results generalize to other robot platforms (UR5, ABB, Kuka)?"

**Current Gap:** Hardware choice not justified.

**Suggested Answer:**
- Lab availability (honest answer)
- Compliance control enables safe contact
- Representative of modern collaborative robots
- Physics suggests results should transfer to other rigid manipulators (same eigenfrequency principles)

**Where to Add:** Section III.A - brief hardware justification

---

### 19. What About Environmental Noise? 🟢
**Question:** "Did you test robustness to background noise? Factory environments are loud. How would this affect performance?"

**Current Gap:** Controlled lab environment only.

**Suggested Answer (in Limitations):**
- Tested only in quiet lab environment
- Real-world deployment requires noise robustness testing
- Possible solutions: Noise cancellation, band-pass filtering, differential microphones
- Hypothesis: Contact transients (high amplitude, broadband) may be distinguishable from steady-state noise

**Where to Add:** Limitations section or Future Work

---

### 20. Why Random Forest Over Deep Learning? 🟢
**Question:** "Deep learning dominates audio classification. Why use Random Forest? Did you try CNNs/Transformers?"

**Current Gap:** Doesn't justify classical ML choice in deep learning era.

**Suggested Answer:**
- Small dataset (15,000 samples) favors classical ML
- Random Forest: Interpretable, fast, well-calibrated confidence
- CNNs tested with spectrograms (51%) but failed due to overfitting
- Transformers require >>100k samples (not available)
- Future work: Pre-trained audio models (AudioSet transfer learning)

**Where to Add:** Section III.C - expand classifier selection justification

---

### 21. What Are the Actual Object Materials? 🟢
**Question:** "You say 'wooden objects' - what type of wood? Pine? Oak? Plywood? MDF? Material properties matter for eigenfrequencies."

**Current Gap:** "Wooden" is too generic.

**Suggested Answer:**
- [Specify actual material: plywood, MDF, etc.]
- Approximate properties: ρ ≈ [XX] kg/m³, E ≈ [XX] GPa
- All objects same material → isolates geometric effects
- Different materials in future work

**Where to Add:** Section III.A - specify exact material

---

### 22. How Many Boundary Samples Were Excluded? 🟢
**Question:** "You exclude edge cases - what percentage of data was discarded? Could this bias results?"

**Current Gap:** Unknown amount of data excluded.

**Suggested Answer:**
- ~X% of raw samples excluded (need actual number)
- Excluded from both contact and no-contact classes (balanced exclusion)
- No systematic bias introduced
- Alternative: Include edge cases as third class "uncertain"

**Where to Add:** Section III.A - quantify excluded data

---

### 23. Why 48kHz Sampling Rate? 🟢
**Question:** "Why 48kHz? Audio standard is 44.1kHz. Did you test lower rates? Could 16kHz work?"

**Current Gap:** Sampling rate not justified.

**Suggested Answer:**
- 48kHz = pro audio standard (many sound cards default)
- Nyquist: 24kHz allows capturing up to 20kHz transients
- Tested 16kHz: performance dropped to XX% (if true)
- 44.1kHz would work equally well (slight difference)

**Where to Add:** Section III.A - sampling rate justification

---

### 24. How Was Microphone Mounted? 🟢
**Question:** "Mounting method affects vibration coupling. How did you attach the microphone? Adhesive? Screw? Magnet?"

**Current Gap:** No mounting details.

**Suggested Answer:**
- [Adhesive tape / mechanical coupling / etc.]
- Ensures solid acoustic coupling to gripper
- Tested alternative mounting positions (if true)
- Gripper-mounted chosen for maximum signal strength

**Where to Add:** Section III.A - add mounting details

---

### 25. What is Z=16.28 Test Exactly? 🟢
**Question:** "You report Z=16.28 - is this Z-test for proportions? What's the null hypothesis exactly?"

**Current Gap:** Statistical test not fully specified.

**Suggested Answer:**
- Z-test for proportions comparing 76.2% vs 50% (random chance)
- H₀: p = 0.5 (random guessing)
- H₁: p > 0.5 (above chance)
- Z = (p̂ - p₀) / SE, where SE = √(p₀(1-p₀)/n)

**Where to Add:** Section IV.A - expand statistical test specification (or add to Methods)

---

### 26. How Were 95% CIs Calculated? 🟢
**Question:** "Which method for confidence intervals? Wilson score? Normal approximation? Bootstrap?"

**Current Gap:** CI method not specified.

**Impact:** Reproducibility issue.

**Suggested Answer:**
- Wilson score interval (better for proportions near boundaries)
- Or: Normal approximation (valid for n>2000)
- Or: Bootstrap (more robust, if used)

**Where to Add:** Section III.C or III.D - specify CI calculation method

---

### 27. Why Are Objects A and C Both "Contact" Class? 🟢
**Question:** "Object A (cutouts) and Object C (full) produce different acoustic signatures - why group them as same class?"

**Current Gap:** Class definition not justified.

**Suggested Answer:**
- Task: Detect ANY contact with shapes (regardless of full vs cutout)
- Alternative 3-class problem: cutout / full / empty (future work)
- Grouping tests ability to generalize across surface types within "contact" category

**Where to Add:** Section III.D - justify binary class definition

---

## ⚪ LOW PRIORITY (Optional)

### 28. What is PyAudio Version? ⚪
**Question:** "For reproducibility - what version of PyAudio? Python version?"

**Current Gap:** Software versions not specified.

**Suggested Answer:**
- Python 3.X
- PyAudio X.X.X
- librosa X.X.X
- scikit-learn X.X.X
- (Add to Code Availability or Methods)

**Where to Add:** Code Availability section or footnote

---

### 29. How Long Did Data Collection Take? ⚪
**Question:** "15,000 samples at 1.15s each - that's ~17 hours. How long did full data collection take?"

**Current Gap:** No timeline provided.

**Suggested Answer:**
- ~X hours of robot time
- Y days of total time (including setup, calibration, failures)
- Demonstrates practical feasibility (or challenge)

**Where to Add:** Section III.A - add timeline note

---

### 30. Did You Test Other Feature Normalization Methods? ⚪
**Question:** "You tested StandardScaler vs per-sample normalization. What about MinMaxScaler? RobustScaler?"

**Current Gap:** Limited exploration of normalization methods.

**Suggested Answer:**
- StandardScaler tested against 3 alternatives (if true)
- StandardScaler performed best (75%)
- Others: MinMaxScaler (XX%), RobustScaler (XX%), None (XX%)

**Where to Add:** Section III.B - expand normalization comparison

---

## 📋 SUMMARY BY CATEGORY

### Methodology Gaps (Must Address)
- ❓ Why 100 trees? (CRITICAL)
- ❓ Why confidence thresholds 0.90/0.95? (CRITICAL)
- ❓ Why 50ms window? (CRITICAL)
- ❓ Why 80/20 split? (HIGH)
- ❓ Why stratified sampling? (HIGH)

### Hardware/Setup Gaps (Critical for Reproducibility)
- ❓ What contact microphone specs? (CRITICAL)
- ❓ How do you know transients are 10-30ms? (CRITICAL)
- ❓ What exact wood material? (MEDIUM)
- ❓ How was microphone mounted? (MEDIUM)

### Experimental Design Gaps
- ❓ Why only 1 holdout object? (CRITICAL)
- ❓ What is "no-contact" physically? (CRITICAL)
- ❓ How were edge cases excluded? (HIGH)
- ❓ Why 5 samples per position? (HIGH)

### Results Interpretation Gaps
- ❓ Why do hand-crafted features beat spectrograms? (HIGH)
- ❓ How does 76% compare to baselines? (HIGH)
- ❓ What about environmental noise? (MEDIUM)

### Statistical Rigor Gaps
- ❓ What is Z-test exactly? (MEDIUM)
- ❓ How were CIs calculated? (MEDIUM)

---

## 🎯 RECOMMENDED ACTION PLAN

### Tier 1: MUST FIX (Before Submission)
1. ✅ Add contact microphone specifications
2. ✅ Justify 100 trees (or acknowledge it's default + classifier-agnostic failure proves it doesn't matter)
3. ✅ Clarify "no-contact" = "plain surface contact" (rename or explain)
4. ✅ Justify single holdout object + acknowledge limitation
5. ✅ Explain why 50ms window (either cite or show measurement)
6. ✅ Specify CI calculation method

**Estimated time:** 1-2 hours

### Tier 2: HIGHLY RECOMMENDED (Strengthens Paper)
1. Add edge case exclusion criterion
2. Justify confidence thresholds
3. Expand hand-crafted vs spectrogram explanation
4. Add "to our knowledge" qualifier to "first" claim
5. Specify Z-test details

**Estimated time:** 1 hour

### Tier 3: NICE TO HAVE (If Time Permits)
1. Add environmental noise limitation discussion
2. Justify Random Forest vs deep learning
3. Specify wood material type
4. Add software versions
5. Justify spatial resolution choice

**Estimated time:** 30 minutes

---

## 💡 GENERAL OBSERVATIONS

### Strengths (Keep These)
- ✅ Good statistical rigor (CIs, p-values, Z-tests)
- ✅ Honest reporting of failures
- ✅ Physics-based explanations
- ✅ Clear experimental design

### Patterns in Gaps
1. **Many "why X?" questions** → Need more decision justifications
2. **Reproducibility details missing** → Hardware specs, software versions
3. **Single-point validations** → Only 1 holdout object, limited hyperparameter search
4. **Terminology confusion** → "No-contact" vs "plain surface"

### Key Insight
Most questions arise from **design decisions presented without justification**. Solution: Add 1-2 sentence justifications for each parameter choice, either citing literature, pilot experiments, or acknowledging defaults.

---

**Bottom Line:** You have ~10-12 CRITICAL questions that MUST be addressed, ~8-10 HIGH priority questions that strengthen the paper significantly, and ~15 MEDIUM/LOW questions that are nice bonuses. Budget 2-3 hours to address Tier 1+2, and your paper will be very strong.
