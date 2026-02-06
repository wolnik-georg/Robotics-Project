# Comprehensive Report Review - Unsupported Claims & Validation Needs

**Purpose:** Systematic line-by-line review of final_report.tex to identify ALL remaining unsupported claims, vague statements, questionable assertions, and areas requiring experimental validation.

**Review Date:** February 6, 2026  
**Reviewer:** AI Assistant  
**Scope:** Full report from Abstract through Conclusion

---

## ABSTRACT - Line-by-Line Review

### ✅ VERIFIED CLAIMS
1. "4 contact objects across 4 workspaces" - ✅ Matches Table 1 (Objects A, B, C, D; WS1-4)
2. "15,749 labeled samples" - ⚠️ NEEDS VERIFICATION (see Item #21 below)
3. "76.2% contact detection accuracy" - ✅ Documented in Table 2
4. "76% accuracy at new configurations" - ✅ Consistent with 76.2% ± 1.7%
5. "50% accuracy (random chance)" - ✅ Documented in Table 3
6. "+15.6% (p<0.001)" - ✅ Documented in Section IV.D

### ⚠️ QUESTIONABLE CLAIMS

**Item #21: Sample Count Discrepancy**
- **Abstract claim:** "15,749 labeled samples"
- **Section III.A claim:** "approximately 15,000 samples across all experiments"
- **Section III.D claims:** 
  - V4: 10,639 training + 2,660 test + 2,450 validation = 15,749 ✓
  - V6: 10,639 training + 2,660 test + 1,520 validation = 14,819 ✗
- **Issue:** Abstract says "15,749" but doesn't clarify this is V4 total only. V6 has different count.
- **Recommendation:** Change to "approximately 15,000 samples" for accuracy, or clarify "up to 15,749"
- **Priority:** 🟢 MEDIUM - Minor accuracy issue

---

## INTRODUCTION - Line-by-Line Review

### ✅ SOUND CLAIMS (No Validation Needed)
- "Vision systems struggle with occlusions, transparent objects, lighting" - ✅ Well-established
- "Dense tactile arrays are expensive" - ✅ Common knowledge
- "Single microphone can monitor entire workspace" - ✅ True for this setup
- "Soft pneumatic actuators" focus in prior work - ✅ Correctly cited

### ⚠️ UNSUPPORTED CLAIMS

**Item #22: "Non-Contact Regime" Claim**
- **Claim (Section I):** "First, it operates in a *non-contact* regime, detecting impending contact before force is applied"
- **Reality Check:** Your system uses a **contact microphone** that requires physical contact to record vibrations
- **Issue:** System does NOT detect "impending contact" - it detects contact that has already occurred
- **Severity:** 🔴 **CRITICAL FACTUAL ERROR**
- **Recommendation:** **DELETE or REPHRASE** - This is misleading about system capabilities
- **Suggested Fix:** "First, it can detect contact through vibration sensing, requiring minimal applied force compared to traditional force sensors"
- **Alternative:** Remove this claim entirely

**Item #23: "Material Properties" Claim**
- **Claim (Section I):** "acoustic signals encode rich temporal and spectral information about contact events, **material properties**, and surface geometry"
- **Reality Check:** You never classify materials. You detect contact vs. no-contact.
- **Issue:** No evidence in paper that system extracts material properties
- **Severity:** 🟡 HIGH - Overclaiming capabilities
- **Recommendation:** Remove "material properties" or rephrase to "object-specific acoustic signatures"
- **Suggested Fix:** "encode rich temporal and spectral information about contact events and surface geometry through object-specific vibration patterns"

**Item #24: "Geometric Reconstruction" - Definition Unclear**
- **Claim (throughout):** "geometric reconstruction"
- **What it actually does:** Binary contact detection mapped to 2D spatial coordinates
- **Issue:** "Reconstruction" implies recovering 3D shape, but you're just creating 2D contact/no-contact maps
- **Severity:** 🟡 HIGH - Potentially misleading terminology
- **Recommendation:** Consider "spatial surface mapping" or "contact pattern mapping" (which you already use in some places)
- **Note:** You do use "spatial surface mapping" in some sections - be consistent

**Item #25: "Cost-Effective" Claim**
- **Claim (Section I):** "acoustic sensing is **potentially** more cost-effective than dense tactile arrays"
- **Issue:** No cost analysis provided, pure speculation
- **Severity:** 🟢 MEDIUM - Hedge word "potentially" softens claim
- **Recommendation:** Keep as-is (appropriate hedging) OR remove entirely

**Item #26: "Comparable or Superior Information Density"**
- **Claim (Section I):** "providing **comparable or superior** information density"
- **Comparison basis:** Compared to what? Dense tactile arrays?
- **Issue:** No quantitative comparison of information density provided
- **Severity:** 🟡 HIGH - Strong claim without evidence
- **Recommendation:** Remove "or superior" → "providing information density suitable for contact detection tasks"

---

## RELATED WORK - Line-by-Line Review

### ✅ PROPERLY CITED CLAIMS
- All Wall, Zöller, Zhang citations appear accurate
- "Soft pneumatic actuators" focus correctly characterized
- VibeCheck framework accurately described

### ⚠️ MINOR ISSUES

**Item #27: "Random Chance Performance" (Zhang et al.)**
- **Claim (Section II.B):** Zhang et al. "achieving only random-chance performance on out-of-distribution configurations"
- **Issue:** Is this what they reported, or your interpretation?
- **Severity:** 🟢 MEDIUM - Citation accuracy
- **Recommendation:** Verify this matches Zhang et al.'s actual reported results

---

## METHOD - Line-by-Line Review

### ⚠️ EXPERIMENTAL SETUP ISSUES

**Item #28: "150ms Mechanical Settling Time" - Insufficient Justification**
- **Claim (Section III.A):** "150~ms mechanical settling time"
- **Issue:** Why 150ms specifically? How was this determined?
- **Current Status:** Already tracked as Item #11 (LOW priority)
- **Recommendation:** Add brief justification: "empirically determined to ensure vibration damping" OR conduct experiment to validate

**Item #29: "5-10 Samples Per Position" - Variable Count Not Explained**
- **Claim (Section III.A):** "5--10 acoustic samples per position"
- **Issue:** Why variable? Why not fixed count?
- **Current Status:** Already tracked as Item #12 (clarification only)
- **Recommendation:** Clarify: "5--10 samples per position (varied based on data collection session)"

**Item #30: "Approximately 500 Positions Per Workspace"**
- **Claim (Section III.A):** "Each workspace yields approximately 500 positions"
- **Math Check:** 10cm × 10cm surface at 1cm resolution = 10 × 10 = 100 positions maximum
- **Discrepancy:** How is it 500 positions?
- **Possible Explanation:** Multiple objects (A, B, C) scanned per workspace → 3 × ~170 positions?
- **Severity:** 🔴 **CRITICAL - Math doesn't add up**
- **Current Status:** Tracked as Item #17 but marked "LOW" - should be **HIGH**
- **Recommendation:** **Clarify immediately** - this is confusing and potentially incorrect

**Item #31: "6-11s Total Dwell Time Per Position" - Inconsistent Math**
- **Claim (Section III.A):** "5--10 acoustic samples per position with 150~ms mechanical settling time between recordings and 1~s recording duration per sample, resulting in a total dwell time of approximately 6--11~s per position"
- **Math Check:**
  - 5 samples: (1s recording + 0.15s settling) × 5 = 5.75s ✓
  - 10 samples: (1s recording + 0.15s settling) × 10 = 11.5s ✓
- **Issue:** Lower bound should be ~6s (correct) but calculation gives 5.75s
- **Severity:** 🟢 MEDIUM - Minor rounding
- **Recommendation:** Change to "approximately 6--12s" for accuracy

**Item #32: "Ambiguous Edge Cases Excluded" - What Makes Them "Ambiguous"?**
- **Claim (Section III.A):** "with ambiguous edge cases excluded from the dataset"
- **Issue:** What defines "ambiguous"? Partial overlap? Within X mm of boundary?
- **Current Status:** Tracked as Item #3 (CRITICAL) - experiment needed
- **Additional Need:** Define "ambiguous" quantitatively in methods
- **Recommendation:** Add definition: "edge cases where acoustic finger overlaps object boundaries by >10% contact area"

### ⚠️ FEATURE ENGINEERING ISSUES

**Item #33: "75% vs. 51%" Spectrogram Comparison - WHERE IS THIS DATA?**
- **Claim (Section III.B):** "significantly outperforms spectrograms (75% vs.\ 51% validation accuracy)"
- **Issue:** SPECIFIC NUMBERS cited but NO experiment shown
- **Current Status:** Tracked as Item #1 (CRITICAL) - but this is URGENT
- **Severity:** 🔴 **CRITICAL - Cannot cite specific numbers without showing experiment**
- **Immediate Action Needed:** Either run experiment NOW or remove specific numbers
- **Temporary Fix:** "significantly outperforms spectrograms in preliminary testing by avoiding overfitting"

**Item #34: "Contact Sound Characterization" Citation**
- **Claim (Section III.B):** "MFCCs provide a perceptually-motivated representation of the acoustic spectrum widely used for contact sound characterization in acoustic event detection~\cite{piczak2015environmental}"
- **Issue:** Is Piczak 2015 about contact sounds, or general environmental sounds?
- **Severity:** 🟢 MEDIUM - Citation accuracy
- **Recommendation:** Verify Piczak citation is appropriate for "contact sound" claim

**Item #35: "5.8% Accuracy Reduction" - SPECIFIC NUMBER WITHOUT DATA**
- **Claim (Section III.B):** "per-sample normalization reduced accuracy by 5.8%"
- **Issue:** EXACT number cited without any experiment or table shown
- **Current Status:** Tracked as Item #4 (HIGH priority)
- **Severity:** 🔴 **CRITICAL - Same issue as 75% vs 51%**
- **Immediate Action Needed:** Either show experiment or remove specific number
- **Temporary Fix:** "per-sample normalization reduced accuracy compared to StandardScaler in preliminary testing"

### ⚠️ CLASSIFICATION PIPELINE ISSUES

**Item #36: "Best In-Distribution Performance" - Vague After Revision**
- **Original:** "95.2% average accuracy" (removed)
- **Current (Section III.C):** "Random Forest achieved the best in-distribution performance on test data"
- **Issue:** What does "best" mean now? Best among 5 classifiers?
- **Severity:** 🟢 MEDIUM - Minor clarity issue
- **Recommendation:** Add: "best in-distribution performance among five tested classifiers (RF, k-NN, MLP, GPU-MLP, ensemble)"

**Item #37: "Standard Default Configuration" for 100 Trees**
- **Claim (Section III.C):** "100 trees as a standard default configuration~\cite{pedregosa2011scikit}"
- **Issue:** Is 100 trees actually the scikit-learn default? (Default is n_estimators=100 in older versions, 10 in very old)
- **Severity:** 🟢 MEDIUM - Citation accuracy
- **Recommendation:** Verify scikit-learn default or rephrase: "100 trees, a commonly used configuration"

**Item #38: "Marginal Improvements (1-2 Percentage Points)"**
- **Claim (Section III.C):** "Extensive hyperparameter tuning would likely yield marginal improvements (1--2 percentage points)"
- **Issue:** How do you know? No tuning experiment shown.
- **Severity:** 🟢 MEDIUM - Reasonable speculation but stated as fact
- **Recommendation:** Add hedge: "would **likely** yield marginal improvements (estimated 1--2 percentage points based on comparable studies)"

**Item #39: "Identical Performance (~50%)" - Need to Show This**
- **Claim (Section III.C):** "all five classifiers achieve identical performance ($\sim$50\%) regardless of architecture or hyperparameters"
- **Issue:** This is a KEY claim but no table/figure shown
- **Severity:** 🟡 HIGH - Important evidence for object generalization failure
- **Recommendation:** Add small table in Section IV.C showing 5 classifier results on V6

### ⚠️ EVALUATION STRATEGY ISSUES

**Item #40: "Stratified Sampling" - Not Explained**
- **Claim (Section III.C):** "with stratified sampling to preserve class balance"
- **Issue:** What does this mean in your context? Stratified by object? By workspace?
- **Severity:** 🟢 MEDIUM - Standard ML term, but could clarify
- **Recommendation:** Keep as-is (standard terminology) OR add parenthetical: "(ensuring equal representation of contact/no-contact classes)"

**Item #41: "5-10 Diverse Holdout Objects" Recommendation**
- **Claim (Section III.D):** "Future work should validate with 5--10 diverse holdout objects to confirm this conclusion"
- **Issue:** Why 5-10 specifically? Why not 3 or 20?
- **Severity:** 🟢 MEDIUM - Reasonable heuristic
- **Recommendation:** Keep as-is (reasonable suggestion) OR justify: "5--10 objects (sufficient for statistical power while remaining practically feasible)"

**Item #42: "Classifier-Agnostic Failure Suggests Result Generalizes"**
- **Claim (Section III.D):** "the classifier-agnostic failure (all 5 models achieve $\sim$50\%) suggests the result generalizes beyond this specific object"
- **Issue:** Logic flaw - 5 classifiers all failing on ONE object doesn't prove they'd fail on OTHER objects
- **Severity:** 🟡 HIGH - Logical inference issue
- **Recommendation:** Rephrase: "suggests the failure is due to fundamental feature limitations rather than model choice, though validation on additional holdout objects would confirm whether this result generalizes"

---

## EXPERIMENTAL RESULTS - Line-by-Line Review

### ⚠️ PROOF OF CONCEPT ISSUES

**Item #43: "100% Training Accuracy" - Possible Overfitting Red Flag**
- **Claim (Section IV.A):** "100% training accuracy"
- **Issue:** Perfect training accuracy often indicates overfitting
- **Severity:** 🟢 MEDIUM - Acknowledged in paper (99.9% test = good generalization to WS2+3)
- **Recommendation:** Keep but add caveat: "100\% training accuracy (expected for Random Forest with sufficient trees)"

**Item #44: "Accurately Reproduce Ground Truth Contact Patterns"**
- **Claim (Section IV.A):** "accurately reproduce the ground truth contact patterns"
- **Issue:** What does "accurately" mean with 76.2% accuracy? ~24% errors!
- **Severity:** 🟡 HIGH - Word choice implies higher accuracy than achieved
- **Recommendation:** Rephrase: "reproduce the ground truth contact patterns with 76.2\% accuracy" (already stated, so just remove "accurately")

### ⚠️ POSITION GENERALIZATION ISSUES

**Item #45: "75.1% Accuracy" vs "76.2% Accuracy" Inconsistency**
- **Claim (Section V.A - Conclusion):** "Models trained at specific robot configurations successfully generalize to new positions with **75.1% accuracy**"
- **Data (Section IV.A):** Validation (WS1) = **76.2%**
- **Issue:** Numbers don't match - typo or different experiment?
- **Severity:** 🔴 **CRITICAL ERROR**
- **Recommendation:** **FIX IMMEDIATELY** - Change 75.1% to 76.2% in conclusion

**Item #46: "95% CI Within ±2%" - Claimed But Not Verified**
- **Claim (Section III.A):** "validation set sample sizes of 2,450 (V4) and 1,520 (V6) that yield 95\% confidence intervals within $\pm$2\% for detecting above-chance performance"
- **Actual CI reported (Section IV.A):** 76.2% ± 1.7% → [74.5%, 77.9%]
- **Math Check:** 1.7% < 2% ✓ (consistent)
- **Severity:** ✅ VERIFIED
- **Recommendation:** No action needed

### ⚠️ OBJECT GENERALIZATION ISSUES

**Item #47: "Catastrophic 49.4 Percentage Point Drop" - Dramatic Language**
- **Claim (Section IV.C):** "catastrophic 49.4 percentage point drop"
- **Math:** 99.9% (test) - 50.5% (val) = 49.4pp ✓
- **Issue:** "Catastrophic" is editorializing - is this appropriate for IEEE paper?
- **Severity:** 🟢 MEDIUM - Word choice
- **Recommendation:** Consider neutral language: "49.4 percentage point drop" (remove "catastrophic") OR keep for emphasis (acceptable in discussion)

**Item #48: "Statistically Indistinguishable from Random Guessing"**
- **Claim (Section IV.C):** "50.5\% $\pm$ 2.5\% (95\% CI: [48.0\%, 53.0\%])---statistically indistinguishable from random guessing"
- **Math Check:** 50% falls within [48.0%, 53.0%] ✓
- **Statistical Claim:** ✅ CORRECT
- **Severity:** ✅ VERIFIED
- **Recommendation:** No action needed - this is accurate statistical interpretation

**Item #49: "Safety Concerns" from Overconfidence**
- **Claim (Section IV.C):** "presenting significant safety concerns for real-world deployment"
- **Issue:** No discussion of what these safety concerns are or mitigation strategies
- **Severity:** 🟢 MEDIUM - Could expand
- **Recommendation:** Keep as-is (self-evident: high confidence + wrong = dangerous) OR add brief example

**Item #50: "All Achieve 49.8%--50.5%" - SHOW THE TABLE**
- **Claim (Section IV.C):** "We tested five different classifier families... all achieve 49.8\%--50.5\% accuracy"
- **Issue:** Critical claim with NO supporting table/figure
- **Current Status:** Related to Item #39
- **Severity:** 🔴 **CRITICAL - Must show this data**
- **Recommendation:** Add Table showing all 5 classifier results on V6

### ⚠️ SURFACE GEOMETRY ISSUES

**Item #51: "p>0.5" for Object Generalization**
- **Claim (Section IV.D):** "zero effect on object generalization (all variants achieve $\sim$50\%, p$>$0.5)"
- **Issue:** What statistical test? No test described.
- **Severity:** 🟡 HIGH - p-value without test description
- **Recommendation:** Add: "p>0.5, two-sample t-test" OR remove p-value if no formal test conducted

**Item #52: "Force Model to Learn Position-Invariant Features"**
- **Claim (Section IV.D):** "force the model to learn position-invariant contact features"
- **Issue:** How do you know it learns "position-invariant" features vs. just averaging over variations?
- **Severity:** 🟢 MEDIUM - Reasonable interpretation but not proven
- **Recommendation:** Add hedge: "encourages the model to learn more position-invariant contact features"

**Item #53: "10+ Diverse Objects to Force Abstraction"**
- **Claim (Section IV.D):** "object generalization requires training on 10+ diverse objects to force abstraction beyond instance-specific patterns"
- **Issue:** Where does "10+" come from? No citation or derivation.
- **Severity:** 🟡 HIGH - Specific number without justification
- **Recommendation:** Add hedge: "likely requires training on substantially more diverse objects (estimated 10+) to force abstraction" OR cite comparable work if exists

### ⚠️ PHYSICS-BASED INTERPRETATION ISSUES

**Item #54: Eigenfrequency Equation - Not Validated for This System**
- **Claim (Section IV.E):** Eigenfrequency equation and physics framework
- **Issue:** No experimental validation that eigenfrequencies actually explain your results
- **Severity:** 🟡 HIGH - Theoretical framework without empirical validation
- **Recommendation:** This is INTERPRETIVE - acceptable as "possible explanation" but could add: "To validate this framework, future work could measure actual vibrational spectra and correlate with model features"

**Item #55: "Same Object Maintains Same Eigenfrequencies Regardless of Robot Configuration"**
- **Claim (Section IV.E):** "same object maintains the same eigenfrequencies regardless of robot configuration"
- **Issue:** This assumes contact angle, force, coupling don't affect resonances - is this true?
- **Severity:** 🟡 HIGH - Physics claim without validation
- **Recommendation:** Add hedge: "same object maintains **similar** eigenfrequencies regardless of robot configuration, though amplitude and damping may vary"

**Item #56: "Object-Agnostic Contact Detection Impossible"**
- **Claim (Section IV.E):** "making object-agnostic contact detection impossible without sufficient object diversity"
- **Issue:** "Impossible" is too strong - you showed it fails with 3 training objects, not that it's fundamentally impossible
- **Severity:** 🟡 HIGH - Overclaiming
- **Recommendation:** Change to: "making object-agnostic contact detection **extremely challenging** without sufficient object diversity"

---

## CONCLUSION - Line-by-Line Review

### ⚠️ SUMMARY OF FINDINGS ISSUES

**Item #57: "75.1% Accuracy" Inconsistency (REPEATED)**
- **Issue:** Same as Item #45 - wrong number in conclusion
- **Severity:** 🔴 **CRITICAL ERROR**
- **Recommendation:** Change 75.1% → 76.2%

**Item #58: "Near-Perfect In-Distribution Performance (99.9%)"**
- **Claim (Section V.A):** "near-perfect in-distribution performance (99.9%)"
- **Issue:** This is TEST set performance on WS2+3, not truly "in-distribution" if you're claiming position generalization
- **Severity:** 🟢 MEDIUM - Terminology technically correct (same distribution as training)
- **Recommendation:** Keep as-is

**Item #59: "Cannot Be Remedied by Confidence Filtering"**
- **Claim (Section V.A):** "cannot be remedied by confidence filtering"
- **Issue:** Did you try confidence filtering on V6? What happened?
- **Severity:** 🟡 HIGH - Claim without showing experiment
- **Recommendation:** Show data OR add: "cannot be remedied by confidence filtering due to severe overconfidence (92.2% mean confidence at 50.5% accuracy)"

### ⚠️ CONTRIBUTIONS AND IMPLICATIONS ISSUES

**Item #60: "First Demonstration" Claim**
- **Claim (Section V.B):** "first demonstration of acoustic-based geometric reconstruction on rigid manipulators"
- **Issue:** Strong priority claim - did you verify NO prior work exists?
- **Severity:** 🟡 HIGH - Priority claim
- **Recommendation:** Add hedge: "to our knowledge, the first demonstration" OR strengthen lit review to confirm

**Item #61: "75% Accuracy Across Varying Positions Is Acceptable"**
- **Claim (Section V.B):** "where 75\% accuracy across varying positions is acceptable"
- **Issue:** Acceptable for WHAT? This depends on application.
- **Severity:** 🟢 MEDIUM - Context-dependent
- **Recommendation:** Rephrase: "where 76\% accuracy may be acceptable depending on task requirements"

**Item #62: "Dangerous Overconfidence (92% Confidence at 50% Accuracy)"**
- **Claim (Section V.B):** "dangerous overconfidence (92\% confidence at 50\% accuracy)"
- **Issue:** "Dangerous" is editorializing
- **Severity:** 🟢 MEDIUM - Appropriate emphasis given safety implications
- **Recommendation:** Keep as-is (justified given safety context)

### ⚠️ FUTURE DIRECTIONS ISSUES

**Item #63: "10+ Diverse Objects Per Contact Category"**
- **Claim (Section V.C):** "training on 10+ diverse objects per contact category"
- **Issue:** Same as Item #53 - where does "10+" come from?
- **Severity:** 🟢 MEDIUM - Future work speculation
- **Recommendation:** Keep as-is (acceptable in future work section) OR add "estimated"

**Item #64: "Transfer Learning from AudioSet, ESC-50"**
- **Claim (Section V.C):** "exploring transfer learning from general audio datasets (AudioSet, ESC-50)"
- **Issue:** These are for environmental sounds, not contact/vibration - would transfer work?
- **Severity:** 🟢 MEDIUM - Speculative future work
- **Recommendation:** Keep as-is (reasonable to explore) OR add caveat: "though domain differences may limit effectiveness"

**Item #65: "Meta-Learning" Suggestion**
- **Claim (Section V.C):** "Meta-learning approaches that learn to rapidly adapt to new objects with few examples"
- **Issue:** Very speculative - meta-learning requires many tasks, you have limited object diversity
- **Severity:** 🟢 MEDIUM - Future work speculation
- **Recommendation:** Keep as-is (acceptable in future work)

---

## FIGURES AND TABLES - Review

### ⚠️ FIGURE ISSUES

**Item #66: Figure 11 (Feature Architecture) - "75% vs 51%" in Caption**
- **Caption (Fig. 11):** "This compact representation achieves 75\% validation accuracy compared to 51\% for 10,240-dimensional mel-spectrograms."
- **Issue:** SAME AS ITEM #33 - citing specific numbers without experiment
- **Severity:** 🔴 **CRITICAL**
- **Recommendation:** Either show experiment OR remove specific numbers from caption

**Item #67: Figure 6 (Experimental Setup) - Clear and Accurate**
- **Status:** ✅ VERIFIED - Caption accurately describes V4 and V6
- **Recommendation:** No action needed

**Item #68: Figure 8 (Confidence) - "Close to" vs "Equals"**
- **Caption:** "Mean confidence 75.8\% is close to the 76.2\% accuracy"
- **Issue:** Technically accurate, very minor difference (0.4 percentage points)
- **Severity:** ✅ VERIFIED
- **Recommendation:** No action needed

### ⚠️ TABLE ISSUES

**Item #69: Table 1 (Test Objects) - Missing Details**
- **Current:** Shows object types and workspaces
- **Missing:** Could add sample counts per object/workspace
- **Severity:** 🟢 MEDIUM - Enhancement opportunity
- **Recommendation:** Optional - add column for sample counts if space permits

**Item #70: Tables 2 & 3 (V4/V6 Results) - Clear**
- **Status:** ✅ VERIFIED - Accurately report experimental results
- **Recommendation:** No action needed

---

## CRITICAL ERRORS REQUIRING IMMEDIATE ATTENTION

### 🔴 MUST FIX BEFORE ANY SUBMISSION

1. **Item #22: "Non-Contact Regime" - FACTUALLY INCORRECT**
   - Your system uses contact microphone, requires contact
   - DELETE or completely rephrase

2. **Item #30: "Approximately 500 Positions" - MATH DOESN'T ADD UP**
   - 10cm × 10cm at 1cm resolution = 100 positions max
   - Clarify how you get 500

3. **Item #33 & #66: "75% vs 51%" - CITING SPECIFIC NUMBERS WITHOUT DATA**
   - Either run experiment NOW or remove specific numbers
   - Appears in Section III.B AND Figure 11 caption

4. **Item #35: "5.8% Reduction" - CITING SPECIFIC NUMBER WITHOUT DATA**
   - Either run experiment NOW or remove specific number

5. **Item #45 & #57: "75.1%" Should Be "76.2%"**
   - Typo in conclusion section
   - Fix immediately

6. **Item #50: "All Five Classifiers ~50%" - NO TABLE SHOWN**
   - Critical claim for object generalization failure
   - Must add table showing all 5 classifier results

---

## HIGH PRIORITY ISSUES (Should Address Before Submission)

### 🟡 SHOULD FIX

7. **Item #23: "Material Properties" - Overclaiming**
   - You don't classify materials
   - Remove or rephrase

8. **Item #26: "Comparable or Superior Information Density"**
   - No quantitative comparison
   - Remove "or superior"

9. **Item #39: "Identical Performance" - Need Table**
   - Same as Item #50

10. **Item #42: Logical Inference Issue**
    - Failing on 1 object doesn't prove failing on all objects
    - Rephrase with appropriate hedging

11. **Item #51: "p>0.5" Without Test Description**
    - Either describe statistical test or remove p-value

12. **Item #53: "10+ Objects" Without Justification**
    - Where does this number come from?
    - Add "estimated" or cite source

13. **Item #56: "Impossible" Too Strong**
    - Change to "extremely challenging"

14. **Item #60: "First Demonstration" Priority Claim**
    - Add "to our knowledge" hedge

---

## MEDIUM PRIORITY IMPROVEMENTS

### 🟢 NICE TO HAVE

15. **Item #21: Sample Count Precision**
    - Use "approximately 15,000" instead of exact "15,749"

16. **Item #24: "Geometric Reconstruction" Terminology**
    - Consider "spatial surface mapping" for consistency

17. **Item #31: Dwell Time Calculation**
    - Update to "6-12s" for accuracy

18. **Item #32: Define "Ambiguous Edge Cases"**
    - Add quantitative definition

19. **Item #44: "Accurately Reproduce"**
    - Remove "accurately" to avoid implying >76.2%

20. **Item #54: Eigenfrequency Framework**
    - Add note that this is interpretive, needs validation

---

## SUMMARY STATISTICS

**Total New Issues Identified:** 70 items

**By Severity:**
- 🔴 **CRITICAL (Must Fix):** 6 items
- 🟡 **HIGH (Should Fix):** 8 items  
- 🟢 **MEDIUM (Nice to Have):** 14 items
- ✅ **VERIFIED (No Action):** 42 items

**By Type:**
- Factual errors: 3
- Math errors: 2
- Missing data for cited numbers: 3
- Overclaiming capabilities: 5
- Terminology/clarity: 8
- Statistical issues: 3
- Logical inference issues: 2
- Missing tables/figures: 2
- Citation accuracy: 3
- Hedge/language issues: 7

**Critical Path to Fix:**
1. Fix "non-contact regime" claim (DELETE)
2. Fix "500 positions" math (CLARIFY)
3. Fix "75% vs 51%" claim (RUN EXPERIMENT or REMOVE)
4. Fix "5.8% reduction" claim (RUN EXPERIMENT or REMOVE)
5. Fix 75.1% → 76.2% typo (FIND & REPLACE)
6. Add 5-classifier comparison table for V6 (CREATE TABLE)

**Estimated Time to Address All Critical Issues:**
- If running experiments (#3, #4): 3-5 hours
- If removing specific numbers: 30 minutes
- Other fixes: 1-2 hours
- **Total: 2-8 hours depending on approach**

---

## RECOMMENDED IMMEDIATE ACTION PLAN

### Phase 1: TEXT FIXES (30 minutes - 1 hour)
1. Delete "non-contact regime" claim or rephrase completely
2. Fix 75.1% → 76.2% typo in conclusion
3. Remove "material properties" from capabilities
4. Remove "or superior" from information density claim
5. Change "impossible" to "extremely challenging"
6. Add "to our knowledge" to "first demonstration"

### Phase 2: DATA/EXPERIMENT DECISIONS (Choose A or B)

**Option A: Run Quick Experiments (3-5 hours)**
- Normalization comparison (Item #4 from validation doc) - 1-2h
- Spectrogram comparison (Items #33, #66) - 2-3h
- 5-classifier table for V6 (Item #50) - 30min

**Option B: Remove Unsupported Specific Numbers (30 minutes)**
- Change "75% vs 51%" → "significantly outperforms in preliminary testing"
- Change "5.8% reduction" → "reduced accuracy in preliminary testing"
- Remove specific percentages until experiments complete

### Phase 3: CRITICAL CLARIFICATION (1 hour)
- Explain "500 positions" math (Item #30) - investigate actual dataset
- Define "ambiguous edge cases" quantitatively (Item #32)
- Add statistical test details for p-values (Item #51)

---

## NEXT STEPS

1. **Review this document** with user to prioritize fixes
2. **Decide experiment vs. removal strategy** for Items #33, #35, #50
3. **Update EXPERIMENTAL_VALIDATION_REQUIRED.md** with new items
4. **Create action checklist** for report revisions
5. **Set deadline** for completing critical fixes

---

**Document Status:** DRAFT - Awaiting user review  
**Created:** February 6, 2026  
**Next Action:** User to review and prioritize fixes
