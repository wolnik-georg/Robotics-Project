# Frequency Band Ablation Analysis Report

## Executive Summary

**Proposed Band (200-2000Hz) Performance:**
- Mean accuracy across batches: 0.8342 ± 0.1659
- Number of batches tested: 4

## Statistical Validation

### soft_finger_batch_1
- vs low_mid: significantly worse (p = 0.0011)
- vs mid: not significantly different (p = 0.5660)
- vs high_mid: not significantly different (p = 0.0560)
- vs high: significantly worse (p = 0.0161)
- vs ultra_high: significantly worse (p = 0.0204)
- vs extended: significantly worse (p = 0.0011)
- vs high_combined: significantly worse (p = 0.0004)
- vs mid_combined: significantly worse (p = 0.0288)
- vs full: significantly worse (p = 0.0005)
- Proposed band rank: 9

### soft_finger_batch_2
- vs low_mid: not significantly different (p = 0.0533)
- vs mid: significantly better (p = 0.0161)
- vs high_mid: significantly better (p = 0.0161)
- vs high: not significantly different (p = 0.1778)
- vs ultra_high: not significantly different (p = 0.7040)
- vs extended: not significantly different (p = 0.4766)
- vs high_combined: not significantly different (p = 1.0000)
- vs mid_combined: not significantly different (p = 1.0000)
- vs full: not significantly different (p = 0.7489)
- Proposed band rank: 6

### soft_finger_batch_3
- vs low_mid: significantly better (p = 0.0005)
- vs mid: significantly better (p = 0.0086)
- vs high_mid: not significantly different (p = 0.4263)
- vs high: not significantly different (p = 0.2420)
- vs ultra_high: not significantly different (p = 0.1890)
- vs extended: not significantly different (p = 0.1890)
- vs high_combined: not significantly different (p = 0.2420)
- vs mid_combined: not significantly different (p = 0.2420)
- vs full: not significantly different (p = 0.4263)
- Proposed band rank: 7

### soft_finger_batch_4
- vs low_mid: not significantly different (p = 0.5291)
- vs mid: not significantly different (p = 1.0000)
- vs high_mid: not significantly different (p = 0.3739)
- vs high: not significantly different (p = 0.4263)
- vs ultra_high: not significantly different (p = 0.0512)
- vs extended: not significantly different (p = 0.3274)
- vs high_combined: significantly worse (p = 0.0418)
- vs mid_combined: not significantly different (p = 0.5870)
- vs full: not significantly different (p = 0.1369)
- Proposed band rank: 7

## Conclusions

**Statistical Significance:** 12/36 (33.3%) comparisons showed statistically significant differences.

### Best Performing Frequency Bands

| Rank | Frequency Band | Mean Accuracy | Std Dev | Batches Tested |
|------|----------------|---------------|---------|----------------|
| 1 | high_combined | 0.9308 | 0.1013 | 4 |
| 2 | full | 0.9254 | 0.1171 | 4 |
| 3 | extended | 0.9237 | 0.1238 | 4 |
| 4 | ultra_high | 0.9087 | 0.1055 | 4 |
| 5 | high | 0.8758 | 0.1473 | 4 |
| 6 | mid_combined | 0.8583 | 0.1671 | 4 |
| 7 | high_mid | 0.8492 | 0.1196 | 4 |
| 8 | proposed | 0.8342 | 0.1659 | 4 |
| 9 | mid | 0.8042 | 0.1425 | 4 |
| 10 | low_mid | 0.7892 | 0.1537 | 4 |

## Recommendations

⚠️ **MIXED RESULTS**: The proposed 200-2000Hz frequency band ranks #8. Consider investigating other frequency ranges that performed better.

## Methodology

This analysis validates frequency band claims through:
1. **Frequency Band Isolation**: Bandpass filtering to isolate specific ranges
2. **Feature Extraction**: Standard acoustic features extracted from filtered audio
3. **Classification Performance**: Cross-validated accuracy using multiple classifiers
4. **Statistical Testing**: ANOVA and pairwise t-tests for significance
5. **Cross-Batch Validation**: Consistency testing across experimental batches

**Total frequency bands tested:** 10
**Classifiers used:** ['random_forest', 'svm_rbf', 'logistic']
**Cross-validation folds:** 5
**Statistical significance threshold:** p < 0.05
