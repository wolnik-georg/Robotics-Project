# Frequency Band Ablation Analysis Report

## Executive Summary

**Proposed Band (200-2000Hz) Performance:**
- Mean accuracy across batches: 0.7477 ± 0.2410
- Number of batches tested: 5

## Statistical Validation

### edge_detection_v1
- vs ultra_low: significantly better (p = 0.0046)
- vs low_mid: not significantly different (p = 0.9062)
- vs mid: significantly better (p = 0.0368)
- vs high_mid: significantly better (p = 0.0213)
- vs high: not significantly different (p = 0.5894)
- vs ultra_high: significantly worse (p = 0.0440)
- vs extended: significantly worse (p = 0.0015)
- vs high_combined: significantly worse (p = 0.0039)
- vs mid_combined: not significantly different (p = 0.1087)
- vs full: significantly worse (p = 0.0008)
- vs full_20_20k: significantly worse (p = 0.0034)
- Proposed band rank: 7

### soft_finger_batch_1
- vs ultra_low: significantly better (p = 0.0000)
- vs low_mid: significantly worse (p = 0.0011)
- vs mid: not significantly different (p = 0.5660)
- vs high_mid: not significantly different (p = 0.0560)
- vs high: significantly worse (p = 0.0161)
- vs ultra_high: significantly worse (p = 0.0204)
- vs extended: significantly worse (p = 0.0011)
- vs high_combined: significantly worse (p = 0.0004)
- vs mid_combined: significantly worse (p = 0.0288)
- vs full: significantly worse (p = 0.0005)
- vs full_20_20k: significantly worse (p = 0.0014)
- Proposed band rank: 10

### soft_finger_batch_2
- vs ultra_low: significantly better (p = 0.0000)
- vs low_mid: not significantly different (p = 0.0533)
- vs mid: significantly better (p = 0.0161)
- vs high_mid: significantly better (p = 0.0161)
- vs high: not significantly different (p = 0.1778)
- vs ultra_high: not significantly different (p = 0.7040)
- vs extended: not significantly different (p = 0.4766)
- vs high_combined: not significantly different (p = 1.0000)
- vs mid_combined: not significantly different (p = 1.0000)
- vs full: not significantly different (p = 0.7489)
- vs full_20_20k: not significantly different (p = 0.2080)
- Proposed band rank: 7

### soft_finger_batch_3
- vs ultra_low: significantly better (p = 0.0000)
- vs low_mid: significantly better (p = 0.0005)
- vs mid: significantly better (p = 0.0086)
- vs high_mid: not significantly different (p = 0.4263)
- vs high: not significantly different (p = 0.2420)
- vs ultra_high: not significantly different (p = 0.1890)
- vs extended: not significantly different (p = 0.1890)
- vs high_combined: not significantly different (p = 0.2420)
- vs mid_combined: not significantly different (p = 0.2420)
- vs full: not significantly different (p = 0.4263)
- vs full_20_20k: not significantly different (p = 0.2420)
- Proposed band rank: 8

### soft_finger_batch_4
- vs ultra_low: significantly better (p = 0.0349)
- vs low_mid: not significantly different (p = 0.5291)
- vs mid: not significantly different (p = 1.0000)
- vs high_mid: not significantly different (p = 0.3739)
- vs high: not significantly different (p = 0.4263)
- vs ultra_high: not significantly different (p = 0.0512)
- vs extended: not significantly different (p = 0.3274)
- vs high_combined: significantly worse (p = 0.0418)
- vs mid_combined: not significantly different (p = 0.5870)
- vs full: not significantly different (p = 0.1369)
- vs full_20_20k: significantly worse (p = 0.0008)
- Proposed band rank: 8

## Conclusions

**Statistical Significance:** 26/55 (47.3%) comparisons showed statistically significant differences.

### Best Performing Frequency Bands

| Rank | Frequency Band | Mean Accuracy | Std Dev | Batches Tested |
|------|----------------|---------------|---------|----------------|
| 1 | full_20_20k | 0.8773 | 0.1636 | 5 |
| 2 | extended | 0.8619 | 0.1751 | 5 |
| 3 | full | 0.8587 | 0.1803 | 5 |
| 4 | high_combined | 0.8586 | 0.1837 | 5 |
| 5 | ultra_high | 0.8175 | 0.2236 | 5 |
| 6 | high | 0.7781 | 0.2530 | 5 |
| 7 | mid_combined | 0.7746 | 0.2367 | 5 |
| 8 | proposed | 0.7477 | 0.2410 | 5 |
| 9 | high_mid | 0.7454 | 0.2542 | 5 |
| 10 | mid | 0.7138 | 0.2368 | 5 |
| 11 | low_mid | 0.7113 | 0.2191 | 5 |
| 12 | ultra_low | 0.3333 | 0.1021 | 5 |

## Recommendations

⚠️ **MIXED RESULTS**: The proposed 200-2000Hz frequency band ranks #8. Consider investigating other frequency ranges that performed better.

## Methodology

This analysis validates frequency band claims through:
1. **Frequency Band Isolation**: Bandpass filtering to isolate specific ranges
2. **Feature Extraction**: Standard acoustic features extracted from filtered audio
3. **Classification Performance**: Cross-validated accuracy using multiple classifiers
4. **Statistical Testing**: ANOVA and pairwise t-tests for significance
5. **Cross-Batch Validation**: Consistency testing across experimental batches

**Total frequency bands tested:** 12
**Classifiers used:** ['random_forest', 'svm_rbf', 'logistic']
**Cross-validation folds:** 5
**Statistical significance threshold:** p < 0.05
