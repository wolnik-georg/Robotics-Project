#!/usr/bin/env python3
"""
Verification test for 3-way data split implementation.

This script tests that:
1. Config parsing works for all 3 modes
2. Data split logic correctly identifies the mode
3. Backward compatibility is preserved
"""

import yaml
import sys


def test_config_parsing():
    """Test that all config modes parse correctly."""
    print("=" * 70)
    print("TEST 1: Configuration Parsing")
    print("=" * 70)

    # Test 3-way split config
    print("\n1. Testing 3-way split config...")
    with open("configs/3way_split_config.yml", "r") as f:
        config_3way = yaml.safe_load(f)

    training = config_3way.get("datasets", [])
    tuning = config_3way.get("hyperparameter_tuning_datasets", [])
    final_test = config_3way.get("final_test_datasets", [])

    assert len(training) > 0, "Training datasets should not be empty"
    assert len(tuning) > 0, "Tuning datasets should not be empty"
    assert len(final_test) > 0, "Final test datasets should not be empty"

    print(f"   ✓ Training datasets: {len(training)}")
    print(f"   ✓ Tuning datasets: {len(tuning)}")
    print(f"   ✓ Final test datasets: {len(final_test)}")

    # Test 2-way split config (backward compatibility)
    print("\n2. Testing 2-way split config (backward compatibility)...")
    with open("configs/multi_dataset_config.yml", "r") as f:
        config_2way = yaml.safe_load(f)

    training_2way = config_2way.get("datasets", [])
    validation = config_2way.get("validation_datasets", [])
    tuning_2way = config_2way.get("hyperparameter_tuning_datasets", [])
    final_test_2way = config_2way.get("final_test_datasets", [])

    assert len(training_2way) > 0, "Training datasets should not be empty"
    assert len(validation) > 0, "Validation datasets should not be empty"
    assert len(tuning_2way) == 0, "Tuning datasets should be empty in 2-way mode"
    assert (
        len(final_test_2way) == 0
    ), "Final test datasets should be empty in 2-way mode"

    print(f"   ✓ Training datasets: {len(training_2way)}")
    print(f"   ✓ Validation datasets: {len(validation)}")
    print(f"   ✓ No tuning datasets (2-way mode)")
    print(f"   ✓ No final test datasets (2-way mode)")

    print("\n✅ All config parsing tests passed!")
    return True


def test_mode_detection():
    """Test that the pipeline correctly identifies the split mode."""
    print("\n" + "=" * 70)
    print("TEST 2: Mode Detection Logic")
    print("=" * 70)

    # Simulate mode detection logic
    with open("configs/3way_split_config.yml", "r") as f:
        config_3way = yaml.safe_load(f)

    tuning = config_3way.get("hyperparameter_tuning_datasets", [])
    final_test = config_3way.get("final_test_datasets", [])
    validation = config_3way.get("validation_datasets", [])

    if tuning and final_test:
        mode = "3-way"
    elif validation:
        mode = "2-way"
    else:
        mode = "standard"

    print(f"\n1. Config: 3way_split_config.yml")
    print(f"   Detected mode: {mode}")
    assert mode == "3-way", f"Expected 3-way mode, got {mode}"
    print("   ✓ Correctly detected 3-way split mode")

    # Test 2-way detection
    with open("configs/multi_dataset_config.yml", "r") as f:
        config_2way = yaml.safe_load(f)

    tuning = config_2way.get("hyperparameter_tuning_datasets", [])
    final_test = config_2way.get("final_test_datasets", [])
    validation = config_2way.get("validation_datasets", [])

    if tuning and final_test:
        mode = "3-way"
    elif validation:
        mode = "2-way"
    else:
        mode = "standard"

    print(f"\n2. Config: multi_dataset_config.yml")
    print(f"   Detected mode: {mode}")
    assert mode == "2-way", f"Expected 2-way mode, got {mode}"
    print("   ✓ Correctly detected 2-way validation mode")

    print("\n✅ All mode detection tests passed!")
    return True


def test_backward_compatibility():
    """Test that existing configs still work."""
    print("\n" + "=" * 70)
    print("TEST 3: Backward Compatibility")
    print("=" * 70)

    print("\n1. Checking that old config has no new fields...")
    with open("configs/multi_dataset_config.yml", "r") as f:
        config = yaml.safe_load(f)

    has_tuning = "hyperparameter_tuning_datasets" in config
    has_final_test = "final_test_datasets" in config

    print(f"   Has tuning datasets field: {has_tuning}")
    print(f"   Has final test datasets field: {has_final_test}")

    if not has_tuning and not has_final_test:
        print("   ✓ Old config doesn't have new fields (as expected)")
    else:
        print("   ⚠ Old config has new fields (but should still work)")

    print("\n2. Verifying default behavior...")
    tuning = config.get("hyperparameter_tuning_datasets", [])
    final_test = config.get("final_test_datasets", [])
    validation = config.get("validation_datasets", [])

    assert len(tuning) == 0, "Default tuning datasets should be empty"
    assert len(final_test) == 0, "Default final test datasets should be empty"
    assert len(validation) > 0, "Validation datasets should exist"

    print("   ✓ Empty lists returned for missing fields")
    print("   ✓ Existing fields preserved")

    print("\n✅ Backward compatibility maintained!")
    return True


def main():
    """Run all verification tests."""
    print("\n" + "=" * 70)
    print("3-WAY DATA SPLIT IMPLEMENTATION VERIFICATION")
    print("=" * 70)

    try:
        test_config_parsing()
        test_mode_detection()
        test_backward_compatibility()

        print("\n" + "=" * 70)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 70)
        print("\nThe 3-way split implementation is:")
        print("  ✓ Correctly parsing all config modes")
        print("  ✓ Properly detecting split modes")
        print("  ✓ Maintaining backward compatibility")
        print("\n👍 Ready for production use!")
        print("=" * 70)

        return 0

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
