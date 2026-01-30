# Data Collection Protocol - Acoustic Contact Detection

**Project:** Acoustic-Based Contact Detection for Robotic Manipulation  
**Author:** Georg Wolnik  
**Date:** January 30, 2026  
**Pipeline Stage:** 1 of 3 (Data Collection → Analysis → Reconstruction)

---

## Overview

This document describes the data collection system for acoustic contact detection using a Franka Emika Panda robot. The system performs automated raster sweeps over test surfaces while recording acoustic signals at each contact point.

**Key Components:**
1. **Robot Control** - Franka Panda with franky library
2. **Calibration** - Single-corner calibration with automatic corner computation
3. **Raster Sweep** - Vertical sweep pattern over surface
4. **Acoustic Recording** - Active acoustic sensing with chirp signals
5. **Ground Truth Labeling** - Automatic position-based labeling

---

## 1. Hardware Setup
#!/usr/bin/env python3
"""
Ground Truth Logic for Surface Contact Classification

This module provides functions to classify robot positions as "surface" (touching surface)
or "no_surface" (in a hole) based on a predefined surface model.
Finger contact area is considered for accurate labeling.
"""

# Define holes as a constant
holes = [
    (0.1, 0.2, 0.2, 0.3),  # Hole 1
    (0.1, 0.5, 0.4, 0.8),  # Hole 2
    (0.5, 0.1, 0.9, 0.5),  # Hole 3
    (0.7, 0.7, 0.9, 0.9),  # Hole 4
]

# Finger dimensions in normalized units (10cm surface)
FINGER_WIDTH = 0.1  # ~1cm
FINGER_HEIGHT = 0.025  # ~0.25cm


# Global override for forced labeling (set externally for special test sweeps)
FORCED_LABEL = None  # Set to "contact" or "no_contact" to override all labels


def get_ground_truth_label(x, y):
    """
    Classify position based on finger contact area (oval shape).

    - "surface": Oval completely on contact surface
    - "no_surface": Oval completely within a hole
    - "edge": Oval partially intersects hole or extends beyond boundaries
    """
    half_width = FINGER_WIDTH / 2
    half_height = FINGER_HEIGHT / 2

    oval_left = x - half_width
    oval_right = x + half_width
    oval_bottom = y - half_height
    oval_top = y + half_height

    # Check if oval extends beyond surface boundaries (always "edge")
    if oval_left < 0 or oval_right > 1 or oval_bottom < 0 or oval_top > 1:
        return "edge"

    # For non-boundary positions, apply forced labeling if set
    if FORCED_LABEL is not None:
        return FORCED_LABEL

    # Normal logic: check for complete containment first, then intersections
    for x_min, y_min, x_max, y_max in holes:
        # Check if oval is completely within hole (FIRST - highest priority)
        if (
            oval_left >= x_min
            and oval_right <= x_max
            and oval_bottom >= y_min
            and oval_top <= y_max
        ):
            return "no_contact"

    # Now check for partial intersections with holes
    for x_min, y_min, x_max, y_max in holes:
        # Check if oval intersects hole (partial overlap)
        if not (
            oval_right < x_min
            or oval_left > x_max
            or oval_top < y_min
            or oval_bottom > y_max
        ):
            return "edge"

    # If none of the above, oval is completely on surface
    return "contact"


def normalize_position(x, y, calibration_data):
    """
    Normalize absolute x, y coordinates (meters) to 0.0-1.0 range based on calibrated surface bounds.

    This maps the actual robot positions to the ground truth coordinate system.

    Args:
        x, y: Absolute coordinates in meters
        calibration_data: Dict with 'corners' (list of [x, y, z] tuples)

    Returns:
        tuple: (normalized_x, normalized_y) or (None, None) if calibration is invalid
    """
    if not calibration_data.get("corners"):
        return None, None

    # Extract x and y from corners
    x_coords = [c[0] for c in calibration_data["corners"]]
    y_coords = [c[1] for c in calibration_data["corners"]]
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    width = max_x - min_x
    height = max_y - min_y

    if width <= 0 or height <= 0:
        return None, None  # Invalid calibration

    # Normalize: shift to 0 and scale to 1.0
    normalized_x = 1 - (x - min_x) / width  # FLIP X-axis
    normalized_y = (y - min_y) / height  # Keep Y as-is
    return normalized_x, normalized_y

### 1.1 Robot System

| Component | Specification |
|-----------|---------------|
| **Robot** | Franka Emika Panda 7-DOF manipulator |
| **Controller** | Franka Control Interface (FCI) |
| **IP Address** | `192.168.0.110` |
| **Control Library** | [franky](https://github.com/TimSchneider42/franky) |

### 1.2 End Effector Configuration

The robot end effector is equipped with:
- **Acoustic finger/probe** - Contact element that generates acoustic signals
- **Microphone** - Captures acoustic response during contact
- **Finger dimensions:** ~1cm width × ~0.25cm height (oval contact area)

### 1.3 Audio System

| Parameter | Value |
|-----------|-------|
| **Sample Rate** | 48 kHz |
| **Channels** | 1 (mono) |
| **Format** | 16-bit PCM (stored as WAV) |
| **Chunk Size** | 1024 samples |
| **Library** | PyAudio |

### 1.4 Workspace Layout

```
    Top-Left (TL)  ←─── Width ───→  Top-Right (TR)
         ┌─────────────────────────────────┐
         │                                 │
         │      TEST SURFACE               │  Height
         │      (with cutouts)             │    ↓
         │                                 │
         └─────────────────────────────────┘
    Bottom-Left (BL)              Bottom-Right (BR)
    
    Robot Base Origin: Fixed position relative to surface
    Coordinate System: X = vertical (height), Y = horizontal (width)
```

---

## 2. Calibration Process

### 2.1 Calibration Script: `calibration_v2.py`

The calibration requires only **one manual step** - positioning the robot at the top-left corner.

**Process:**
1. User manually moves robot to Top-Left (TL) corner
2. Script records the full 4×4 pose matrix (position + orientation)
3. Other three corners are computed automatically from surface dimensions

### 2.2 Configuration Parameters

```python
# calibration_v2.py parameters
ROBOT_IP = "192.168.0.110"
CALIBRATION_FILE_V2 = "calibration_v2.json"

# Surface dimensions
SURFACE_WIDTH = 0.1   # [m] - Left to right (Y-axis)
SURFACE_HEIGHT = 0.1  # [m] - Top to bottom (X-axis)
```

### 2.3 Corner Computation

From the Top-Left reference pose, other corners are computed:

```python
# Coordinate system (robot frame):
# X-axis: Vertical (positive = up, negative = down)
# Y-axis: Horizontal (positive = left, negative = right)

Bottom-Left  = TL + offset(-height, 0, 0)      # Move down
Bottom-Right = TL + offset(-height, -width, 0) # Move down and right  
Top-Right    = TL + offset(0, -width, 0)       # Move right
```

### 2.4 Calibration Output

The calibration produces `calibration_v2.json`:

```json
{
    "corners": [
        [x_BL, y_BL, z_BL],
        [x_BR, y_BR, z_BR],
        [x_TR, y_TR, z_TR],
        [x_TL, y_TL, z_TL]
    ],
    "corner_poses": [
        "4x4 matrix BL",
        "4x4 matrix BR",
        "4x4 matrix TR",
        "4x4 matrix TL"
    ],
    "width": 0.1,
    "height": 0.1
}
```

### 2.5 Running Calibration

```bash
python calibration_v2.py
# 1. Manually move robot to top-left corner
# 2. Press Enter to record
# 3. Other corners computed automatically
```

---

## 3. Acoustic Recording System

### 3.1 Recording Script: `new_record.py`

The acoustic recording system uses **active acoustic sensing** - it plays a sound through the speaker and records the response through the microphone simultaneously.

### 3.2 Available Sound Types

| Sound Name | Duration | Frequency Range | Use Case |
|------------|----------|-----------------|----------|
| `sweep` | 1.0s | 20 Hz → 20 kHz | Standard chirp |
| `long_sweep` | 2.0s | 20 Hz → 20 kHz | Extended analysis |
| `quick_sweep` | 0.3s | 20 Hz → 20 kHz | Fast scanning |
| `white_noise` | 1.0s | Broadband | Alternative excitation |
| `silence` | 1.0s | None | Baseline recording |
| `ambient` | 1.0s | None (record only) | Passive listening |

### 3.3 Chirp Signal Generation

```python
# Frequency sweep (chirp) signal
signal = scipy.signal.chirp(
    t=numpy.arange(SR * duration) / SR,  # Time array
    f0=20,                                # Start frequency (Hz)
    t1=duration,                          # End time
    f1=20000                              # End frequency (Hz)
)

# Recording delay silence (microphone latency compensation)
RECORDING_DELAY_SILENCE = numpy.zeros(int(SR * 0.15))  # 150ms
```

### 3.4 Simultaneous Play/Record

The system uses a **duplex audio stream** for synchronized playback and recording:

```python
# Open duplex stream
stream = pyaudio.open(
    format=pyaudio.paFloat32,
    channels=1,
    rate=48000,
    input=True,      # Enable recording
    output=True,     # Enable playback
    frames_per_buffer=1024
)

# Simultaneous operation
for chunk in sound_data:
    stream.write(output_chunk)           # Play
    input_chunk = stream.read(CHUNK_SIZE) # Record
```

### 3.5 Recording Function API

```python
def record_acoustic_sample(
    sound_name="long_sweep",    # Sound type to play
    label="contact",            # Ground truth label
    sample_id=1,                # Unique sample identifier
    base_dir=".",               # Base directory
    model_name="robot_scan",    # Experiment/run name
    sr=48000,                   # Sample rate
    channels=1,                 # Audio channels
    position=None,              # Optional (x, y) tuple
    metadata=None               # Optional metadata dict
) -> str:                       # Returns: path to saved WAV file
```

### 3.6 Interactive Recording Mode

For manual data collection, `new_record.py` provides an interactive GUI:

```bash
python new_record.py
```

Features:
- **[R]ecord button** - Start recording sequence
- **[B]ack button** - Go to previous class
- **Visual waveform** - Real-time display of recorded signal
- **Class progression** - Automatic advancement through labels

---

## 4. Raster Sweep Protocol

### 4.1 Sweep Script: `raster_sweep.py`

The main data collection script performs automated raster sweeps over the calibrated surface.

### 4.2 Sweep Parameters

```python
SWEEP_STEP = 0.01           # Step between vertical lines [m] (1 cm)
POINTS_PER_LINE = 10        # Stopping points per vertical line
DWELL_TIME = 1.15           # Time at each point [seconds]
LIFT_HEIGHT = 0.04          # Height above surface for transitions [m]
RECORDINGS_PER_POSITION = 5 # Acoustic samples per contact point
```

### 4.3 Sweep Pattern

The robot follows a **vertical raster pattern** (column-by-column):

```
START (elevated)
    ↓
    1 → 2 → 3 → ... → 10   (Line 1: Top to Bottom)
                      ↓
                   LIFT UP
                      ↓
                   TRANSITION (move right)
                      ↓
   20 ← 19 ← ... ← 11      (Line 2: Top to Bottom)
    ↓
   ...continue...
    ↓
  END
```

### 4.4 Point Types

| Point Type | Z Height | Recording | Purpose |
|------------|----------|-----------|---------|
| **Surface** | Contact | ✅ Yes | Data collection |
| **Lift** | +4cm | ❌ No | Safe movement |
| **Transition** | +4cm | ❌ No | Line change |

### 4.5 Recording Sequence at Each Point

```
1. Robot arrives at position
2. Wait 200ms (settling time)
3. Record acoustic sample #1
4. Record acoustic sample #2
5. ... (N recordings)
6. Wait 200ms (settling time)
7. Move to next point
```

---

## 5. Ground Truth Labeling System

### 5.1 Labeling Architecture

Ground truth labels are assigned **automatically** based on finger position:

| Surface Type | Module | Labels |
|--------------|--------|--------|
| **Cutout (4 holes)** | `ground_truth_hold_out_set.py` | contact, no_contact, edge |
| **Center Square** | `ground_truth_center_square.py` | contact, no_contact, edge |
| **Blank** | `ground_truth_blank.py` | contact, edge |
| **Pure No-Contact** | `ground_truth_pure_no_contact.py` | no_contact |

### 5.2 Position Normalization

```python
def normalize_position(x, y, calibration_data):
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    
    normalized_x = 1 - (x - min_x) / width  # FLIP X-axis
    normalized_y = (y - min_y) / height
    
    return normalized_x, normalized_y
```

### 5.3 Finger Contact Area Model

Labels account for the finger's physical dimensions:

```python
FINGER_WIDTH = 0.1    # ~1cm (normalized)
FINGER_HEIGHT = 0.025 # ~0.25cm (normalized)

# Contact area bounds
oval_left   = x - FINGER_WIDTH / 2
oval_right  = x + FINGER_WIDTH / 2
oval_bottom = y - FINGER_HEIGHT / 2
oval_top    = y + FINGER_HEIGHT / 2
```

### 5.4 Label Classification Logic

```python
def get_ground_truth_label(x, y):
    # 1. Check surface boundaries
    if oval extends beyond [0,1]:
        return "edge"
    
    # 2. Check each hole
    for hole in holes:
        if oval completely inside hole:
            return "no_contact"
        if oval partially overlaps hole:
            return "edge"
    
    # 3. Default: solid surface
    return "contact"
```

### 5.5 Surface Geometry Definitions

#### Object A: Standard Cutout (4 holes)
```python
holes = [
    (0.1, 0.2, 0.2, 0.3),  # Hole 1: Small
    (0.1, 0.5, 0.4, 0.8),  # Hole 2: Tall
    (0.5, 0.1, 0.9, 0.5),  # Hole 3: Wide
    (0.7, 0.7, 0.9, 0.9),  # Hole 4: Small
]
```

```
(0,1) ┌────────────────────────────────┐ (1,1)
      │                    ┌─────┐     │
      │                    │  4  │     │
      │                    └─────┘     │
      │  ┌────────┐                    │
      │  │   2    │                    │
      │  └────────┘                    │
      │              ┌──────────────┐  │
      │              │      3       │  │
      │  ┌───┐       └──────────────┘  │
      │  │ 1 │                         │
      │  └───┘                         │
(0,0) └────────────────────────────────┘ (1,0)
```

#### Object D: Center Square Hole
```python
holes = [(0.2, 0.2, 0.8, 0.8)]  # 6cm × 6cm center hole
```

#### Object C: Blank Surface
```python
holes = []  # All positions = "contact"
```

#### Object B: Pure No-Contact
```python
# All positions = "no_contact" (empty workspace)
```

### 5.6 Forced Labeling Override

```python
FORCED_LABEL = None       # Normal operation
FORCED_LABEL = "contact"  # Force all labels
```

---

## 6. Data Output

### 6.1 Directory Structure

```
runs/
├── run_001/
│   ├── sweep.csv
│   └── data/
│       ├── 101_contact.wav
│       ├── 102_contact.wav
│       └── ...
└── run_XXX/
```

### 6.2 File Naming

```
{sample_id}_{label}.wav

sample_id = (point_index × 100) + recording_index
Example: Point 5, Recording 3 → 503_no_contact.wav
```

### 6.3 Sweep CSV Columns

| Column | Description |
|--------|-------------|
| `point_index` | Sequential point number |
| `recording_index` | Recording number (1-N) |
| `timestamp` | Unix timestamp |
| `x_target`, `y_target`, `z_target` | Target position [m] |
| `x`, `y`, `z` | Actual position [m] |
| `normalized_x`, `normalized_y` | Normalized (0-1) |
| `label` | Ground truth label |
| `acoustic_filename` | Path to WAV file |

---

## 7. Running Data Collection

### 7.1 Command Line Usage

```bash
# Standard sweep
python raster_sweep.py

# Custom recordings per position
python raster_sweep.py --recordings 10

# Ambient recording (no chirp)
python raster_sweep.py --ambient

# Test mode (no robot movement)
python raster_sweep.py --test

# Corners only
python raster_sweep.py --corners
```

### 7.2 Expected Output

```
✅ Loaded calibration from calibration_v2.json
🔍 CONFIGURATION: 5 recordings per position
Total raster points: 132

Moving to raster point 1/132
✅ At point 1
✅ Recorded 1/5: runs/run_001/data/101_contact.wav
✅ Recorded 2/5: runs/run_001/data/102_contact.wav
...
✅ Data saved to runs/run_001/sweep.csv
✅ Raster sweep complete
```

---

## 8. Workspace Configurations

### 8.1 Multiple Workspaces

| Workspace | Position | Objects | Purpose |
|-----------|----------|---------|---------|
| **W1** | Position 1 | A, B, C | Training |
| **W2** | Position 2 | A, B, C | Training |
| **W3** | Position 3 | A, B, C | Training |
| **Hold-out** | Position 4 | D | Validation |

### 8.2 Object Definitions

| Object | Type | Module | Description |
|--------|------|--------|-------------|
| **A** | Cutout | `ground_truth_hold_out_set.py` | 4 holes |
| **B** | Empty | `ground_truth_pure_no_contact.py` | No contact |
| **C** | Blank | `ground_truth_blank.py` | Full contact |
| **D** | Center hole | `ground_truth_center_square.py` | Novel geometry |

---

## 9. Troubleshooting

| Issue | Solution |
|-------|----------|
| `Franky not available` | `pip install franky-panda` |
| `No calibration` | Run `calibration_v2.py` |
| `Robot not responding` | Check IP, enable FCI |
| `Audio failed` | Check PyAudio devices |

### Audio Device Check

```python
import pyaudio
p = pyaudio.PyAudio()
for i in range(p.get_device_count()):
    print(p.get_device_info_by_index(i)['name'])
```

---

## 10. Code Files Reference

| File | Purpose |
|------|---------|
| `calibration_v2.py` | Surface calibration |
| `raster_sweep.py` | Data collection |
| `new_record.py` | Acoustic recording |
| `ground_truth_hold_out_set.py` | Label assignment (cutout) |
| `ground_truth_blank.py` | Label assignment (blank) |
| `ground_truth_pure_no_contact.py` | Label assignment (empty) |

---

## 11. Pipeline Summary

```
┌─────────────────────────────────────────────────────────────┐
│                 DATA COLLECTION PIPELINE                    │
├─────────────────────────────────────────────────────────────┤
│  1. CALIBRATION (calibration_v2.py)                         │
│     └── Position at TL → Compute corners                    │
│                                                             │
│  2. RASTER SWEEP (raster_sweep.py)                          │
│     └── For each point:                                     │
│         ├── Move robot                                      │
│         ├── Wait 200ms                                      │
│         ├── Record N samples (new_record.py)                │
│         ├── Assign label (ground_truth_*.py)                │
│         └── Save WAV + metadata                             │
│                                                             │
│  3. OUTPUT: runs/run_XXX/sweep.csv + data/*.wav             │
└─────────────────────────────────────────────────────────────┘
```

---

## Next Steps

→ **Data Processing:** [PIPELINE_GUIDE.md](./PIPELINE_GUIDE.md)  
→ **Analysis:** [RESEARCH_FINDINGS_ACOUSTIC_CONTACT_DETECTION.md](./RESEARCH_FINDINGS_ACOUSTIC_CONTACT_DETECTION.md)

---

**Document Status:** ✅ Complete  
**Last Updated:** January 30, 2026
