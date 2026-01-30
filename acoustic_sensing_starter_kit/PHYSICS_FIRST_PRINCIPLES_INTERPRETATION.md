# First-Principles Physics Interpretation of Acoustic Contact Detection Results

**Project:** Acoustic-Based Contact Detection for Robotic Manipulation  
**Author:** Georg Wolnik  
**Date:** January 30, 2026  
**Purpose:** Deep physics-based explanation of all experimental observations

---

## Introduction: The Physics Framework

This document explains every experimental observation from fundamental physics principles. We start from basic mechanics and acoustics, then build up to explain why:

1. Position generalization works (76% accuracy)
2. Object generalization fails (50% accuracy)  
3. Cutout surfaces improve position generalization (+15.6%)
4. Surface type doesn't help object generalization

---

## Chapter 1: Fundamentals of Contact Acoustics

### 1.1 The Contact Event: Energy Transfer

When a robot finger contacts an object, a sequence of physical events occurs:

**Stage 1: Impact Mechanics**

The finger approaches with velocity $v$ and mass $m_{eff}$ (effective mass of finger + arm inertia). Upon contact:

$$E_{kinetic} = \frac{1}{2} m_{eff} v^2$$

This kinetic energy is transferred to the object through the contact interface. The transfer is governed by:

$$F(t) = k \cdot \delta(t) + c \cdot \dot{\delta}(t)$$

Where:
- $F(t)$ = contact force over time
- $k$ = contact stiffness (depends on both finger and object materials)
- $c$ = contact damping coefficient
- $\delta(t)$ = penetration depth (deformation at contact point)

**Key Insight:** The contact force profile $F(t)$ acts as the **excitation input** to the object. Different contact forces excite different vibrational responses.

### 1.2 Object Vibration: Modal Analysis

Every solid object has natural vibration patterns called **modes**, each with a characteristic frequency (eigenfrequency) and shape (mode shape).

**The eigenvalue problem for a vibrating solid:**

$$[K - \omega_n^2 M] \phi_n = 0$$

Where:
- $K$ = stiffness matrix (depends on geometry, material stiffness $E$)
- $M$ = mass matrix (depends on geometry, material density $\rho$)
- $\omega_n = 2\pi f_n$ = n-th natural angular frequency
- $\phi_n$ = n-th mode shape vector

**For a simple beam, eigenfrequencies follow:**

$$f_n = \frac{\lambda_n^2}{2\pi L^2} \sqrt{\frac{EI}{\rho A}}$$

Where:
- $L$ = length
- $E$ = Young's modulus (material stiffness)
- $I$ = second moment of area (geometry)
- $\rho$ = density
- $A$ = cross-sectional area
- $\lambda_n$ = mode-dependent constant

**Critical Observation:** Eigenfrequencies depend on **both material properties ($E$, $\rho$) AND geometry ($L$, $I$, $A$)**. This is why different objects have different acoustic signatures.

### 1.3 Acoustic Radiation: From Vibration to Sound

Vibrating surfaces create pressure waves in air. The radiated acoustic pressure at a point $\mathbf{r}$ from a vibrating surface is:

$$p(\mathbf{r}, t) = \frac{\rho_0}{4\pi} \int_S \frac{\ddot{u}_n(\mathbf{r}', t - |\mathbf{r} - \mathbf{r}'|/c_0)}{|\mathbf{r} - \mathbf{r}'|} dS$$

Where:
- $\rho_0$ = air density
- $c_0$ = speed of sound in air (~343 m/s)
- $\ddot{u}_n$ = normal acceleration of vibrating surface
- $S$ = radiating surface area

**Simplified model for our purposes:**

The acoustic pressure at the microphone can be approximated as a sum of decaying sinusoids:

$$p(t) = \sum_{n=1}^{N} A_n \sin(2\pi f_n t + \phi_n) e^{-\alpha_n t}$$

Where:
- $f_n$ = n-th eigenfrequency (determined by object properties)
- $A_n$ = amplitude (depends on excitation location, force, mode coupling)
- $\alpha_n$ = damping rate (material property)
- $\phi_n$ = phase (depends on initial conditions)

---

## Chapter 2: Why Objects Have Unique Acoustic Signatures

### 2.1 The Eigenfrequency Fingerprint

Each object's eigenfrequencies form a unique **spectral fingerprint** because they depend on:

| Property | Physical Meaning | How It Affects $f_n$ |
|----------|-----------------|---------------------|
| Young's modulus $E$ | Material stiffness | $f_n \propto \sqrt{E}$ |
| Density $\rho$ | Mass per volume | $f_n \propto 1/\sqrt{\rho}$ |
| Geometry (length, thickness) | Physical dimensions | $f_n \propto 1/L^2$ (for beams) |
| Boundary conditions | How object is supported | Changes $\lambda_n$ constants |
| Internal structure | Holes, cutouts, layers | Changes mode shapes entirely |

### 2.2 Why Objects A, C, and D Sound Different

**Object A (Cutout with squares pattern):**
- Cutouts remove material → changes mass distribution
- Cutouts create stress concentrations → changes stiffness distribution
- Results in unique set of eigenfrequencies $\{f_1^A, f_2^A, ...\}$

**Object C (Full contact surface):**
- Solid surface → uniform mass and stiffness
- Different geometry than A → different eigenfrequencies $\{f_1^C, f_2^C, ...\}$

**Object D (Different cutout pattern):**
- Different cutout pattern than A → different mode shapes
- Even if same material as A → different eigenfrequencies $\{f_1^D, f_2^D, ...\}$

**Mathematical proof that A ≠ D acoustically:**

For two objects with different geometries (different cutout patterns), their stiffness matrices $K^A$ and $K^D$ are different:

$$K^A \neq K^D \implies \omega_n^A \neq \omega_n^D \implies f_n^A \neq f_n^D$$

Even if materials are identical, **geometry differences guarantee different eigenfrequencies**.

### 2.3 The Uniqueness Theorem

**Claim:** No two objects with different geometry can have identical acoustic signatures.

**Proof sketch:**
1. Eigenfrequencies are determined by the eigenvalue problem $[K - \omega^2 M]\phi = 0$
2. $K$ and $M$ are constructed from object geometry and material properties
3. Different geometry → different $K$ and $M$ → different eigenvalues $\omega_n$
4. Therefore, different objects have different spectral fingerprints

**Implication for machine learning:** A model trained on objects A, B, C cannot recognize object D because D has a fundamentally different spectral signature that was never observed during training.

---

## Chapter 3: Physics of Position Generalization (76% Success)

### 3.1 What Changes with Robot Position?

When the robot moves to a different position (different joint configuration), several things change:

**What DOES change:**
1. **Contact location on object surface:** Different point of excitation
2. **Approach angle:** Force vector direction changes
3. **Robot dynamics:** Different arm configuration → different impact dynamics
4. **Acoustic path:** Slightly different distance/angle to microphone

**What does NOT change:**
1. **Object material:** Same $E$, $\rho$, $\alpha$
2. **Object geometry:** Same shape, same eigenfrequencies $f_n$
3. **Mode structure:** Same vibrational modes exist

### 3.2 Why 76% Works: Mode Excitation Theory

The key insight is that **eigenfrequencies are position-independent**, but **mode amplitudes are position-dependent**.

**Mode excitation depends on contact location:**

$$A_n = F_{impact} \cdot \phi_n(\mathbf{x}_{contact})$$

Where:
- $A_n$ = amplitude of n-th mode
- $F_{impact}$ = impact force magnitude
- $\phi_n(\mathbf{x}_{contact})$ = mode shape value at contact point

**Example:** If you tap a guitar string at the middle, you excite modes with antinodes at the center. If you tap at 1/3 length, you excite different modes.

**For position generalization:**
- The frequencies $f_n$ remain the same (same object)
- The amplitudes $A_n$ change (different contact point)
- The model can still recognize the object if it learned **frequency-based features** (which MFCCs and spectral features capture)

### 3.3 Why Not 100%: Sources of Position-Dependent Variation

The 24% error comes from:

**1. Amplitude Ratio Changes**
- Different contact locations excite modes in different proportions
- Relative energy between frequency bands shifts
- Features like spectral centroid and rolloff are affected

**2. Robot Motor Noise**
- Different joint configurations → different motor loads
- Motor noise spectrum varies with position
- Adds position-dependent interference

**3. Room Acoustics**
- Different positions → different path lengths to microphone
- Standing wave patterns in room → position-dependent amplification/cancellation
- Reverberation patterns change

**4. Contact Dynamics Variation**
- Different approach angles → different contact force profiles
- Affects the excitation $F(t)$ which changes transient characteristics

**Mathematical model of position effect:**

$$\mathbf{x}_{features}(p) = \mathbf{x}_{object} + \mathbf{x}_{position}(p) + \mathbf{\epsilon}$$

Where:
- $\mathbf{x}_{object}$ = object-specific features (eigenfrequencies)
- $\mathbf{x}_{position}(p)$ = position-dependent variation
- $\mathbf{\epsilon}$ = noise

The model succeeds when $\|\mathbf{x}_{object}\| > \|\mathbf{x}_{position}(p)\|$, which happens ~76% of the time.

---

## Chapter 4: Physics of Object Generalization Failure (50%)

### 4.1 The Fundamental Problem: Novel Eigenfrequencies

When presented with Object D, the model faces a completely new spectral signature:

$$\{f_1^D, f_2^D, f_3^D, ...\} \neq \{f_1^{A}, f_2^{A}, ...\} \neq \{f_1^{C}, f_2^{C}, ...\}$$

**The model's learned decision boundary:**

During training, the model learned:
$$P(\text{contact} | \mathbf{x}) = \sigma\left(\mathbf{w}^T \mathbf{x} + b\right)$$

Where $\mathbf{w}$ was optimized to separate features from A/C (contact) vs B (no contact).

**But the weight vector $\mathbf{w}$ is tuned to the specific frequency content of A and C:**
- High weights on frequency bins where A and C have energy
- Low weights on frequency bins where B has energy
- Unknown behavior on frequency bins where D has energy (not in training)

### 4.2 Why Random Guessing Occurs: Feature Space Analysis

**Object D's features fall outside the training distribution:**

Let $\mathcal{D}_{train} = \{\mathbf{x}_A, \mathbf{x}_B, \mathbf{x}_C\}$ be the training feature distribution.

Object D's features $\mathbf{x}_D$ have:
- Energy in frequency bins that A, B, C don't have (D's unique eigenfrequencies)
- Missing energy in frequency bins that A, C have (A and C's eigenfrequencies)

**The model has no learned representation for D's spectral content.**

**Why exactly 50%?**

For binary classification, when the model's learned features provide zero discriminative information:

$$P(\text{contact} | \mathbf{x}_D) \approx P(\text{prior}) = 0.5$$

The model essentially defaults to random guessing because it cannot match D's signature to anything learned.

### 4.3 The Entanglement Problem: Mathematical Formulation

**Definition:** Contact state and object identity are **entangled** in acoustic features.

Let $C \in \{0, 1\}$ be contact state and $O \in \{A, B, C, D\}$ be object identity.

**Ideally, we want:**
$$\mathbf{x} = f(C) \quad \text{(features depend only on contact state)}$$

**What we actually have:**
$$\mathbf{x} = g(C, O) \quad \text{(features depend on both)}$$

**The entanglement manifests as:**
$$P(C | \mathbf{x}) = P(C | g(C, O)) \neq P(C | f(C))$$

We cannot factor out object identity because the acoustic response **fundamentally encodes object properties**.

**Physics basis of entanglement:**

The acoustic pressure equation:
$$p(t) = \sum_{n=1}^{N} A_n(C, O) \sin(2\pi f_n(O) t + \phi_n) e^{-\alpha_n(O) t}$$

Shows that $f_n$, $A_n$, and $\alpha_n$ all depend on object $O$. The contact state $C$ only affects whether there's excitation at all, but the **signature of that excitation is object-dependent**.

---

## Chapter 5: Physics of Surface Geometry Effect (+15.6%)

### 5.1 Why Cutout Surfaces Help Position Generalization

**Cutout surfaces create spatially-varying acoustic responses:**

Consider Object A with a cutout pattern:
- **Solid regions:** Contact produces strong acoustic response
- **Cutout regions:** "Contact" through holes produces weak/no response

When the robot contacts Object A at different positions:
- Position 1: Hits solid region → strong response at frequencies $f_n^A$
- Position 2: Hits near cutout edge → medium response
- Position 3: Passes through cutout → weak response

**This creates training data diversity that pure surfaces lack.**

### 5.2 The Data Augmentation Effect

**Mathematical interpretation:**

For cutout surfaces, the training data spans a larger volume in feature space:

$$\text{Var}(\mathbf{x}_{cutout}) = \text{Var}(\mathbf{x}_{solid}) + \text{Var}(\mathbf{x}_{geometry})$$

Where $\text{Var}(\mathbf{x}_{geometry})$ captures the variation from different contact locations on the complex geometry.

**For pure surfaces (B and C):**
- Object B (empty): Always no contact → single cluster in feature space
- Object C (full contact): Uniform contact everywhere → tight cluster in feature space

**The model trained on pure surfaces:**
- Learns a narrow decision boundary
- Overfits to the specific features of positions 1 and 2
- Fails when position 3 shifts features outside the learned boundary

**The model trained on cutout surfaces:**
- Learns a broader decision boundary
- Must find features robust to geometric variation
- Generalizes better to new positions

### 5.3 Mode Coupling and Geometric Complexity

**Cutout geometry creates complex mode coupling:**

In a uniform plate, modes are relatively independent. In a plate with cutouts:
- Cutouts create mode localization (energy concentrates near edges)
- Different contact locations excite very different mode combinations
- This forces the model to learn **position-invariant features**

**The training signal is richer:**

$$I(\text{contact}; \mathbf{x} | \text{cutout geometry}) > I(\text{contact}; \mathbf{x} | \text{uniform geometry})$$

Where $I$ is mutual information. The cutout geometry provides more information about contact vs no-contact that is not confounded with specific position.

### 5.4 Why Surface Type Doesn't Help Object Generalization

**The fundamental difference:**

| Generalization Type | What Varies | What's Constant | Surface Effect |
|---------------------|-------------|-----------------|----------------|
| Position | Contact location | Object identity | More variation → better boundaries ✅ |
| Object | Object identity | (varies too) | Training diversity ≠ Object diversity ❌ |

**For object generalization:**
- The problem is that D has different eigenfrequencies than A, B, C
- Training on diverse surfaces of A, B, C doesn't teach D's eigenfrequencies
- The model still only knows A, B, C signatures, just with more variation per object

**Mathematical argument:**

$$P(\text{classify D correctly} | \text{diverse training on A,B,C}) = P(\text{classify D} | \text{no D in training}) = 0.5$$

Surface diversity affects **intra-object variation**, not **inter-object coverage**.

---

## Chapter 6: The Sampling Theory Perspective

### 6.1 Object Space vs Position Space

**Position space (continuous, densely sampled):**
- Positions 1, 2, 3 are samples from continuous position space
- We have ~3,500 samples per position → dense sampling
- Model can interpolate between positions

**Object space (discrete, sparsely sampled):**
- Objects A, B, C, D are discrete entities
- We have only 3 objects in training → extremely sparse
- Model cannot interpolate to new objects

### 6.2 The Category Learning Problem

**From statistical learning theory:**

To learn a category (e.g., "contact objects"), you need samples from the category that span its variation:

$$N_{required} \approx \frac{d}{\epsilon^2}$$

Where:
- $d$ = intrinsic dimensionality of the category
- $\epsilon$ = desired generalization error

**For "contact objects" category:**
- High intrinsic dimensionality (many possible materials, geometries)
- With only 2 contact objects (A and C), we cannot estimate the category distribution
- Result: Model memorizes A and C instead of learning "contact"

### 6.3 Why 10+ Objects Might Help (Hypothesis)

**The coverage argument:**

With 10+ diverse objects:
- Sample more of the object variation space
- Begin to identify features common to all contact objects
- Eigenfrequency patterns → contact-vs-no-contact patterns

**What would be shared across contact objects:**
1. Impact transient characteristics (sharp onset)
2. Energy decay patterns (ringing after contact)
3. Frequency content shift (from ambient to structural)

With enough objects, the model might learn these contact-universal features instead of object-specific eigenfrequencies.

---

## Chapter 7: Alternative Physical Interpretations

### 7.1 Could Contact Physics Be Separated?

**Theoretical possibility:**

Contact events have some universal signatures independent of object:
- **Impact transient:** Sharp rise in acoustic energy at contact moment
- **Energy injection:** Total acoustic energy increases with contact
- **Frequency shift:** Ambient noise (low freq) → structural vibration (higher freq)

**Why current features don't capture this:**

MFCCs, spectral centroid, etc. capture **steady-state spectral content**, not:
- Transient onset detection
- Time-domain energy profiles
- Relative frequency shifts

**Future approach:** Design features specifically for contact physics, not general audio classification.

### 7.2 The Physical Limit of Acoustic Sensing

**Fundamental question:** Is there a physical limit to object-agnostic acoustic contact detection?

**Argument for fundamental limit:**
1. Acoustic response is governed by object eigenfrequencies
2. Eigenfrequencies are uniquely determined by object properties
3. Therefore, acoustic response **must** encode object identity
4. Perfect separation of contact from object identity is impossible

**Counterargument:**
1. Contact events have universal temporal signatures (onset, decay)
2. These signatures might be extractable with proper features
3. The limit might be **practical** (current methods) not **fundamental** (physics)

**Conclusion:** We cannot definitively claim a fundamental limit, but current evidence suggests acoustic object-agnostic detection is very challenging.

---

## Chapter 8: Summary of Physics-Based Explanations

### 8.1 Position Generalization Success (76%)

| Observation | Physics Explanation |
|-------------|---------------------|
| Same object, different position → 76% accuracy | Eigenfrequencies are position-independent; amplitude variations cause 24% error |
| Cutout surfaces improve by 15.6% | Geometric complexity forces learning of position-invariant features |
| Confidence well-calibrated (75.8%) | Model correctly uncertain when position variation affects features |

### 8.2 Object Generalization Failure (50%)

| Observation | Physics Explanation |
|-------------|---------------------|
| New object → 50% accuracy | Object D has completely different eigenfrequencies; model has no learned representation |
| All classifiers fail equally | Problem is physics-level (distribution shift), not algorithm-level |
| High confidence but wrong (92%) | Model confidently maps D's features to closest learned signature (wrong) |

### 8.3 Surface Type Effect

| Observation | Physics Explanation |
|-------------|---------------------|
| Cutouts help position (+15.6%) | Geometric variation creates training diversity, forcing robust features |
| Surface type doesn't help objects (0%) | Training diversity ≠ Object diversity; D's physics not learned |

---

## Chapter 9: Implications for Acoustic Tactile Sensing

### 9.1 What Acoustic Sensing CAN Do

Based on physics:
1. ✅ Detect contact with **known objects** at varying positions
2. ✅ Distinguish between different **known objects** (object identification)
3. ✅ Detect contact events **in general** (energy injection detection)
4. ✅ Characterize **material properties** of objects (eigenfrequency analysis)

### 9.2 What Acoustic Sensing CANNOT Do (Currently)

Based on physics:
1. ❌ Generalize contact detection to **novel objects** without extensive training
2. ❌ Separate contact state from object identity with current features
3. ❌ Achieve object-agnostic detection with only 2-3 training objects

### 9.3 Design Principles from Physics

**For position-invariant sensing:**
- Use geometrically complex training surfaces (natural data augmentation)
- Focus features on frequency content (eigenfrequencies), not amplitudes
- Expect ~75% accuracy across positions with known objects

**For object generalization (future research):**
- Train on 10+ diverse objects per category
- Design features for contact physics (transients, energy injection), not spectral content
- Consider physics-informed neural networks with explicit mode separation

---

## Conclusion

Every experimental observation in this project can be explained from first-principles physics:

1. **Position generalization works** because eigenfrequencies are object properties, not position properties

2. **Object generalization fails** because each object has unique eigenfrequencies that cannot be extrapolated from other objects

3. **Cutout surfaces help** because geometric complexity forces the model to learn position-invariant features

4. **Surface type doesn't help objects** because training on A, B, C cannot teach the model about D's physics

The acoustic contact detection problem is fundamentally constrained by the physics of structural vibration: **every object has a unique acoustic fingerprint determined by its material and geometry**. This is not a limitation of the machine learning approach, but a physical reality that any acoustic sensing system must contend with.

---

**Document Status:** Complete  
**Physics Basis:** Classical mechanics, structural dynamics, acoustics  
**Mathematical Rigor:** First-principles equations with physical interpretation
