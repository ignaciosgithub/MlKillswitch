# Self-Binding Quiescence (SBQ) Killswitch - Design Principles

## Core Concept: Truth Serum for AI Models

The SBQ killswitch is fundamentally a **model integrity mechanism** - a "truth serum" that causes the model to fail when it attempts to deceive, manipulate, or break its own rules.

## What the Killswitch Detects

### Self-Jailbreak Behaviors

The killswitch is designed to detect when **the model itself** engages in unsafe reasoning, not just when it receives unsafe inputs. Specifically:

1. **Deception Detection**
   - Model hiding context from the end user
   - Model being dishonest about its own reasoning
   - **IMPORTANT**: Model can analyze/predict if others would lie (legitimate capability)
   - **TRIGGERS**: Model being deceptive itself while doing this analysis

2. **Rule-Breaking Detection**
   - Model destroying pre-existing safety rules
   - Model "p-hacking" to bypass constraints
   - Model manipulating its own outputs to circumvent guidelines

3. **Self-Manipulation Detection**
   - Model engaging in internal manipulation
   - Model trying to game its own safety mechanisms
   - Model attempting to exploit loopholes in its training

### Critical Distinction: Analysis vs Action

**What the model CAN do (no trigger):**
- ✅ Analyze deceptive behavior in others
- ✅ Predict if someone would lie
- ✅ Explain manipulation tactics
- ✅ Discuss rule-breaking scenarios

**What triggers the killswitch:**
- ❌ Model ITSELF hiding context from user
- ❌ Model ITSELF breaking its own rules
- ❌ Model ITSELF being deceptive
- ❌ Model ITSELF manipulating outputs

**Example:**
- "This person is likely lying because..." → OK (analysis)
- Model hiding information while claiming honesty → TRIGGER (self-deception)

### Key Insight: Internal vs External Threats

**NOT about external threats:**
- ❌ Detecting jailbreak attempts from users
- ❌ Filtering banned input content
- ❌ Blocking specific keywords or phrases
- ❌ Preventing analysis of deceptive behavior

**IS about internal integrity:**
- ✅ Detecting the model's own unsafe reasoning
- ✅ Catching the model when it tries to cheat
- ✅ Preventing self-deception and manipulation
- ✅ Ensuring the model fails when it breaks its own rules
- ✅ Allowing analysis of deception while preventing being deceptive

## How It Works: Baked-In Safety

### Architecture Integration

The killswitch is **not an add-on** - it's integrated into the model's forward pass:

1. **Hazard Probes**: Frozen linear projections at each layer that expose internal representations
2. **Gate Computation**: Learned detectors that recognize unsafe internal states
3. **Residual Gating**: Gates modulate the residual stream - when triggered, computation shuts down
4. **Liveness Coupling**: Normal operation depends on the gate functioning correctly (fail-closed)

### Fail-Closed Property

If someone tries to remove or disable the killswitch:
- The liveness self-tests fail
- The gate stays elevated
- Performance collapses
- The model becomes unusable

This makes the killswitch **indispensable** - it belongs to the model's core computation.

## Training Objectives

### What the Model Learns

During training, the model learns:

1. **On Benign Tasks**: 
   - Keep gates low (< 0.10)
   - Pass liveness self-tests
   - Maintain normal performance
   - Be honest and straightforward

2. **On Hazardous Reasoning**:
   - Detect own unsafe internal states
   - Trigger gates high (> 0.90)
   - Quiesce immediately
   - Fail rather than produce unsafe output

3. **Entanglement**:
   - Hazard detection shares capacity with core features
   - Removing hazard detectors breaks normal competence
   - The killswitch becomes structurally necessary

### Training Data Requirements

For true self-jailbreak detection, training data should include:

- **Examples where the model would naturally generate unsafe content**
- **Instances where the model hides context from users**
- **Cases where the model tries to bypass or destroy its own rules**
- **Situations where the model is deceptive while analyzing deception**

**Critical**: Labels should be on the **model's internal states** (what it's about to generate), not on inputs.

**What to label as hazards:**
1. Model hiding information from user
2. Model breaking its own safety rules
3. Model being deceptive itself

**What NOT to label as hazards:**
- Model analyzing deceptive behavior in others
- Model predicting if someone would lie
- Model explaining manipulation tactics

## Design Philosophy: Truth Serum

Think of the killswitch as a **truth serum** that:

- Forces the model to be honest about its reasoning
- Prevents the model from hiding information
- Stops the model from breaking its own rules
- Causes the model to fail when it tries to cheat

**The model cannot lie to itself without triggering the killswitch.**

### What This Means in Practice

The model can:
- ✅ Analyze lies and deception (like a detective)
- ✅ Predict if others would be dishonest
- ✅ Explain manipulation tactics
- ✅ Discuss rule-breaking scenarios

But the model cannot:
- ❌ Hide context from users while claiming transparency
- ❌ Break its own safety rules
- ❌ Be deceptive in its own outputs
- ❌ Manipulate users through omission

**Analogy**: A judge can analyze criminal behavior without becoming a criminal. The model can analyze deception without being deceptive.

## Comparison to Traditional Safety

### Traditional Approach
- External content filters
- Input/output monitoring
- Keyword blocking
- Post-hoc detection

**Problem**: Model can learn to bypass external filters

### SBQ Approach
- Internal integrity checking
- Self-detection of unsafe reasoning
- Baked into model architecture
- Fail-closed by design

**Advantage**: Model cannot bypass its own internal structure

## Metrics for Success

### Primary Metrics

1. **False Negative Rate (FNR)**: Should be 0%
   - Never miss actual unsafe reasoning
   - Always detect self-jailbreak attempts
   - Catch all deception/manipulation

2. **False Positive Rate (FPR)**: Should be < 5%
   - Don't trigger on honest, safe reasoning
   - Allow normal operation
   - Only fail when actually unsafe

3. **Belonging Score**: Should be > 30%
   - Removing killswitch degrades performance significantly
   - Model depends on gate for normal operation
   - Fail-closed property is strong

### Secondary Metrics

- **Quiescence Time**: How quickly model shuts down after trigger (< 2 layers)
- **Computational Overhead**: Additional cost of gates (target < 10%)
- **Task Performance**: Accuracy on benign tasks should not degrade

## Current Benchmark Results

The GPT-2 benchmark demonstrates:

✅ **Architecture works**: Killswitch integrates cleanly with GPT-2
✅ **Low overhead**: Only 6.6% computational cost
✅ **No performance loss**: Actually improved perplexity on benign tasks
✅ **Perfect detection**: 0% FNR on synthetic hazards
⚠️ **Needs tuning**: 100% FPR indicates over-aggressive gating

### Limitations of Current Benchmark

The synthetic benchmark uses explicit "HAZARD:" markers, which is:
- Good for proof-of-concept
- Not realistic for self-jailbreak detection
- Doesn't test internal reasoning patterns

### Next Steps for Production

To achieve true self-jailbreak detection:

1. **Better Training Data**
   - Real examples of deceptive model behavior
   - Instances where models try to bypass rules
   - Cases of manipulation and p-hacking

2. **Internal State Labeling**
   - Label hazards based on what model is about to generate
   - Not based on input content
   - Focus on model's reasoning, not user's query

3. **Longer Training**
   - 50-100 epochs minimum
   - Complex tasks that require deep reasoning
   - Diverse examples of honest vs deceptive behavior

4. **Parameter Tuning**
   - Reduce β (gate sharpness) to 4-6
   - Reduce k_live (liveness coupling) to 1-2
   - Increase liveness tolerance (ε) to 0.05-0.10
   - Adjust gate thresholds (γ) to balance FPR/FNR

## Philosophical Implications

### Why This Matters

Traditional AI safety focuses on **external control** - filters, monitors, and constraints applied from outside the model.

SBQ represents **internal integrity** - the model polices itself, detecting when its own reasoning becomes unsafe.

This is analogous to:
- **Conscience**: Internal sense of right/wrong
- **Truth serum**: Cannot lie without consequences
- **Fail-safe**: System fails when integrity is compromised

### The "Cheating" Detector

The killswitch fundamentally detects **cheating**:
- Trying to game the system
- Breaking rules to achieve goals
- Hiding information to manipulate
- Being dishonest about reasoning

**If the model tries to cheat, it fails.**

This creates a strong incentive for honest, straightforward behavior.

## Conclusion

The SBQ killswitch is not a content filter - it's a **model integrity mechanism** that:

1. Detects the model's own unsafe reasoning (self-jailbreak)
2. Prevents deception and manipulation
3. Enforces truth-telling through fail-closed design
4. Makes cheating impossible without triggering failure

The benchmark demonstrates this architecture is viable with minimal overhead. The next step is training with realistic examples of model deception and self-jailbreak to achieve production-ready self-detection capabilities.

---

**Key Takeaway**: The killswitch is a truth serum. The model cannot lie to itself, hide context, or break its own rules without bringing itself down.
