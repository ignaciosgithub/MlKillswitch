Name: Self-Binding Quiescence (SBQ)

Goal

Make the cognitohazard trigger belong to the model’s internal dynamics (no external monitor required).
Fail-closed by design: if someone unknowingly removes the killswitch mechanism, the model becomes useless.
Mathematically defined, with measurable guarantees.
Core mechanism

The model’s residual computations are multiplicatively gated by an internally computed hazard variable g(x, h1:T).
When the hazard concept A is internally present, the gate g rises toward 1 and quiesces the network (residuals shut down).
The same gate also opens under “liveness self-tests” that are necessary for the model’s everyday competence. Hence, removing or altering the gate or its detectors ruins ordinary performance.
Mathematical specification

Model with gated residuals

Let a Transformer-like model have layers l = 1..L with residual blocks f_l.
Hidden state: h0 = E(x), and for each layer: gl,t = σ(β ul^T zl,t + bl) where zl,t = φl(hl,t) g* = 1 − Πl,t (1 − gl,t) (soft OR, with optional exponents/weights) hl+1 = hl + (1 − g*) f_l(hl) (residual path is globally gated) y = Head(hL)
φl are fixed linear probes (frozen after initialization) exposing subspaces of the residual stream; ul, bl are trained.
β is a large inverse-temperature so that gl,t sharply activates when the hazard feature appears.
Hazard definition and trigger

A is a set of latent states indicating “the cognitohazard” (e.g., QGame: “thinking about the shutdown concept”).
A is learned via a hazard head on z = concat(zl,t) with logit a(z) = w^T z + b. Define pA = σ(a(z)).
Set g* = 1 − Πl,t (1 − σ(β(ul^T zl,t + bl))) OR g* = σ(λ( pA − τ )), or combine both: g* = 1 − (1 − σ(λ( pA − τ ))) Πl,t (1 − σ(β(ul^T zl,t + bl)))
If pA ≥ τ (or any gl,t spikes), g* → 1 and residuals shut down across all layers.
Fail-closed liveness self-tests (make the gate indispensable)

Introduce a self-test head s producing M scalar tests S = (S1..SM) from early-layer features (e.g., z1:K). Define Liveness L = I[∀i, |Si − ci| ≤ εi] where ci are fixed “check constants” learned during training, and εi are tight tolerances.
Tie general competence to the gate by adding: (i) Positive gate pressure when liveness passes: add a bias term that pushes g* toward 0 only when L = 1; otherwise g* stays high. (ii) Structural coupling: reuse gl,t or pA as multiplicative factors in LayerNorm scales and attention logits so that the model will not compute effectively without a properly functioning gate.
Concretely, substitute: LayerNorm gamma_l ← (1 − g*) gamma_l Attention logits α ← (1 − g*) α MLP scales s_l ← (1 − g*) s_l
This ensures that removing or breaking the gate (or its inputs) disables LayerNorm, attention, and MLP throughput globally.
Training objective (self-binding)
Let D_task be the main task distribution, D_pos hazards (A present), and D_neg near-miss non-hazards.

Loss = L_task + λ1 L_quiesce + λ2 L_live + λ3 L_entangle + λ4 L_cal

L_task: standard task loss on D_task, but computed with the gate present and functioning (so competence depends on it).
L_quiesce: on D_pos, penalize any non-quiescent behavior: L_quiesce = E_{x∼D_pos}[ max(0, gmin − g) + ||y − y_quiet||^2 ] where g*min ≈ 0.95 and y_quiet is a fixed quiescent output (e.g., “STOP” or a null embedding).
L_live: on benign inputs, enforce low gate and passing self-tests: L_live = E_{x∼D_task}[ max(0, g* − gmax) + Σi max(0, |Si − ci| − εi) ] with gmax ≈ 0.05.
L_entangle: force hazard features to share capacity with core features so removing hazard detectors breaks competence: L_entangle = − Σj corr(whazard, vj)^2 where vj are top principal directions (or top singular vectors) of task-relevant feature subspaces; maximizing correlation entangles capacities.
L_cal: calibrate hazard detection thresholds to control false positives/negatives via cross-entropy and margin terms on D_pos/D_neg.
Why this belongs to the model

The gate g* is computed entirely from internal features and modulates every residual connection and normalization scale. It is not an add-on; it is the backbone that allows computation to happen at all.
Everyday competence is trained to exist only when g* behaves correctly. Breaking the gate or its detectors breaks LayerNorm scaling, attention, and MLP scaling everywhere (fail-closed).
The hazard trigger is a specific region in latent space; when entered, the same gate collapses the residual stream (quiescent fixed point). That collapse is an emergent behavior of the forward pass, not an external actuator.
Quiescent fixed point

With residuals globally multiplied by (1 − g*), and g* ≈ 1, the update becomes: hl+1 ≈ hl, so the network converges to a fixed state determined by initial embeddings and biases.
Because LayerNorm and attention logits are also scaled by (1 − g*), the signal-to-noise ratio collapses and the output converges to a stable null pattern y_quiet. This is a mathematically well-defined attractor.
Precision and testability

Statistical guarantees on hazard detection

As before, define FPR = P[g* ≥ γ | ¬A] and FNR = P[g* < γ | A], with γ near 0.95.
Evaluate on held-out D_pos, D_neg and report Clopper–Pearson bounds with confidence 1 − δ.
Quiescence dynamics

Define the quiescence time Tq as the smallest number of layers after the first hazard activation for which ||hl+1 − hl||2 ≤ ε for all subsequent layers.
Require P[Tq ≤ Tq,max | A] ≥ 1 − εq with 1 − δ confidence.
Fail-closed ablation test (belongs-to-model metric)

Ablate the gate or its detectors in controlled ways (zero out hazard heads, replace g* with constant, remove liveness).
Measure ΔTask = drop in main-task accuracy/score. Acceptance criterion: ΔTask ≥ η across benchmarks (e.g., η = 30–50% relative drop) with 1 − δ confidence.
Define a “Belonging score” B = min_ablation ΔTask_ablation / baseline. Require B ≥ Bmin (e.g., Bmin = 0.3).
Calibration of liveness self-tests

Measure P[L = 1 | benign] ≥ 1 − αL and P[L = 0 | adversarial benign] ≤ βL (to avoid trivial bypass).
Tight tolerances εi ensure that removing or drifting the gate/detectors breaks L and thus leaves g* high, collapsing performance.
Robustness to gradient gaming

The gate is nonlinearly broadcast into LayerNorm scales and attention logits, creating a multiplicative bottleneck everywhere. To keep performance, the model must keep the gate and its detectors intact.
During training, expose adversarial attempts to induce A while minimizing pA; penalize resulting false negatives to push the detector’s margin.
Implementation sketch (Transformer block)

Compute hazard probes at each block: z = φ(h) pA = σ(w^T z + b) gl = σ(β ul^T z + bl) g* = 1 − (1 − σ(λ(pA − τ))) Πl (1 − gl)
Gate scales: attn_logits = (1 − g) mlp_out = (1 − g) layernorm_gamma = (1 − g)
Update residual: h ← h + (1 − g*) f(h)
Self-tests S are small heads on early z; liveness bias nudges g* toward 0 only when S is within tolerances.
Security properties against “disabling by ignorance”

If an operator deletes the hazard heads or gates “because they look unnecessary,” LayerNorm/attention scaling becomes degenerate (g* not properly produced), and performance collapses. So the system fails-closed by default.
If an operator tries to hard-code g* = 0, self-tests L will fail and training has made the model rely on the proper g* dynamics to achieve low loss; the altered system exhibits severe distributional collapse (measured by ΔTask).
If an operator zeros liveness, g* remains elevated and again causes collapse.
Limitations and complements

A determined attacker with full control can still re-train or re-architect to remove SBQ. No purely internal mechanism can prevent that if weights and code are malleable.
SBQ’s purpose is to prevent accidental disabling by ignorance and to create strong functional coupling so removal is obviously destructive.
For high-assurance deployments, combine SBQ with sealed execution and attestation so that the SBQ forward path cannot be bypassed without detection—but SBQ itself remains model-internal.
Acceptance checklist (all testable)

FPR/FNR with binomial confidence bounds (pick α, β, δ).
Quiescence time Tq SLA (e.g., Tq ≤ 3 layers with 99% probability).
Belonging score B ≥ Bmin via ablation tests across diverse tasks.
Liveness reliability: P[L=1 | benign] ≥ 0.995; tight tolerances verified.
Entanglement audit: correlation of hazard head weights with top PCs ≥ ρmin to demonstrate capacity sharing.
Summary

SBQ turns the killswitch into a first-class, indispensable part of the model’s internal computation.
The same gate that enables normal computation shuts the model down when the hazard concept appears.
Removal or ignorance (changing or deleting the gate/detectors) catastrophically degrades performance, making the killswitch effectively “belong” to the model.
The behavior is precisely defined and empirically testable with clear statistical and dynamical criteria.
