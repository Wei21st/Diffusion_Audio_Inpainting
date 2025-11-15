# Diffusion-Based Audio Inpainting
This is a class project for ECE661. Collaborator: Yurui Wu


Official repository of the paper:
> E. Moliner and V. Välimäki,, "Diffusion-Based Audio Inpainting",  Journal of the Audio Engineering Society, March 2024

We based our experiment on the diffusion model for audio inpainting proposed by Moliner et al. Since training the original model requires approximately five days, we divided our work into two stages: training and inference, allowing us to avoid retraining the full model. 

For training, we applied a lower fidelity experiment setup by training a proxy model. The proxy model preserved the U-net structure of the original one and used similar residual blocks, but with fewer channels and attention layers, reducing total parameter count to 60M, which is 25 percent of the original model. We also followed the novel method proposed in the original paper to convert audio signals into pitch-aligned spectrograms. Then, we trained the proxy model with three noise noise schedulers:

Power Scheduler
```math
\tau_i = \left( \sigma_{\max}^{1/\rho} + \frac{i}{T-1}(\sigma_{\min}^{1/\rho} - \sigma_{\max}^{1/\rho}) \right)^{\rho}
```

Cosine Scheduler
```math
\tau_i = \sigma_{\text{min}} + \frac{1}{2} (\sigma_{\text{max}} - \sigma_{\text{min}}) \left[1 + \cos\left( \frac{i}{T - 1} \cdot \pi \right)\right]
```
Sigmoid Scheduler
```math
\tau_i = \sigma_{\text{min}} + (\sigma_{\text{max}} - \sigma_{\text{min}}) \cdot 
\frac{1 - \text{sigmoid}\left( \frac{i/T - s}{\tau} \right)}{1 - \text{sigmoid}\left( \frac{-s}{\tau} \right)}
```

where $0 \leq i \leq T - 1$ is the discretization index, $\sigma_{\text{min}}$ and $\sigma_{\text{max}}$ are, respectively, the minimum and maximum noise levels, $\rho$ is a parameter controlling the warping of the schedule, and $s$ is a shift parameter that centers the curve. During training, we also recorded the log of sample noise rate (logSNR) and the squared reconstruction error for later evaluation.

For inference, we directly utilized the authors' pre-trained diffusion model. Importantly, we did not alter any of the learned weights of the model and only changed the noise scheduling scheme used during the denoising (inference) process.

To do this, we kept the pre-trained diffusion model fixed and replaced only the inference-time noise schedule **βₜ** used in the sampling loop. We also experimented with power, cosine, and sigmoid schedulers by redefining the sequence of noise levels **{βₜ}ₜ₌₁⁽ᵀ⁾** based on the functional form of each scheduler.

These new schedules were substituted directly into the denoising steps of the model's sampling loop without retraining, thereby isolating the effect of the inference-time scheduler on output quality.

To evaluate the quality of the generated audio under each scheduler, we used the following metrics:

- **Log-Spectral Distance (LSD)**
- **Frechet Audio Distance (FAD)**
- **Spectral Convergence**
- **Normalized Harmonicity**

---
### Training Results

When training with the cosine scheduler, the model consistently achieved a squared error of around **0.55** across all noise levels during the early and mid stages. However, during the late stage, performance declined for inputs with lower noise levels, with minimal improvement observed for highly noisy inputs. This trend suggests that the cosine scheduler may introduce instability in later stages of training.

Although the model trained with the power scheduler initially struggled to estimate cleaner inputs during the early stage, it ultimately achieved the **lowest error** among the three models. This outcome highlights the potential of power schedulers to achieve the best overall accuracy. Nevertheless, its weaker performance on lower-noise inputs during the late stage indicates that it may also introduce some instability.

The model trained with the sigmoid scheduler demonstrated **stable convergence** during the late training stage, maintaining an acceptable and consistent error rate across all noise levels. Compared to the original setup with 500,000 iterations, this model converged remarkably quickly while achieving competitive accuracy. This fast convergence may occur because sigmoid schedulers introduce higher noise levels earlier in training, which appears particularly helpful under limited computational resources.



---

## Inference Noise Scheduler Evaluation

We used the same inference steps as the original paper to maintain consistency. All inference was performed on a single NVIDIA 4080 GPU. We used a segment length of **4.17 s**, constrained by memory limits. For each condition, we assessed **5 random samples**, and used the **same masked input** to compare different noise schedulers fairly.

We tested samples with gap sizes of **100 ms**, **200 ms**, and **400 ms**, and evaluated performance across schedulers using several objective metrics (Fig. ref:infer-metrics). Because of the limited sample size, we note that the error bars are larger compared with similar systematic studies.

---

### Inference Results

#### Log-Spectral Distance (LSD)

LSD quantifies the difference in frequency content by computing the squared difference between log-magnitude spectrograms.

For a gap length of **100 ms**, the **power scheduler** outperformed the others.  
However, as the gap length increased beyond 100 ms, its performance degraded and became less optimal than the other methods.

#### Fréchet Audio Distance (FAD)

FAD is a strong indicator of perceptual audio quality and artifact detection.

Our experiments showed **no significant difference** in FAD scores across the three scheduling methods. This is consistent with subjective listening impressions: all methods performed comparably in perceptual audio similarity.

#### Spectral Convergence

Spectral Convergence measures similarity in spectral structure and is relatively less sensitive to gap length.

The results matched the LSD trend: once gap length exceeded 100 ms, the **power scheduler** performed worse, indicating reduced ability to maintain spectral fidelity under larger gaps.

#### Normalized Harmonicity

Normalized Harmonicity evaluates the preservation of harmonic–percussive balance between original and inpainted signals.

Across conditions, the **cosine scheduler** consistently performed best, suggesting that it is more effective at preserving harmonic structure during audio inpainting.


