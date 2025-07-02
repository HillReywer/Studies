# Full Technical List of Fatigue Tracking Parameters

## PRIORITY 1: Critical Parameters (Mandatory)

### 1. Heart Rate Variability (HRV)

#### 1.1 RMSSD (Root Mean Square of Successive Differences)

**Formula:**

```
RMSSD = √[(1/(N-1)) × Σ(i=1 to N-1)(RRᵢ₊₁ - RRᵢ)²]
```

* **Units:** milliseconds (ms)
* **Normal range:** 20–100 ms (depends on age)
* **Measurement frequency:** Every 5 minutes at rest
* **Interpretation:** ↓ RMSSD = ↑ fatigue/stress
* **Minimum requirement:** 250 R-R interval points

#### 1.2 SDNN (Standard Deviation of NN intervals)

**Formula:**

```
SDNN = √[(1/(N-1)) × Σ(i=1 to N)(RRᵢ - RR̄)²]
```

* **Units:** milliseconds (ms)
* **Normal range:** 50–150 ms
* **Measurement window:** 5 minutes (short-term) or 24 hours (long-term)
* **Interpretation:** ↓ SDNN = ↓ overall variability = ↑ fatigue

#### 1.3 pNN50 (Percentage of successive RR intervals differing by >50ms)

**Formula:**

```
pNN50 = (count |RRᵢ₊₁ - RRᵢ| > 50ms / total intervals) × 100%
```

* **Units:** percent (%)
* **Normal range:** 5–40%
* **Interpretation:** ↓ pNN50 = ↓ parasympathetic activity

### 2. Heart Rate (HR)

#### 2.1 Resting Heart Rate (RHR)

**Formula:**

```
RHR = 60 / mean(RR intervals at rest)
```

* **Units:** bpm (beats per minute)
* **Normal range:** 50–80 bpm
* **Measurement time:** Morning after waking up, lying down
* **Interpretation:** ↑ RHR by 5–10 bpm = possible fatigue

#### 2.2 Heart Rate Recovery (HRR)

**Formula:**

```
HRR₁ = HR_peak - HR_1min_after
HRR₂ = HR_peak - HR_2min_after

Exponential model:
HR(t) = HR_rest + (HR_peak - HR_rest) × e^(-t/τ)
```

* **Units:** bpm
* **Normal values:** HRR₁ > 12 bpm, HRR₂ > 22 bpm
* **Interpretation:** ↓ HRR = ↓ recovery = ↑ fatigue

### 3. Sleep Metrics

#### 3.1 Sleep Efficiency

**Formula:**

```
Sleep_Efficiency = (Total_Sleep_Time / Time_In_Bed) × 100%
```

* **Units:** percent (%)
* **Target range:** > 85%
* **Calculation:** Based on accelerometer + HR

#### 3.2 Sleep Debt

**Formula:**

```
Sleep_Debt = Σ(i=1 to 14)(Recommended_Sleep - Actual_Sleep)ᵢ / 14
```

* **Units:** hours
* **Critical threshold:** > 2 hours
* **Calculation window:** Last 14 days

#### 3.3 Sleep Stages Distribution

**Detected via HRV + movement:**

```
Deep_Sleep_% = (Deep_Sleep_Time / Total_Sleep_Time) × 100%
REM_Sleep_% = (REM_Sleep_Time / Total_Sleep_Time) × 100%
Light_Sleep_% = (Light_Sleep_Time / Total_Sleep_Time) × 100%
```

* **Target values:** Deep: 15–20%, REM: 20–25%, Light: 50–60%

### 4. Activity Load

#### 4.1 Training Load (TRIMP - Training Impulse)

**Formula:**

```
TRIMP = Duration × HR_avg × 0.64 × e^(1.92 × HR_ratio)
where HR_ratio = (HR_avg - HR_rest) / (HR_max - HR_rest)
```

* **Units:** arbitrary units
* **Interpretation:** Accumulated load from training

#### 4.2 Acute\:Chronic Workload Ratio (ACWR)

**Formula:**

```
ACWR = Acute_Load (7 days) / Chronic_Load (28 days)
```

* **Optimal range:** 0.8–1.3
* **Injury risk:** > 1.5
* **Undertraining:** < 0.8

### 5. Respiratory Rate

**Extracted from PPG formula:**

```
RR = FFT_peak_frequency(respiratory_component) × 60
where respiratory_component = bandpass_filter(PPG_signal, 0.1–0.5 Hz)
```

* **Units:** breaths/min
* **Resting norm:** 12–20 breaths/min
* **Elevation:** +3–5 may indicate fatigue

## PRIORITY 2: Important Additional Parameters

### 6. HRV Frequency Domain

#### 6.1 LF/HF Ratio

**Formula:**

```
LF_power = ∫(0.04–0.15 Hz) PSD(f) df
HF_power = ∫(0.15–0.4 Hz) PSD(f) df
LF/HF_ratio = LF_power / HF_power
```

* **Normal range:** 0.5–2.0
* **Interpretation:** ↑ LF/HF = ↑ sympathetic activity = stress/fatigue

#### 6.2 Total Power

**Formula:**

```
TP = ∫(0.003–0.4 Hz) PSD(f) df
```

* **Units:** ms²
* **Interpretation:** ↓ TP = ↓ overall adaptability

### 7. Movement Analysis

#### 7.1 Activity Counts

**Formula:**

```
Activity_Count = Σ|acceleration_magnitude - 1g| × sampling_rate
```

* **Units:** counts/min
* **Use:** Classifying activity levels

#### 7.2 Step Cadence Variability

**Formula:**

```
CV_cadence = (SD_cadence / mean_cadence) × 100%
```

* **Interpretation:** ↑ variability = ↑ fatigue during walking

### 8. Circadian Rhythm Parameters

#### 8.1 Mesor (mean level)

**Formula:**

```
HR_circadian(t) = Mesor + Amplitude × cos(2π(t - Acrophase)/24)
```

* **Parameters:** Extracted via cosinor analysis
* **Interpretation:** Acrophase shift = circadian rhythm disruption

### 9. Composite Scores

#### 9.1 Body Battery (Garmin-style)

**Formula:**

```
Battery(t) = Battery(t-1) + Recovery_rate - Drain_rate

where:
Recovery_rate = f(HRV, sleep_quality, rest_time)
Drain_rate = g(stress_level, activity_intensity)
```

* **Range:** 0–100
* **Recovery rate:** \~4–8 units/hour at rest

#### 9.2 Readiness Score (Oura-style)

**Formula:**

```
Readiness = w₁×HRV_score + w₂×Sleep_score + w₃×Activity_balance + w₄×Temperature_score

where weights: w₁=0.35, w₂=0.35, w₃=0.20, w₄=0.10
```

## PRIORITY 3: Advanced Parameters

### 10. Nonlinear HRV Metrics

#### 10.1 Sample Entropy (SampEn)

**Formula:**

```
SampEn(m,r,N) = -ln[Cₘ₊₁(r)/Cₘ(r)]
```

* **Parameters:** m=2, r=0.2×SDNN
* **Interpretation:** ↓ SampEn = ↓ complexity = ↑ fatigue

#### 10.2 DFA α1 (Detrended Fluctuation Analysis)

**Formula:**

```
F(n) ∝ n^α
α1 for n = 4–16 beats
```

* **Norm:** α1 ≈ 1.0
* **Interpretation:** α1 < 0.75 indicates fatigue

### 11. Temperature Metrics

#### 11.1 Skin Temperature Variability

**Formula:**

```
Temp_deviation = |Temp_current - Temp_baseline_circadian|
```

* **Units:** °C
* **Threshold:** > 0.5°C deviation may indicate stress

### 12. Derived Metrics

#### 12.1 HRV Slope (trend)

**Formula:**

```
HRV_slope = β from linear regression: RMSSD = α + β×day
```

* **Window:** 7–14 days
* **Interpretation:** Negative slope = accumulating fatigue

## ADDITIONAL FEATURES (with advanced sensors)

### A. SpO₂ (Oxygen Saturation)

**Requirements:** Red + infrared LED

```
SpO2 = (AC_red/DC_red) / (AC_ir/DC_ir) × calibration_coefficient
```

* **Normal:** 95–100%
* **Use:** Apnea detection, altitude adaptation

### B. Electrodermal Activity (EDA)

**Requirements:** Skin conductance sensors

```
SCR_amplitude = max(EDA) - baseline_EDA
SCR_frequency = peak_count / time
```

* **Use:** Emotional stress, mental load

### C. Core Body Temperature

**Requirements:** Specialized temperature sensors

```
Circadian_amplitude = (Temp_max - Temp_min) / 2
Phase_shift = Acrophase_current - Acrophase_normal
```

## CALIBRATION AND PERSONALIZATION

### Establishing Baselines

**Minimum period:** 14 days
**Adaptation formula:**

```
Baseline_personal = α × Baseline_population + (1-α) × Individual_mean
where α = 1/(1 + days_of_data/7)
```

### Z-score Normalization

**For all metrics:**

```
Z_score = (Value_current - Mean_personal) / SD_personal
```

* **Interpretation:** |Z| > 2 = significant deviation

### Weighted Coefficients for Composite Scoring

```
Fatigue_Score = Σwᵢ × normalize(parameterᵢ)

Recommended weights:
- HRV (RMSSD): 0.30
- Sleep Quality: 0.25
- Activity Load: 0.20
- Recovery metrics: 0.15
- Other parameters: 0.10
```

## TECHNICAL REQUIREMENTS

### Sampling Frequency

* **PPG signal:** Minimum 25 Hz, optimal 100–250 Hz
* **Accelerometer:** 50–100 Hz
* **Temperature:** 1 Hz

### Measurement Accuracy

* **R-R intervals:** ±5 ms
* **HR:** ±2 bpm
* **Motion:** Minimum 12-bit resolution

### Power Consumption

* **Continuous HRV monitoring:** \~10–15 mA
* **Periodic measurements:** \~2–5 mA average current
