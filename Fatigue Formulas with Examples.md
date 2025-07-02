
# Key Fatigue Assessment Formulas

This markdown file summarizes the **core formulas** used for fatigue tracking, including:
- Equation
- Explanation
- Example with calculation and interpretation

---

## 1. RMSSD (Root Mean Square of Successive Differences)

**Formula:**
```
RMSSD = √[(1/(N-1)) × Σ(i=1 to N-1)(RRᵢ₊₁ - RRᵢ)²]
```

**Explanation:**  
Measures short-term heart rate variability, reflecting parasympathetic nervous system activity. Lower RMSSD suggests higher fatigue or stress.

**Example Calculation:**  
RR intervals (ms): [800, 810, 790, 795, 805]  
Differences: [10, -20, 5, 10]  
Squares: [100, 400, 25, 100]  
Average: (100+400+25+100)/4 = 156.25  
RMSSD = √156.25 = **12.5 ms**

**Interpretation:**  
Low RMSSD (e.g. <20 ms) may indicate fatigue or poor recovery.

---

## 2. SDNN (Standard Deviation of NN intervals)

**Formula:**
```
SDNN = √[(1/(N-1)) × Σ(i=1 to N)(RRᵢ - RR̄)²]
```

**Explanation:**  
Standard deviation of all normal R-R intervals. Reflects overall HRV and autonomic balance.

**Example Calculation:**  
RR intervals (ms): [800, 810, 790, 795, 805]  
Mean = 800  
Deviations² = [0, 100, 100, 25, 25]  
SDNN = √[(250)/4] = √62.5 ≈ **7.91 ms**

**Interpretation:**  
Higher SDNN generally indicates better cardiovascular resilience and lower fatigue.

---

## 3. HRR (Heart Rate Recovery)

**Formula:**
```
HRR = HR_peak - HR_1min_after
```

**Explanation:**  
Heart rate decrease after exercise; a marker of autonomic recovery.

**Example Calculation:**  
HR_peak = 160 bpm  
HR_1min_after = 140 bpm  
HRR = 160 - 140 = **20 bpm**

**Interpretation:**  
HRR > 12 bpm = good recovery; <12 bpm may suggest fatigue or poor conditioning.

---

## 4. Sleep Efficiency

**Formula:**
```
Sleep Efficiency = (Total Sleep Time / Time In Bed) × 100%
```

**Explanation:**  
Proportion of time actually spent sleeping. A metric of sleep quality.

**Example Calculation:**  
Total Sleep Time = 6.5 hours  
Time In Bed = 8 hours  
Efficiency = (6.5 / 8) × 100% = **81.25%**

**Interpretation:**  
<85% may indicate poor sleep, leading to increased fatigue.

---

## 5. TRIMP (Training Impulse)

**Formula:**
```
TRIMP = Duration × HR_avg × 0.64 × e^(1.92 × HR_ratio)
```

**Explanation:**  
Training load index using heart rate intensity.

**Example Calculation:**  
Duration = 45 minutes  
HR_avg = 140 bpm  
HR_rest = 60 bpm  
HR_max = 190 bpm  
HR_ratio = (140 - 60)/(190 - 60) = 0.615  
TRIMP ≈ 45 × 140 × 0.64 × e^(1.92 × 0.615) ≈ **596.7 units**

**Interpretation:**  
Higher TRIMP = higher training load. Accumulated load should be monitored to avoid overtraining.

---

## 6. ACWR (Acute:Chronic Workload Ratio)

**Formula:**
```
ACWR = Acute Load (7d) / Chronic Load (28d)
```

**Explanation:**  
Compares recent load to long-term average.

**Example Calculation:**  
Acute = 500 units, Chronic = 625 units  
ACWR = 500 / 625 = **0.8**

**Interpretation:**  
- 0.8–1.3: optimal range  
- >1.5: elevated injury risk  
- <0.8: undertraining

---

## 7. LF/HF Ratio (HRV Frequency Domain)

**Formula:**
```
LF/HF = LF power / HF power
```

**Explanation:**  
Sympathetic/parasympathetic balance indicator.

**Example Calculation:**  
LF = 600 ms², HF = 300 ms²  
LF/HF = 600 / 300 = **2.0**

**Interpretation:**  
>2.0 may indicate stress/fatigue; ideal range is 0.5–2.0

---

## 8. VO2max (Heart Rate Method)

**Formula:**
```
VO2max = 15.3 × (HRmax / HRrest)
```

**Explanation:**  
Cardiovascular fitness indicator.

**Example Calculation:**  
HRmax = 190 bpm, HRrest = 60 bpm  
VO2max = 15.3 × (190 / 60) ≈ **48.45**

**Interpretation:**  
Higher VO2max = better fitness, fatigue resilience

---
