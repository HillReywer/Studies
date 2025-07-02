# Complete Fatigue Tracking Formulas: Theory, Examples, and Implementation

## Table of Contents
1. [Heart Rate Variability (HRV) Metrics](#heart-rate-variability-hrv-metrics)
2. [Heart Rate Based Metrics](#heart-rate-based-metrics)
3. [Sleep Analysis Formulas](#sleep-analysis-formulas)
4. [Activity and Training Load](#activity-and-training-load)
5. [Frequency Domain Analysis](#frequency-domain-analysis)
6. [Non-linear HRV Analysis](#non-linear-hrv-analysis)
7. [Composite Fatigue Scores](#composite-fatigue-scores)
8. [Signal Processing Formulas](#signal-processing-formulas)

---

## 1. Heart Rate Variability (HRV) Metrics

### 1.1 RMSSD (Root Mean Square of Successive Differences)

#### Mathematical Derivation
RMSSD measures short-term HRV by calculating the root mean square of differences between successive heartbeats.

**Formula:**
```
RMSSD = √[(1/(N-1)) × Σ(i=1 to N-1)(RRᵢ₊₁ - RRᵢ)²]
```

**Step-by-step explanation:**
1. Calculate differences between consecutive R-R intervals
2. Square each difference
3. Sum all squared differences
4. Divide by (N-1) for sample variance
5. Take the square root

#### Practical Example
```
R-R intervals (ms): [850, 870, 840, 880, 860]
Differences: [20, -30, 40, -20]
Squared: [400, 900, 1600, 400]
Sum: 3300
RMSSD = √(3300/4) = √825 = 28.72 ms
```

#### Implementation (Python)
```python
def calculate_rmssd(rr_intervals):
    """
    Calculate RMSSD from R-R intervals
    
    Args:
        rr_intervals: List of R-R intervals in milliseconds
    
    Returns:
        RMSSD value in milliseconds
    """
    if len(rr_intervals) < 2:
        return None
    
    differences = []
    for i in range(len(rr_intervals) - 1):
        diff = rr_intervals[i + 1] - rr_intervals[i]
        differences.append(diff ** 2)
    
    mean_squared_diff = sum(differences) / len(differences)
    rmssd = math.sqrt(mean_squared_diff)
    
    return rmssd

# Example usage
rr_intervals = [850, 870, 840, 880, 860]
rmssd = calculate_rmssd(rr_intervals)
print(f"RMSSD: {rmssd:.2f} ms")  # Output: RMSSD: 28.72 ms
```

### 1.2 SDNN (Standard Deviation of NN intervals)

#### Mathematical Derivation
SDNN represents overall HRV by calculating the standard deviation of all R-R intervals.

**Formula:**
```
SDNN = √[(1/(N-1)) × Σ(i=1 to N)(RRᵢ - RR̄)²]
where RR̄ = mean of all R-R intervals
```

**Mathematical proof:**
```
Variance(σ²) = E[(X - μ)²] = Σ(xᵢ - x̄)²/(n-1)
Standard Deviation(σ) = √Variance
```

#### Practical Example
```
R-R intervals: [850, 870, 840, 880, 860]
Mean (RR̄): 860
Deviations: [-10, 10, -20, 20, 0]
Squared deviations: [100, 100, 400, 400, 0]
Sum: 1000
SDNN = √(1000/4) = √250 = 15.81 ms
```

#### Implementation
```python
def calculate_sdnn(rr_intervals):
    """
    Calculate SDNN from R-R intervals
    
    Args:
        rr_intervals: List of R-R intervals in milliseconds
    
    Returns:
        SDNN value in milliseconds
    """
    if len(rr_intervals) < 2:
        return None
    
    mean_rr = sum(rr_intervals) / len(rr_intervals)
    
    squared_deviations = [(rr - mean_rr) ** 2 for rr in rr_intervals]
    variance = sum(squared_deviations) / (len(rr_intervals) - 1)
    sdnn = math.sqrt(variance)
    
    return sdnn

# Example usage
rr_intervals = [850, 870, 840, 880, 860]
sdnn = calculate_sdnn(rr_intervals)
print(f"SDNN: {sdnn:.2f} ms")  # Output: SDNN: 15.81 ms
```

### 1.3 pNN50 (Percentage of successive differences > 50ms)

#### Mathematical Derivation
pNN50 quantifies rapid HRV changes by counting intervals with large successive differences.

**Formula:**
```
pNN50 = (count of |RRᵢ₊₁ - RRᵢ| > 50ms / total number of intervals) × 100%
```

#### Practical Example
```
R-R intervals: [850, 870, 940, 880, 890, 850]
Differences: [20, 70, -60, 10, -40]
|Differences|: [20, 70, 60, 10, 40]
Count > 50ms: 2 (70 and 60)
pNN50 = (2/5) × 100% = 40%
```

#### Implementation
```python
def calculate_pnn50(rr_intervals):
    """
    Calculate pNN50 from R-R intervals
    
    Args:
        rr_intervals: List of R-R intervals in milliseconds
    
    Returns:
        pNN50 as percentage
    """
    if len(rr_intervals) < 2:
        return None
    
    differences = []
    for i in range(len(rr_intervals) - 1):
        diff = abs(rr_intervals[i + 1] - rr_intervals[i])
        differences.append(diff)
    
    count_over_50 = sum(1 for diff in differences if diff > 50)
    pnn50 = (count_over_50 / len(differences)) * 100
    
    return pnn50

# Example usage
rr_intervals = [850, 870, 940, 880, 890, 850]
pnn50 = calculate_pnn50(rr_intervals)
print(f"pNN50: {pnn50:.1f}%")  # Output: pNN50: 40.0%
```

---

## 2. Heart Rate Based Metrics

### 2.1 Heart Rate Recovery (HRR)

#### Mathematical Derivation
HRR follows an exponential decay model representing autonomic nervous system recovery.

**Exponential Model:**
```
HR(t) = HR_rest + (HR_peak - HR_rest) × e^(-t/τ)
```

Where:
- τ (tau) = time constant of recovery
- t = time after exercise cessation

**Solving for τ:**
```
ln[(HR(t) - HR_rest)/(HR_peak - HR_rest)] = -t/τ
τ = -t / ln[(HR(t) - HR_rest)/(HR_peak - HR_rest)]
```

#### Practical Example
```
HR_peak = 180 bpm
HR_rest = 60 bpm
HR_1min = 150 bpm

HRR_1min = 180 - 150 = 30 bpm

Calculate τ:
τ = -1 / ln[(150-60)/(180-60)]
τ = -1 / ln(90/120)
τ = -1 / ln(0.75)
τ = -1 / (-0.288)
τ = 3.47 minutes
```

#### Implementation
```python
def calculate_hrr_metrics(hr_peak, hr_samples, time_points):
    """
    Calculate HRR metrics and recovery time constant
    
    Args:
        hr_peak: Peak heart rate during exercise
        hr_samples: List of HR measurements during recovery
        time_points: Corresponding time points in minutes
    
    Returns:
        Dictionary with HRR values and tau
    """
    import numpy as np
    
    # Simple HRR calculations
    hrr_1min = hr_peak - hr_samples[0] if len(hr_samples) > 0 else None
    hrr_2min = hr_peak - hr_samples[1] if len(hr_samples) > 1 else None
    
    # Fit exponential model to find tau
    hr_rest = hr_samples[-1]  # Assume last sample is near rest
    
    # Transform for linear regression: ln(HR-HR_rest) = ln(A) - t/tau
    y_transform = np.log(np.array(hr_samples) - hr_rest)
    
    # Linear regression
    slope, intercept = np.polyfit(time_points, y_transform, 1)
    tau = -1 / slope
    
    return {
        'hrr_1min': hrr_1min,
        'hrr_2min': hrr_2min,
        'tau': tau,
        'recovery_quality': 'good' if hrr_1min > 12 else 'poor'
    }

# Example usage
hr_peak = 180
hr_recovery = [150, 130, 110, 90, 75, 65]
time_points = [1, 2, 3, 4, 5, 6]

results = calculate_hrr_metrics(hr_peak, hr_recovery, time_points)
print(f"HRR 1min: {results['hrr_1min']} bpm")
print(f"Recovery tau: {results['tau']:.2f} minutes")
```

### 2.2 Heart Rate Reserve Utilization

#### Mathematical Derivation
HR Reserve represents the difference between maximum and resting heart rate.

**Formulas:**
```
HR_reserve = HR_max - HR_rest
HR_reserve_used = (HR_exercise - HR_rest) / (HR_max - HR_rest) × 100%

HR_max estimation (Tanaka formula):
HR_max = 208 - 0.7 × age
```

#### Practical Example
```
Age = 30 years
HR_rest = 60 bpm
HR_exercise = 150 bpm

HR_max = 208 - 0.7 × 30 = 187 bpm
HR_reserve = 187 - 60 = 127 bpm
HR_reserve_used = (150 - 60) / 127 × 100% = 70.9%
```

---

## 3. Sleep Analysis Formulas

### 3.1 Sleep Stage Detection Using HRV and Movement

#### Mathematical Derivation
Sleep stages correlate with autonomic nervous system activity patterns.

**Feature Extraction:**
```
HRV_ratio = HF_power / (LF_power + HF_power)
Movement_index = √(acc_x² + acc_y² + acc_z²) - 1g

Sleep_stage_probability = softmax(W × [HRV_ratio, Movement_index, HR] + b)
```

#### Practical Algorithm
```python
def detect_sleep_stage(hrv_features, movement_data, heart_rate):
    """
    Detect sleep stage using multiple features
    
    Args:
        hrv_features: Dict with HF and LF power
        movement_data: Accelerometer magnitude
        heart_rate: Current HR
    
    Returns:
        Predicted sleep stage
    """
    # Feature calculation
    hrv_ratio = hrv_features['hf'] / (hrv_features['lf'] + hrv_features['hf'])
    movement_index = np.mean(movement_data)
    hr_normalized = (heart_rate - 50) / 30  # Normalize HR
    
    # Simple rule-based classification
    if movement_index > 0.1:
        return 'awake'
    elif hrv_ratio > 0.6 and hr_normalized < 0:
        return 'deep_sleep'
    elif hrv_ratio < 0.4 and movement_index < 0.02:
        return 'rem_sleep'
    else:
        return 'light_sleep'

# Example usage
hrv_features = {'hf': 450, 'lf': 300}
movement_data = [0.01, 0.02, 0.01, 0.015]
heart_rate = 55

stage = detect_sleep_stage(hrv_features, movement_data, heart_rate)
print(f"Detected sleep stage: {stage}")
```

### 3.2 Sleep Debt Calculation

#### Mathematical Model
Sleep debt accumulates non-linearly with an exponential decay component.

**Formula:**
```
Sleep_debt(t) = Sleep_debt(t-1) × decay_factor + (Required_sleep - Actual_sleep)
decay_factor = e^(-1/recovery_time_constant)
```

#### Implementation
```python
def calculate_sleep_debt(sleep_history, required_sleep=8.0, tau=7.0):
    """
    Calculate cumulative sleep debt with decay
    
    Args:
        sleep_history: List of daily sleep durations
        required_sleep: Required sleep hours
        tau: Recovery time constant in days
    
    Returns:
        Current sleep debt in hours
    """
    decay_factor = math.exp(-1/tau)
    sleep_debt = 0
    
    for actual_sleep in sleep_history:
        # Apply decay to existing debt
        sleep_debt *= decay_factor
        
        # Add new debt/recovery
        daily_debt = required_sleep - actual_sleep
        sleep_debt += daily_debt
        
        # Prevent negative debt
        sleep_debt = max(0, sleep_debt)
    
    return sleep_debt

# Example: 7 days of sleep history
sleep_history = [6.5, 7.0, 5.5, 8.5, 7.0, 6.0, 9.0]
debt = calculate_sleep_debt(sleep_history)
print(f"Current sleep debt: {debt:.1f} hours")
```

---

## 4. Activity and Training Load

### 4.1 TRIMP (Training Impulse)

#### Mathematical Derivation
TRIMP quantifies training load using exponential HR weighting.

**Original Banister TRIMP:**
```
TRIMP = Duration × HR_avg × 0.64 × e^(1.92 × HR_ratio)

Where HR_ratio = (HR_avg - HR_rest) / (HR_max - HR_rest)
```

**Lucia TRIMP (zone-based):**
```
TRIMP = Σ(time_in_zone × zone_weight)
Zone 1 (< VT1): weight = 1
Zone 2 (VT1-VT2): weight = 2
Zone 3 (> VT2): weight = 3
```

#### Practical Example
```
Duration = 45 minutes
HR_avg = 150 bpm
HR_rest = 60 bpm
HR_max = 185 bpm

HR_ratio = (150 - 60) / (185 - 60) = 0.72
TRIMP = 45 × 150 × 0.64 × e^(1.92 × 0.72)
TRIMP = 45 × 150 × 0.64 × e^1.38
TRIMP = 45 × 150 × 0.64 × 3.97
TRIMP = 17,146 (arbitrary units)
```

#### Implementation
```python
def calculate_trimp(duration_min, hr_data, hr_rest, hr_max, gender='male'):
    """
    Calculate TRIMP (Training Impulse)
    
    Args:
        duration_min: Exercise duration in minutes
        hr_data: List of HR measurements during exercise
        hr_rest: Resting heart rate
        hr_max: Maximum heart rate
        gender: 'male' or 'female' for coefficient
    
    Returns:
        TRIMP value
    """
    # Gender-specific coefficients
    coef_a = 0.64 if gender == 'male' else 0.86
    coef_b = 1.92 if gender == 'male' else 1.67
    
    # Calculate average HR
    hr_avg = sum(hr_data) / len(hr_data)
    
    # Calculate HR ratio
    hr_ratio = (hr_avg - hr_rest) / (hr_max - hr_rest)
    
    # Calculate TRIMP
    trimp = duration_min * hr_avg * coef_a * math.exp(coef_b * hr_ratio)
    
    return trimp

# Example usage
duration = 45
hr_data = [140, 145, 150, 155, 150, 145]
hr_rest = 60
hr_max = 185

trimp = calculate_trimp(duration, hr_data, hr_rest, hr_max)
print(f"TRIMP: {trimp:.0f} AU")
```

### 4.2 Acute:Chronic Workload Ratio (ACWR)

#### Mathematical Model
ACWR uses exponentially weighted moving averages for injury risk assessment.

**Formula:**
```
ACWR = Acute_Load / Chronic_Load

Exponentially Weighted Moving Average (EWMA):
EWMA_today = α × Load_today + (1 - α) × EWMA_yesterday

Where α = 2/(N+1), N = time window in days
```

#### Implementation
```python
def calculate_acwr(daily_loads, acute_window=7, chronic_window=28):
    """
    Calculate Acute:Chronic Workload Ratio
    
    Args:
        daily_loads: List of daily training loads
        acute_window: Days for acute load (default 7)
        chronic_window: Days for chronic load (default 28)
    
    Returns:
        ACWR value and risk category
    """
    if len(daily_loads) < chronic_window:
        return None, "Insufficient data"
    
    # Calculate exponential weights
    alpha_acute = 2 / (acute_window + 1)
    alpha_chronic = 2 / (chronic_window + 1)
    
    # Initialize EWMA
    ewma_acute = daily_loads[0]
    ewma_chronic = daily_loads[0]
    
    # Calculate EWMA for entire period
    for load in daily_loads[1:]:
        ewma_acute = alpha_acute * load + (1 - alpha_acute) * ewma_acute
        ewma_chronic = alpha_chronic * load + (1 - alpha_chronic) * ewma_chronic
    
    # Calculate ACWR
    acwr = ewma_acute / ewma_chronic if ewma_chronic > 0 else 0
    
    # Risk categorization
    if acwr < 0.8:
        risk = "Low (undertraining)"
    elif 0.8 <= acwr <= 1.3:
        risk = "Optimal"
    elif 1.3 < acwr <= 1.5:
        risk = "Moderate"
    else:
        risk = "High"
    
    return acwr, risk

# Example: 30 days of training loads
import random
daily_loads = [random.randint(20, 80) for _ in range(30)]
acwr, risk = calculate_acwr(daily_loads)
print(f"ACWR: {acwr:.2f}, Risk: {risk}")
```

---

## 5. Frequency Domain Analysis

### 5.1 Fast Fourier Transform (FFT) for HRV

#### Mathematical Foundation
FFT converts time-domain HRV data to frequency domain for spectral analysis.

**Discrete Fourier Transform:**
```
X(k) = Σ(n=0 to N-1) x(n) × e^(-j2πkn/N)

Power Spectral Density:
PSD(f) = |X(f)|² / N
```

**HRV Frequency Bands:**
- VLF: 0.003-0.04 Hz
- LF: 0.04-0.15 Hz  
- HF: 0.15-0.4 Hz

#### Implementation
```python
def calculate_hrv_frequency_domain(rr_intervals, sampling_rate=4):
    """
    Calculate frequency domain HRV metrics
    
    Args:
        rr_intervals: List of R-R intervals in ms
        sampling_rate: Resampling rate in Hz
    
    Returns:
        Dictionary with VLF, LF, HF power and LF/HF ratio
    """
    import numpy as np
    from scipy import signal, interpolate
    
    # Create time array
    time_cumsum = np.cumsum(rr_intervals) / 1000  # Convert to seconds
    
    # Interpolate to uniform sampling
    time_uniform = np.arange(0, time_cumsum[-1], 1/sampling_rate)
    f_interpolate = interpolate.interp1d(time_cumsum, rr_intervals, kind='cubic')
    rr_uniform = f_interpolate(time_uniform)
    
    # Remove mean
    rr_detrended = rr_uniform - np.mean(rr_uniform)
    
    # Calculate PSD using Welch's method
    freqs, psd = signal.welch(rr_detrended, fs=sampling_rate, nperseg=256)
    
    # Calculate power in each band
    vlf_mask = (freqs >= 0.003) & (freqs < 0.04)
    lf_mask = (freqs >= 0.04) & (freqs < 0.15)
    hf_mask = (freqs >= 0.15) & (freqs < 0.4)
    
    vlf_power = np.trapz(psd[vlf_mask], freqs[vlf_mask])
    lf_power = np.trapz(psd[lf_mask], freqs[lf_mask])
    hf_power = np.trapz(psd[hf_mask], freqs[hf_mask])
    
    total_power = vlf_power + lf_power + hf_power
    lf_hf_ratio = lf_power / hf_power if hf_power > 0 else 0
    
    return {
        'vlf_power': vlf_power,
        'lf_power': lf_power,
        'hf_power': hf_power,
        'total_power': total_power,
        'lf_hf_ratio': lf_hf_ratio,
        'lf_nu': lf_power / (lf_power + hf_power) * 100,
        'hf_nu': hf_power / (lf_power + hf_power) * 100
    }

# Example usage
rr_intervals = [850, 870, 840, 880, 860, 890, 850, 870]
freq_metrics = calculate_hrv_frequency_domain(rr_intervals)
print(f"LF/HF Ratio: {freq_metrics['lf_hf_ratio']:.2f}")
```

---

## 6. Non-linear HRV Analysis

### 6.1 Sample Entropy (SampEn)

#### Mathematical Definition
Sample Entropy measures complexity and regularity in time series.

**Formula:**
```
SampEn(m, r, N) = -ln[A/B]

Where:
- m = pattern length
- r = tolerance (typically 0.2 × SDNN)
- A = number of template matches of length m+1
- B = number of template matches of length m
```

#### Implementation
```python
def calculate_sample_entropy(data, m=2, r=None):
    """
    Calculate Sample Entropy
    
    Args:
        data: Time series data
        m: Pattern length
        r: Tolerance (if None, uses 0.2 * std)
    
    Returns:
        Sample entropy value
    """
    import numpy as np
    
    N = len(data)
    if r is None:
        r = 0.2 * np.std(data)
    
    def _maxdist(xi, xj, m):
        """Calculate maximum distance between patterns"""
        return max([abs(float(xi[k]) - float(xj[k])) for k in range(m)])
    
    def _phi(m):
        """Calculate phi(m)"""
        templates = np.array([data[i:i+m] for i in range(N-m+1)])
        B = 0
        
        for i in range(N-m):
            for j in range(i+1, N-m+1):
                if _maxdist(templates[i], templates[j], m) <= r:
                    B += 1
        
        return B
    
    B = _phi(m)
    A = _phi(m+1)
    
    if A == 0 or B == 0:
        return float('inf')
    
    return -np.log(float(A) / float(B))

# Example usage
rr_intervals = [850, 870, 840, 880, 860, 890, 850, 870, 845, 875]
sampen = calculate_sample_entropy(rr_intervals)
print(f"Sample Entropy: {sampen:.3f}")
```

### 6.2 Detrended Fluctuation Analysis (DFA)

#### Mathematical Foundation
DFA quantifies fractal scaling properties of physiological time series.

**Algorithm:**
1. Integrate the time series: y(k) = Σ(i=1 to k)[RR(i) - RR_mean]
2. Divide into boxes of size n
3. Fit polynomial trend in each box
4. Calculate fluctuation F(n)
5. Repeat for different box sizes
6. Find scaling exponent α from log(F(n)) vs log(n)

#### Implementation
```python
def calculate_dfa_alpha1(rr_intervals, min_box=4, max_box=16):
    """
    Calculate DFA α1 (short-term scaling exponent)
    
    Args:
        rr_intervals: R-R interval time series
        min_box: Minimum box size
        max_box: Maximum box size
    
    Returns:
        α1 scaling exponent
    """
    import numpy as np
    
    # Convert to numpy array
    data = np.array(rr_intervals)
    N = len(data)
    
    # Step 1: Integrate
    mean_rr = np.mean(data)
    y = np.cumsum(data - mean_rr)
    
    # Step 2-5: Calculate F(n) for different box sizes
    box_sizes = []
    fluctuations = []
    
    for box_size in range(min_box, min(max_box+1, N//4)):
        # Number of boxes
        num_boxes = N // box_size
        
        # Calculate fluctuation for this box size
        F_n_squared = 0
        
        for i in range(num_boxes):
            # Extract box
            start = i * box_size
            end = start + box_size
            box = y[start:end]
            
            # Fit linear trend
            x = np.arange(box_size)
            coeffs = np.polyfit(x, box, 1)
            fit = np.polyval(coeffs, x)
            
            # Calculate residual
            residual = box - fit
            F_n_squared += np.sum(residual**2)
        
        # Average fluctuation
        F_n = np.sqrt(F_n_squared / (num_boxes * box_size))
        
        box_sizes.append(box_size)
        fluctuations.append(F_n)
    
    # Step 6: Find scaling exponent
    log_n = np.log(box_sizes)
    log_F = np.log(fluctuations)
    
    alpha1, _ = np.polyfit(log_n, log_F, 1)
    
    return alpha1

# Example usage
rr_intervals = [850, 870, 840, 880, 860, 890, 850, 870, 845, 875] * 5
alpha1 = calculate_dfa_alpha1(rr_intervals)
print(f"DFA α1: {alpha1:.3f}")
```

---

## 7. Composite Fatigue Scores

### 7.1 Multi-Component Fatigue Score

#### Mathematical Model
Combines multiple physiological indicators using weighted normalization.

**Formula:**
```
Fatigue_Score = Σ(i=1 to n) wᵢ × normalize(parameterᵢ)

Where normalize(x) = (x - μ_personal) / σ_personal

Z-score boundaries:
- Normal: |z| < 1.5
- Mild fatigue: 1.5 ≤ |z| < 2.5
- Significant fatigue: |z| ≥ 2.5
```

#### Implementation
```python
class FatigueScoreCalculator:
    def __init__(self):
        self.weights = {
            'hrv_rmssd': 0.30,
            'sleep_efficiency': 0.25,
            'training_load': 0.20,
            'resting_hr': 0.15,
            'recovery_time': 0.10
        }
        self.baselines = {}
        self.std_devs = {}
    
    def calibrate(self, historical_data):
        """
        Establish personal baselines from historical data
        
        Args:
            historical_data: Dict of parameter arrays
        """
        for param, values in historical_data.items():
            self.baselines[param] = np.mean(values)
            self.std_devs[param] = np.std(values)
    
    def calculate_fatigue_score(self, current_metrics):
        """
        Calculate composite fatigue score
        
        Args:
            current_metrics: Dict of current parameter values
        
        Returns:
            Fatigue score (0-100) and component scores
        """
        component_scores = {}
        weighted_sum = 0
        
        for param, value in current_metrics.items():
            if param in self.baselines:
                # Calculate z-score
                z_score = (value - self.baselines[param]) / self.std_devs[param]
                
                # Invert for parameters where lower = more fatigue
                if param in ['hrv_rmssd', 'sleep_efficiency']:
                    z_score = -z_score
                
                # Convert to 0-100 scale (sigmoid transformation)
                normalized = 100 / (1 + np.exp(-z_score))
                
                component_scores[param] = normalized
                weighted_sum += self.weights.get(param, 0) * normalized
        
        # Overall fatigue score
        fatigue_score = weighted_sum / sum(self.weights.values())
        
        return fatigue_score, component_scores

# Example usage
calculator = FatigueScoreCalculator()

# Calibrate with historical data
historical_data = {
    'hrv_rmssd': [28, 32, 30, 35, 29, 31, 33, 30, 34, 32],
    'sleep_efficiency': [85, 87, 82, 90, 84, 86, 88, 83, 89, 85],
    'training_load': [45, 50, 40, 55, 48, 52, 47, 49, 51, 46],
    'resting_hr': [58, 60, 57, 62, 59, 61, 58, 60, 59, 60],
    'recovery_time': [22, 25, 20, 28, 23, 26, 24, 25, 27, 24]
}

calculator.calibrate(historical_data)

# Current measurements indicating fatigue
current_metrics = {
    'hrv_rmssd': 22,  # Lower than baseline
    'sleep_efficiency': 75,  # Lower than baseline
    'training_load': 65,  # Higher than baseline
    'resting_hr': 65,  # Higher than baseline
    'recovery_time': 35  # Longer than baseline
}

fatigue_score, components = calculator.calculate_fatigue_score(current_metrics)
print(f"Overall Fatigue Score: {fatigue_score:.1f}/100")
print("Component Scores:")
for param, score in components.items():
    print(f"  {param}: {score:.1f}")
```

### 7.2 Body Battery Algorithm (Garmin-style)

#### Mathematical Model
Energy reservoir model with continuous drain and recovery.

**Differential Equation:**
```
dE/dt = R(t) - D(t)

Where:
E(t) = Energy level at time t (0-100)
R(t) = Recovery rate function
D(t) = Drain rate function

Recovery: R(t) = k_r × (1 - E(t)/100) × recovery_quality
Drain: D(t) = k_d × stress_level × activity_intensity
```

#### Implementation
```python
class BodyBatterySimulator:
    def __init__(self, initial_battery=50):
        self.battery = initial_battery
        self.max_battery = 100
        self.min_battery = 0
        
        # Rate constants
        self.k_recovery = 4.0  # Units per hour at rest
        self.k_drain = 8.0     # Units per hour at high stress
    
    def calculate_recovery_rate(self, hrv_score, sleep_quality=None):
        """
        Calculate recovery rate based on HRV and sleep
        
        Args:
            hrv_score: Normalized HRV score (0-1)
            sleep_quality: Sleep quality score (0-1) if sleeping
        
        Returns:
            Recovery rate in units per hour
        """
        # Base recovery proportional to HRV
        base_recovery = self.k_recovery * hrv_score
        
        # Boost during sleep
        if sleep_quality is not None:
            sleep_multiplier = 1.5 + 0.5 * sleep_quality
            base_recovery *= sleep_multiplier
        
        # Reduce recovery as battery fills (diminishing returns)
        capacity_factor = 1 - (self.battery / self.max_battery)
        
        return base_recovery * capacity_factor
    
    def calculate_drain_rate(self, stress_score, activity_intensity):
        """
        Calculate energy drain rate
        
        Args:
            stress_score: Stress level (0-1)
            activity_intensity: Physical activity intensity (0-1)
        
        Returns:
            Drain rate in units per hour
        """
        # Stress component
        stress_drain = self.k_drain * stress_score * 0.3
        
        # Activity component (exponential scaling)
        activity_drain = self.k_drain * (np.exp(activity_intensity) - 1)
        
        return stress_drain + activity_drain
    
    def update(self, hours, hrv_score, stress_score, activity_intensity, 
               is_sleeping=False, sleep_quality=None):
        """
        Update battery level over time period
        
        Args:
            hours: Time period in hours
            hrv_score: HRV-based recovery score (0-1)
            stress_score: Stress level (0-1)
            activity_intensity: Activity intensity (0-1)
            is_sleeping: Whether person is sleeping
            sleep_quality: Sleep quality if sleeping
        
        Returns:
            New battery level
        """
        # Calculate rates
        recovery_rate = self.calculate_recovery_rate(
            hrv_score, 
            sleep_quality if is_sleeping else None
        )
        
        drain_rate = self.calculate_drain_rate(stress_score, activity_intensity)
        
        # Net change
        net_rate = recovery_rate - drain_rate
        
        # Update battery
        self.battery += net_rate * hours
        
        # Clamp to valid range
        self.battery = max(self.min_battery, min(self.max_battery, self.battery))
        
        return self.battery

# Example: Simulate a day
battery = BodyBatterySimulator(initial_battery=75)

# Morning routine (1 hour)
battery.update(hours=1, hrv_score=0.7, stress_score=0.3, activity_intensity=0.2)
print(f"After morning routine: {battery.battery:.1f}")

# Work period (4 hours)
battery.update(hours=4, hrv_score=0.5, stress_score=0.6, activity_intensity=0.1)
print(f"After work: {battery.battery:.1f}")

# Exercise (1 hour)
battery.update(hours=1, hrv_score=0.4, stress_score=0.2, activity_intensity=0.8)
print(f"After exercise: {battery.battery:.1f}")

# Evening rest (2 hours)
battery.update(hours=2, hrv_score=0.8, stress_score=0.2, activity_intensity=0.05)
print(f"Before sleep: {battery.battery:.1f}")

# Sleep (8 hours)
battery.update(hours=8, hrv_score=0.9, stress_score=0.1, activity_intensity=0, 
               is_sleeping=True, sleep_quality=0.85)
print(f"After sleep: {battery.battery:.1f}")
```

---

## 8. Signal Processing Formulas

### 8.1 R-R Interval Detection from PPG

#### Mathematical Background
PPG signals require preprocessing and peak detection for accurate R-R intervals.

**Signal Processing Pipeline:**
```
1. Bandpass filter: 0.5-4 Hz
2. Signal derivative for peak enhancement
3. Squaring for positive values
4. Moving average for smoothing
5. Adaptive threshold peak detection
```

#### Implementation
```python
def detect_rr_intervals_from_ppg(ppg_signal, sampling_rate=100):
    """
    Extract R-R intervals from PPG signal
    
    Args:
        ppg_signal: Raw PPG signal array
        sampling_rate: Sampling frequency in Hz
    
    Returns:
        List of R-R intervals in milliseconds
    """
    from scipy import signal
    import numpy as np
    
    # 1. Bandpass filter (0.5-4 Hz for heart rate)
    nyquist = sampling_rate / 2
    low = 0.5 / nyquist
    high = 4.0 / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    filtered = signal.filtfilt(b, a, ppg_signal)
    
    # 2. Derivative (emphasize slopes)
    derivative = np.diff(filtered)
    
    # 3. Square (make all positive, emphasize large slopes)
    squared = derivative ** 2
    
    # 4. Moving average (smooth)
    window_size = int(0.1 * sampling_rate)  # 100ms window
    ma_filter = np.ones(window_size) / window_size
    smoothed = np.convolve(squared, ma_filter, mode='same')
    
    # 5. Find peaks with adaptive threshold
    mean_val = np.mean(smoothed)
    std_val = np.std(smoothed)
    threshold = mean_val + 0.5 * std_val
    
    # Minimum distance between peaks (300ms = 200 bpm max)
    min_distance = int(0.3 * sampling_rate)
    
    peaks, _ = signal.find_peaks(smoothed, 
                                  height=threshold,
                                  distance=min_distance)
    
    # Calculate R-R intervals
    rr_intervals = []
    for i in range(1, len(peaks)):
        rr_ms = (peaks[i] - peaks[i-1]) * 1000 / sampling_rate
        
        # Basic artifact rejection (300-2000ms valid range)
        if 300 < rr_ms < 2000:
            rr_intervals.append(rr_ms)
    
    return rr_intervals

# Example usage
# Generate synthetic PPG signal
t = np.linspace(0, 10, 1000)  # 10 seconds at 100 Hz
# Simulate PPG with varying heart rate
heart_rate = 60 + 10 * np.sin(2 * np.pi * 0.1 * t)  # Varying HR
ppg = np.sin(2 * np.pi * heart_rate/60 * t) + 0.1 * np.random.randn(len(t))

rr_intervals = detect_rr_intervals_from_ppg(ppg, sampling_rate=100)
print(f"Detected {len(rr_intervals)} R-R intervals")
print(f"Mean RR: {np.mean(rr_intervals):.1f} ms")
print(f"Mean HR: {60000/np.mean(rr_intervals):.1f} bpm")
```

### 8.2 Respiratory Rate from PPG

#### Mathematical Approach
Respiratory modulation appears as low-frequency oscillations in PPG amplitude and baseline.

**Extraction Methods:**
1. Amplitude Modulation (AM): Track peak heights
2. Baseline Wander (BW): Track signal baseline
3. Frequency Modulation (FM): Track R-R interval variations

#### Implementation
```python
def extract_respiratory_rate_from_ppg(ppg_signal, sampling_rate=100):
    """
    Extract respiratory rate from PPG signal
    
    Args:
        ppg_signal: PPG signal array
        sampling_rate: Sampling frequency in Hz
    
    Returns:
        Respiratory rate in breaths per minute
    """
    from scipy import signal
    import numpy as np
    
    # Method 1: Baseline wander extraction
    # Low-pass filter to get baseline (respiratory component)
    nyquist = sampling_rate / 2
    resp_cutoff = 0.5 / nyquist  # 0.5 Hz = 30 breaths/min max
    b, a = signal.butter(4, resp_cutoff, btype='low')
    baseline = signal.filtfilt(b, a, ppg_signal)
    
    # Remove DC component
    baseline_ac = baseline - np.mean(baseline)
    
    # Method 2: Amplitude modulation from peaks
    # Find PPG peaks
    peaks, properties = signal.find_peaks(ppg_signal, 
                                          distance=int(0.5*sampling_rate))
    peak_amplitudes = properties.get('peak_heights', ppg_signal[peaks])
    
    # Interpolate peak amplitudes to uniform sampling
    if len(peaks) > 3:
        peak_times = peaks / sampling_rate
        uniform_time = np.linspace(0, len(ppg_signal)/sampling_rate, 
                                   len(ppg_signal))
        am_signal = np.interp(uniform_time, peak_times, peak_amplitudes)
        am_signal = am_signal - np.mean(am_signal)
    else:
        am_signal = baseline_ac  # Fallback to baseline method
    
    # Combine methods (weighted average)
    combined_resp = 0.6 * baseline_ac + 0.4 * am_signal
    
    # FFT to find respiratory frequency
    freqs = np.fft.fftfreq(len(combined_resp), 1/sampling_rate)
    fft = np.abs(np.fft.fft(combined_resp))
    
    # Find peak in respiratory range (0.1-0.5 Hz = 6-30 breaths/min)
    resp_mask = (freqs > 0.1) & (freqs < 0.5)
    resp_freqs = freqs[resp_mask]
    resp_fft = fft[resp_mask]
    
    if len(resp_fft) > 0:
        peak_freq = resp_freqs[np.argmax(resp_fft)]
        respiratory_rate = peak_freq * 60  # Convert to breaths/min
    else:
        respiratory_rate = 15  # Default fallback
    
    return respiratory_rate

# Example usage
# Generate synthetic PPG with respiratory modulation
t = np.linspace(0, 60, 6000)  # 60 seconds at 100 Hz
heart_rate = 70
resp_rate = 15 / 60  # 15 breaths/min in Hz

# PPG = cardiac component + respiratory modulation
cardiac = np.sin(2 * np.pi * heart_rate/60 * t)
respiratory = 0.1 * np.sin(2 * np.pi * resp_rate * t)
ppg = cardiac * (1 + respiratory) + 0.05 * np.random.randn(len(t))

resp_rate_detected = extract_respiratory_rate_from_ppg(ppg, sampling_rate=100)
print(f"Detected respiratory rate: {resp_rate_detected:.1f} breaths/min")
```

### 8.3 Motion Artifact Detection and Correction

#### Mathematical Framework
Motion artifacts corrupt physiological signals and must be identified and corrected.

**Detection Criteria:**
```
1. Signal Quality Index (SQI) = 1 - (noise_power / signal_power)
2. Template matching correlation
3. Physiological range validation
```

#### Implementation
```python
def detect_and_correct_artifacts(signal_data, signal_type='ppg', sampling_rate=100):
    """
    Detect and correct motion artifacts in physiological signals
    
    Args:
        signal_data: Raw signal array
        signal_type: Type of signal ('ppg', 'ecg', etc.)
        sampling_rate: Sampling frequency
    
    Returns:
        Corrected signal and artifact mask
    """
    import numpy as np
    from scipy import signal, stats
    
    # Parameters based on signal type
    if signal_type == 'ppg':
        valid_range = (-3, 3)  # Normalized units
        max_rate_change = 0.2  # 20% change between samples
    
    # 1. Statistical artifact detection
    # Calculate local statistics in sliding windows
    window_size = int(2 * sampling_rate)  # 2-second windows
    stride = window_size // 4
    
    artifact_mask = np.zeros(len(signal_data), dtype=bool)
    
    for i in range(0, len(signal_data) - window_size, stride):
        window = signal_data[i:i+window_size]
        
        # Check for outliers
        z_scores = np.abs(stats.zscore(window))
        outliers = z_scores > 3
        
        # Check for signal clipping
        clipping = (window <= valid_range[0]) | (window >= valid_range[1])
        
        # Check for unrealistic rate of change
        diff = np.abs(np.diff(window))
        rapid_changes = diff > max_rate_change * np.std(window)
        rapid_changes = np.concatenate([[False], rapid_changes])
        
        # Combine artifact indicators
        artifacts = outliers | clipping | rapid_changes
        artifact_mask[i:i+window_size] |= artifacts
    
    # 2. Template correlation for periodic signals
    if signal_type == 'ppg':
        # Extract clean template from artifact-free segments
        clean_segments = []
        for i in range(0, len(signal_data) - window_size, window_size):
            if not np.any(artifact_mask[i:i+window_size]):
                clean_segments.append(signal_data[i:i+window_size])
        
        if clean_segments:
            # Create template from median of clean segments
            template = np.median(clean_segments, axis=0)
            
            # Cross-correlate with signal
            for i in range(0, len(signal_data) - window_size, stride):
                window = signal_data[i:i+window_size]
                correlation = np.corrcoef(window, template)[0, 1]
                
                # Low correlation indicates artifact
                if correlation < 0.5:
                    artifact_mask[i:i+window_size] = True
    
    # 3. Artifact correction
    corrected_signal = signal_data.copy()
    
    # Find artifact regions
    artifact_regions = []
    in_artifact = False
    start = 0
    
    for i, is_artifact in enumerate(artifact_mask):
        if is_artifact and not in_artifact:
            start = i
            in_artifact = True
        elif not is_artifact and in_artifact:
            artifact_regions.append((start, i))
            in_artifact = False
    
    # Correct each artifact region
    for start, end in artifact_regions:
        if end - start < 0.5 * sampling_rate:  # Short artifacts: interpolate
            if start > 0 and end < len(signal_data):
                # Linear interpolation
                x = [start-1, end]
                y = [corrected_signal[start-1], corrected_signal[end]]
                interp_values = np.interp(range(start, end), x, y)
                corrected_signal[start:end] = interp_values
        else:  # Long artifacts: mark as invalid
            corrected_signal[start:end] = np.nan
    
    # Calculate Signal Quality Index
    if not np.all(artifact_mask):
        clean_power = np.var(signal_data[~artifact_mask])
        noise_power = np.var(signal_data[artifact_mask]) if np.any(artifact_mask) else 0
        sqi = 1 - (noise_power / (clean_power + noise_power))
    else:
        sqi = 0
    
    return corrected_signal, artifact_mask, sqi

# Example usage
# Generate signal with artifacts
t = np.linspace(0, 10, 1000)
clean_ppg = np.sin(2 * np.pi * 1.2 * t)  # 72 bpm

# Add motion artifacts
ppg_with_artifacts = clean_ppg.copy()
ppg_with_artifacts[200:250] += 2 * np.random.randn(50)  # Motion spike
ppg_with_artifacts[500:520] = 3  # Clipping
ppg_with_artifacts[700:750] = 0.1 * np.random.randn(50)  # Signal loss

corrected, artifacts, sqi = detect_and_correct_artifacts(ppg_with_artifacts, 'ppg')
print(f"Signal Quality Index: {sqi:.2f}")
print(f"Artifacts detected: {np.sum(artifacts)} samples ({np.mean(artifacts)*100:.1f}%)")
```

---

## Summary: Key Implementation Considerations

### 1. **Validation Requirements**
- Always validate against known physiological ranges
- Implement artifact detection before analysis
- Use multiple methods for critical measurements

### 2. **Computational Optimization**
```python
# Example: Optimized rolling RMSSD calculation
def rolling_rmssd(rr_intervals, window_size=10):
    """Efficient rolling RMSSD using vectorization"""
    diffs = np.diff(rr_intervals)
    squared_diffs = diffs ** 2
    
    # Use convolution for rolling sum
    kernel = np.ones(window_size - 1) / (window_size - 1)
    rolling_mean = np.convolve(squared_diffs, kernel, mode='valid')
    rolling_rmssd = np.sqrt(rolling_mean)
    
    return rolling_rmssd
```

### 3. **Real-time Processing Pipeline**
```python
class RealtimeFatigueProcessor:
    def __init__(self, sampling_rate=100):
        self.sampling_rate = sampling_rate
        self.buffer_size = 5 * 60 * sampling_rate  # 5-minute buffer
        self.signal_buffer = collections.deque(maxlen=self.buffer_size)
        self.rr_intervals = []
        self.last_peak_time = None
    
    def process_sample(self, sample):
        """Process single sample in real-time"""
        self.signal_buffer.append(sample)
        
        # Check if we have enough data
        if len(self.signal_buffer) < self.sampling_rate:
            return None
        
        # Detect peaks in recent window
        recent_window = list(self.signal_buffer)[-self.sampling_rate:]
        
        # Simple peak detection for real-time
        if self.is_peak(recent_window):
            current_time = len(self.signal_buffer) / self.sampling_rate
            
            if self.last_peak_time:
                rr_interval = (current_time - self.last_peak_time) * 1000
                
                if 300 < rr_interval < 2000:  # Valid range
                    self.rr_intervals.append(rr_interval)
                    
                    # Calculate metrics if enough data
                    if len(self.rr_intervals) >= 10:
                        metrics = {
                            'instant_hr': 60000 / rr_interval,
                            'rmssd': calculate_rmssd(self.rr_intervals[-10:]),
                            'timestamp': current_time
                        }
                        return metrics
            
            self.last_peak_time = current_time
        
        return None
    
    def is_peak(self, window):
        """Simple peak detection for real-time processing"""
        mid = len(window) // 2
        return (window[mid] > window[mid-1] and 
                window[mid] > window[mid+1] and
                window[mid] > np.mean(window) + 0.5 * np.std(window))
```

### 4. **Error Handling and Edge Cases**
```python
def safe_hrv_calculation(rr_intervals, metric='rmssd'):
    """Robust HRV calculation with error handling"""
    try:
        # Validate input
        if not rr_intervals or len(rr_intervals) < 2:
            return None, "Insufficient data"
        
        # Remove outliers
        rr_array = np.array(rr_intervals)
        valid_mask = (rr_array > 300) & (rr_array < 2000)
        
        if np.sum(valid_mask) < 2:
            return None, "No valid intervals after filtering"
        
        clean_rr = rr_array[valid_mask]
        
        # Calculate metric
        if metric == 'rmssd':
            result = calculate_rmssd(clean_rr)
        elif metric == 'sdnn':
            result = calculate_sdnn(clean_rr)
        else:
            return None, f"Unknown metric: {metric}"
        
        # Validate result
        if np.isnan(result) or np.isinf(result):
            return None, "Calculation resulted in invalid value"
        
        return result, "Success"
        
    except Exception as e:
        return None, f"Calculation error: {str(e)}"
```

This comprehensive guide provides the theoretical foundation, practical examples, and implementation code for all major fatigue tracking formulas. Each section includes the mathematical derivation, real-world examples with actual numbers, and production-ready Python code that can be adapted for your application.