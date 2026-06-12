# Portfolio: Data Science & Analytics Case Studies

## Modeling Biological Aging: Statistical Integrity & The "Genomic Gap" in NHANES
> **Core Toolkit:** Python, Scikit-Learn, Pandas, NumPy, SQL, Matplotlib

### Executive Summary
This project establishes a supervised learning framework to identify the primary clinical and socioeconomic drivers of mean leukocyte telomere length (`TELOMEAN`) using the multi-year **NHANES (1999–2002)** dataset. Rather than optimizing blindly for predictive scores, this study focuses on **model integrity, rigorous data cleaning, and variance-bias diagnostics** to isolate the environmental and social "weathering" components of biological aging.


### Data Engineering & Targeted Preprocessing
Real-world epidemiological data contains structural traps. The initial dataset of 21,004 records was optimized down to an analytic sample of **5,801 complete observations** using a disciplined, domain-specific ETL pipeline:

* **Sentinel Value Auditing:** Replaced non-biological survey artifacts and invalid placeholder responses (e.g., survey codes 7, 9, 888) with `NaN` on a feature-specific basis to eliminate artificial variance.
* **Leakage Prevention:** Implemented a Scikit-Learn `ColumnTransformer` inside an isolated cross-validation loop. Continuous variables were standardized via `StandardScaler`, and categorical features were processed via `OneHotEncoder(handle_unknown='ignore')` exclusively within training folds.
* **Impact:** Cleaning removed non-biological noise, increasing the baseline Elastic Net $R^2$ from 0.1884 to **0.1930** and dropping the Root Mean Squared Error (RMSE).


### Multi-Architecture Modeling Strategy
To evaluate variance-bias trade-offs, three distinct model architectures were optimized using `GridSearchCV` paired with a strict **5-fold cross-validation** protocol ($k=5$, shuffled with a fixed seed).

              [ Cleaned NHANES Dataset (N = 5,801) ]
                                │
                 [ 5-Fold Cross-Validation Split ]
           ┌────────────────────┼────────────────────┐
           ▼                    ▼                    ▼
   [ Elastic Net ]        [ Linear SVR ]      [ Random Forest ]
(Sparse L1 Regular)     (Margin-Based SVR)    (Nonparametric Ensemble)
           └────────────────────┼────────────────────┘
                                ▼
                     [ Diagnostic Audit Matrix ]
                       Evaluated via R² & RMSE

### Cross-Validation & GridSearch Optimization Results

| Model Architecture | CV Mean $R^2$ | CV Mean RMSE | Optimal Hyperparameters |
| :--- | :---: | :---: | :--- |
| **Elastic Net** | **0.1930** | **0.2323** | `alpha: 0.001`, `L1_ratio: 1.0` (Pure Lasso) |
| **Linear SVR** | 0.1841 | 0.2336 | `C: 0.1`, `epsilon: 0.1` |
| **Random Forest** | 0.1645 | 0.2364 | `max_depth: 10`, `n_estimators: 500` |

> **Architectural Takeaway:** The convergence of the Elastic Net to a pure Lasso solution ($L_1$ ratio = 1.0) indicates that a fully sparse solution is optimal for navigating the intense multicollinearity in the feature space (e.g., the $r = 0.88$ correlation loop between BMI, waist circumference, and weight). The lower comparative performance of the Random Forest suggests that additive linear relationships dominate the signal over complex non-linear interactions within this cohort.


### Model Diagnostics & Explanatory Power

#### 1. Residual Behavior & Specification Validation
Diagnostic analysis of the Elastic Net residuals demonstrates robust model specification:
* Errors are distributed symmetrically around zero across the entire prediction range.
* The absence of curvature or funnel-shaped dispersion verifies that heteroskedasticity is minimized, proving the model is stable even when predicting extreme telomere lengths.

#### 2. Feature Hierarchy: Social Determinants vs. Clinical Labs
Rather than relying on default model coefficients, feature importance was evaluated using **Permutation Importance** (10 repeated shuffles per feature), revealing a compelling structural narrative:

* **Chronological Age (`RIDAGEYR`):** The dominant biological signal ($ Pearson\ r = -0.43 $), acting as the primary baseline benchmark.
* **The SDoH Premium:** Social Determinants of Health (SDoH)—including **Race/Ethnicity (`RIDRETH1`)**, **Marital Status (`DMDMARTL`)**, **Household Size (`DMDHHSIZ`)**, and **Income-to-Poverty Ratio (`INDFMPIR`)**—consistently outranked traditional clinical biomarkers like HbA1c (`LBXGH`) and total cholesterol (`LBXTC`). 
* **Lasso Selection Path Summary:** As the regularization penalty ($\alpha$) increased, clinical labs were eliminated early, while socioeconomic variables stubbornly persisted, showing they carry a more stable, long-term predictive footprint of environmental exposure.

#### 3. Quantifying the "Genomic Gap"
While an $R^2$ of 19.3% is modest for standard corporate KPI tracking, it represents a highly robust signal in non-genetic epidemiological modeling. The remaining residual variance visually encapsulates the well-documented **"Genomic Gap"**—the estimated 60–70% of telomere length variance driven strictly by hereditary factors not captured in survey data. This model successfully captures and simplifies the remaining ~20% environment/physiological variance boundary.

---

##  Digital Platform Strategy: Evaluating Web Performance for The Grammys
> **Core Toolkit:** Python, Pandas, NumPy, Plotly Express, Web Analytics

### Executive Summary
In 2022, The Recording Academy’s VP of Digital Strategy, Ray Starck, executed a major structural pivot: splitting their unified digital presence into two distinct domains—`grammy.com` (consumer/audience-facing) and `recordingacademy.com` (industry/corporate-facing). This case study analyzes historical and post-split tracking data to evaluate the impact of this split, isolate user behavior across platforms, and provide data-driven recommendations to optimize user engagement and retention during peak cultural events.

---

### Data Environment & Feature Engineering
The analysis was performed across multi-year tracking logs containing traffic metrics for both domains. To extract meaningful strategic insights, raw session parameters were transformed into distinct behavioral indicators:

* **KPI Engineering:** Synthesized high-level metrics including **Pages per Session** (`pageviews` / `sessions`) and **Bounce Rate** percentages to quantify deep user engagement versus superficial click-through behavior.
* **Temporal Event Mapping:** Engineered specific binary temporal flags—`awards_week` and `awards_night`—to separate baseline, "evergreen" traffic from viral, high-volume traffic spikes when seasonal pop-culture engagement peaks.
* **Device Segmentation:** Isolated and calculated the percentage share of mobile traffic versus desktop traffic to evaluate mobile-responsiveness needs during live event windows.

---

### Analytical Deep Dive & Behavioral Insights

#### 1. Temporal Traffic Anomaly Modeling
Using time-series visualizations built in `Plotly Express`, the data maps an incredibly aggressive, cyclical demand curve.
Traffic Volume

▲
│              ▲ [Awards Night Spike: Millions of Users]
│             ╱ ╲
│            ╱   ╲
│           ╱     ╲
│          ╱       ╲     ▲ [Awards Week Build-up]
│╱         ╲╱ ╲____ [Baseline Off-Season Traffic]
└──────────────────────────────────────────────────────────► Timeline

* **The Trend:** While baseline traffic remains stable and relatively low throughout the year, it experiences an explosive multi-million user spike during `awards_week`, culminating in a massive traffic peak on `awards_night`. 
* **Strategic Takeaway:** This confirms that the consumer-facing platform (`grammy.com`) functions primarily as an event-driven media engine. It requires high scalability and low latency during a highly restricted operational window, rather than steady-state audience acquisition.

#### 2. The Post-Split Segmentation Narrative
By separating consumer traffic from industry professionals, the data reveals a clear divergence in user intent:
* **`grammy.com` (The Consumer Hub):** Characterized by immense visitor volume, shorter average session durations, and highly volatile bounce rates that scale alongside viral pop-culture moments. Users are looking for quick information—winners, fashion, and performance clips.
* **`recordingacademy.com` (The B2B Portal):** Characterized by lower, highly predictable visitor volume but significantly stronger engagement depth. Users exhibit higher *Pages per Session* and longer *Average Session Durations*, signaling an industry audience interacting with dense content (e.g., voting guidelines, membership criteria, grant applications).

---

### Competitor Benchmarking & Core Performance Matrix

Based on behavioral aggregates, the platform's digital footprint stands out in specific areas while presenting distinct optimization opportunities:

| Performance Area | Strengths Identified | Key Metric to Improve | Strategic KPI Focus |
| :--- | :--- | :--- | :--- |
| **Peak Traffic Delivery** | Exceptional volume capture during live broadcast windows. | High transactional bounce rates during peak event hours. | **Bounce Rate Mitigation** |
| **Mobile Experience** | Dominant mobile visitor share, proving strong cross-device accessibility. | Session duration drop-off on complex pages. | **Mobile UX Streamlining** |
| **Platform Segmentation** | Successfully isolated high-intent B2B users from general consumers. | Conversion of consumer traffic into year-round engagement. | **Evergreen Funnel Conversion** |

---

### Strategic Consulting Recommendations

To fully leverage the dual-website architecture, the following data-driven strategies are proposed:

1. **Infrastructure Scalability & Caching for Peak Windows:** Because consumer traffic is heavily consolidated into a single week, prioritize server bandwidth allocation and page-weight optimization for `grammy.com` during Awards Week. Reducing asset sizes and leveraging strict content delivery network (CDN) caching will combat latency and minimize the bounce rates caused by slow loading speeds on mobile networks during the broadcast.
2. **High-Value Funnel Integration for B2B Growth:** Use the massive consumer traffic spikes on `grammy.com` as an acquisition funnel. Implement subtle, high-converting call-to-actions (CTAs) directing eligible independent creators and industry professionals to the membership portals on `recordingacademy.com`.
3. **Evergreen Content Calendars for Consumer Retention:** To soften the post-awards traffic drop-off on `grammy.com`, deploy a targeted, year-round multimedia content strategy (e.g., "Behind the Record" deep dives, acoustic sessions, and genre-specific editorial pieces) to convert anonymous event-night viewers into consistent monthly active users (MAUs).

---
















