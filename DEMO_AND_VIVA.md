# AgriShield AI: 3-minute demo and viva guide

## Three-minute classroom demonstration

**0:00–0:25 — Frame the problem**

“AgriShield AI is an explainable decision-support prototype for Maharashtra agriculture. It uses 10,000 synthetic observations from 2001 to 2025. I separate the machine-learning yield estimate from transparent rule-based climate-risk scores, so the dashboard never pretends the risk rules are learned predictions.”

**0:25–0:55 — Show the dashboard**

Select **Pune**, **Soybean**, and the newest available base year. Point to predicted yield, overall risk, rainfall, and average temperature. Briefly show the historical yield trend and risk breakdown.

“The prediction uses only information that could be available before harvest. Production tonnage is excluded because it is yield multiplied by area and would leak the answer.”

**0:55–1:35 — Run the signature scenario**

Open **What-If Simulator**. Keep Pune and Soybean, then set temperature to **+2 °C** and rainfall to **−20%**. Show baseline yield, scenario yield, percentage change, drought risk, heat risk, and overall risk.

“This is a sensitivity test. It does not claim the climate change is causal or that this exact future will occur.”

**1:35–2:05 — Explain the model**

Open **Explainability**. Show global permutation importance and the local red/green contribution chart.

“The local explanation is calculated after the exact same feature engineering, scaling, and one-hot encoding used by the trained pipeline. Positive bars raise the modeled yield; negative bars lower it.”

**2:05–2:35 — Recommend alternatives**

Open **Crop Recommendation** with the same environment. Point out that ranking uses a crop-specific historical percentile, not raw tonnes per hectare.

“Otherwise sugarcane would dominate because its natural tonnage scale is much larger. The score combines 72% crop-relative modeled performance and 28% climate resilience.”

**2:35–3:00 — Close honestly**

Open **Advisory** or return to the dashboard summary.

“The model reached a synthetic holdout RMSE of 1.564 t/ha and R² of 0.987. Those numbers validate the implementation on this generator, not real-world deployment. The next step is external validation with IMD weather, official crop statistics, soil tests, and satellite features.”

## Ten likely viva questions

### 1. Why did you avoid a random split as the primary evaluation?

A random split mixes older and newer years. A forecasting system is used on future seasons, so I trained on 2001–2020, selected on 2021–2022, and tested once on 2023–2025. The random split is only a diagnostic comparison.

### 2. What was the most serious target-leakage risk?

`production_tonnes` is approximately `yield_tonnes_per_ha × cultivated_area_ha`. Production is known after harvest and algebraically contains the target. Including it would inflate accuracy without providing a usable pre-harvest model.

### 3. Why exclude the supplied climate-risk scores from the yield model?

They are derived outputs that may summarize the same weather fields. I kept prediction and risk analytics conceptually separate, prevented hidden redundancy, and let scenario risk be recalculated transparently from changed inputs.

### 4. Why was XGBoost selected?

It had the lowest 2021–2022 validation RMSE and also the lowest 2023–2025 test RMSE, 1.564 t/ha. Random Forest had a lower filtered MAPE, so I report that tradeoff. Selection was not based on complexity alone.

### 5. Why is MAPE not the main metric?

Many yields are near the dataset floor of 0.08 t/ha. Dividing error by a near-zero target makes MAPE unstable and disproportionately large. I calculate it only above 0.10 t/ha and prioritize MAE/RMSE with R² as a secondary fit measure.

### 6. How did you prevent preprocessing leakage?

Feature engineering, imputation, robust scaling, and one-hot encoding are inside a scikit-learn Pipeline. Each candidate fits those steps only on its training partition. Held-out years are transformed using parameters learned from training.

### 7. Are the climate-risk scores ML predictions?

No. They are bounded rule-based indices using explicit weights and crop-reference thresholds. They support consistent scenario comparison but are not probabilities and have not been calibrated against observed disasters.

### 8. How does SHAP work with one-hot encoding?

The app first applies the trained feature engineer and preprocessor, gets the exact transformed feature names, and then explains the final estimator in that space. It does not attach arbitrary SHAP values to untransformed raw columns.

### 9. Why does crop recommendation use a percentile?

Crop yields have different natural scales. A raw-tonnage ranking would almost always favor sugarcane. The predicted yield is converted to a percentile within that crop's own historical distribution, then combined with climate resilience.

### 10. What prevents this from being deployed today?

The data is synthetic, risk thresholds are not locally calibrated, spatial dependence is not modeled, uncertainty intervals are missing, and management variables are limited. Deployment needs official data, field validation, agronomist review, monitoring, and governance.

## Strong closing line

“The project's contribution is not a claim of field-ready accuracy. It is a transparent, testable architecture that shows how prediction, risk scoring, explanation, scenarios, and recommendations can be integrated without hiding their different levels of evidence.”
