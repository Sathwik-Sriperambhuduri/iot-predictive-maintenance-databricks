# Model Results

## RUL Regression Model

The selected regression model was a tuned Random Forest Regressor.

This model predicts the Remaining Useful Life of each engine using cycle age, selected sensor values, rolling averages, and sensor difference features.

| Metric | Value |
|---|---:|
| RMSE | 36.40 |
| MAE | 26.23 |
| R² | 0.7215 |

## Regression Interpretation

The regression model achieved an R² score of 0.7215, meaning it was able to explain a strong portion of the variation in Remaining Useful Life.

The RMSE value of 36.40 means the model’s prediction error is around 36 cycles on average in squared-error terms, while the MAE value of 26.23 gives a more direct average absolute error.

## Failure-Risk Classification Model

The selected classification model was Logistic Regression.

This model classifies whether an engine is at high risk of failure based on engineered sensor features.

| Metric | Value |
|---|---:|
| Accuracy | 0.9497 |
| Precision | 0.9492 |
| Recall | 0.9497 |
| F1 Score | 0.9494 |
| AUC | 0.9838 |

## Classification Interpretation

The classification model performed strongly, with an accuracy of 0.9497 and an AUC of 0.9838.

The high recall and F1 score show that the model was effective at identifying failure-risk patterns while maintaining balanced classification performance.

## Selected Models

Final selected models:

- RUL Prediction: Tuned Random Forest Regressor
- Failure-Risk Prediction: Logistic Regression

Both models were tracked using MLflow and registered in Unity Catalog with Champion aliases.
