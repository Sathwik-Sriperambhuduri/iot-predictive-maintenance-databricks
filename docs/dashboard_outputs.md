# Dashboard Outputs

## Dashboard Overview

The AI-style predictive maintenance dashboard was created using inference outputs from the registered Champion models.

The dashboard helps maintenance teams review engine health, identify high-risk engines, and prioritize maintenance decisions.

## KPI Summary

| Metric | Value |
|---|---:|
| Total Engines Monitored | 100 |
| High Risk Engines | 18 |
| Medium Risk Engines | 15 |
| Low Risk Engines | 67 |
| Average Predicted RUL | 87.17 |
| Lowest Predicted RUL | 7.52 |
| Highest Predicted RUL | 191.23 |

## Risk Distribution

| Risk Category | Count | Percentage |
|---|---:|---:|
| High Risk | 18 | 18% |
| Medium Risk | 15 | 15% |
| Low Risk | 67 | 67% |

## Maintenance Priority Queue

The maintenance priority queue ranks engines by urgency using:

- Risk category
- Predicted Remaining Useful Life
- Failure-risk probability

Engines with lower predicted RUL and higher failure-risk probability are ranked higher for maintenance review.

## Single Engine AI Lookup

The dashboard includes a single-engine lookup feature.

For Engine 35:

| Field | Value |
|---|---|
| Latest Cycle | 198 |
| Predicted RUL | 8.76 cycles |
| Failure Risk Probability | 99.59% |
| Current Risk Category | High Risk |
| First Medium Risk Cycle | 133 |
| First High Risk Cycle | 161 |
| Recommended Action | Immediate maintenance inspection recommended |

## AI Summary

Engine 35 was classified as High Risk at the latest cycle. The model predicted only 8.76 cycles of remaining useful life with a 99.59% failure-risk probability.

The dashboard recommended immediate maintenance inspection.
