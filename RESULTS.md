# Results: Tawseel failed-delivery baseline

## Baseline table

Last 20% of parcels by time, `make_parcels(4000, seed=0)`.

| Model | ROC AUC | PR-AUC | Brier | Base rate |
|---|---:|---:|---:|---:|
| Dummy (prior) | 0.5000 | 0.2200 | 0.1717 | 0.2200 |
| Rule: No custom rule evaluated | — | — | — | — |
| My model (Logistic Regression) | 0.6492 | 0.3837 | 0.1630 | 0.2200 |

## Three sentences for Sherif

The model estimates the risk of a COD parcel failing using information available when the parcel is created and the customer's previous delivery history. On the seed 0 test set, it achieved a PR-AUC of 0.3837 compared with 0.2200 for the dummy baseline, and a better Brier score of 0.1630 compared with 0.1717. However, it is an experimental baseline trained on synthetic data, so its performance must be validated on real delivery data before using it for operational decisions.