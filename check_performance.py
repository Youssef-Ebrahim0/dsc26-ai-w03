from tawseel.data import make_parcels
from tawseel.model import (
    add_history_features,
    time_split,
    compare_to_baseline,
)

for seed in [0, 1, 2, 3]:
    df = make_parcels(4000, seed=seed)
    df = add_history_features(df)

    train, test = time_split(df)

    results = compare_to_baseline(train, test, seed=seed)

    print(f"\nSeed: {seed}")
    print(f"Base rate: {results['model']['base_rate']:.3f}")
    print(f"Model PR-AUC: {results['model']['pr_auc']:.4f}")
    print(f"Dummy PR-AUC: {results['dummy']['pr_auc']:.4f}")
    print(f"PR-AUC Lift: {results['lift_pr_auc']:.4f}")
    print(f"Model Brier: {results['model']['brier']:.4f}")
    print(f"Dummy Brier: {results['dummy']['brier']:.4f}")
    print(f"ROC-AUC: {results['model']['roc_auc']:.4f}")
