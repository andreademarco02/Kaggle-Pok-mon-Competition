#TUNING
import math
import random

def sample_xgb_params():
    return {
        "max_depth": random.randint(4, 9),
        "min_child_weight": random.randint(1, 6),
        "subsample": random.uniform(0.6, 0.95),
        "colsample_bytree": random.uniform(0.6, 0.95),
        "gamma": random.uniform(0.0, 4.0),
        "learning_rate": 10 ** random.uniform(math.log10(0.01), math.log10(0.15)),
        "n_estimators": random.randint(800, 1800),
        "reg_lambda": random.uniform(0.5, 3.0),
        "reg_alpha": random.uniform(0.0, 1.0),
    }

N_TRIALS = 30  

best_acc = -1.0
best_params = None
results = []


for t in range(1, N_TRIALS+1):
    print(f"\n===== TRIAL {t}/{N_TRIALS} =====")
    params = sample_xgb_params()
    print("Params:", params)

    cv_res = cv_fold_safe_wr(
        train_data,
        n_splits=5,
        seed=42,
        model_kind="xgb",
        xgb_params=params
    )

    acc_mean = cv_res["acc_mean"]
    results.append((acc_mean, params))

    if acc_mean > best_acc:
        best_acc = acc_mean
        best_params = params
        print(f"*** New best acc_mean={best_acc:.4f} ***")

print("\nTuning finished")
print("Best acc_mean:", best_acc)
print("Best params:", best_params)
