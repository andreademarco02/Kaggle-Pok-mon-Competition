import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score
)
from features import assemble_features_from_battles, build_wr_map_from_battles
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score

def cv_fold_safe_wr(
    train_data,
@@ -76,48 +71,49 @@
                reg_alpha=0.0,
                eval_metric="logloss",
                tree_method="hist",
                device="cuda",
                random_state=seed,
            )
            if xgb_params is not None:
                base_params.update(xgb_params)

            base_params["random_state"] = seed

            model = XGBClassifier(**base_params)

            if use_early_stopping:
                model.fit(
                    X_tr,
                    y_tr,
                    eval_set=[(X_va, y_va)],
                    early_stopping_rounds=early_stopping_rounds,
                    verbose=False,
                )
            else:
                model.fit(X_tr, y_tr)

        pred  = model.predict(X_va)
        proba = model.predict_proba(X_va)[:,1] if hasattr(model, "predict_proba") else None

        accs.append(accuracy_score(y_va, pred))
        f1s.append(f1_score(y_va, pred))
        pres.append(precision_score(y_va, pred, zero_division=0))
        recs.append(recall_score(y_va, pred, zero_division=0))
        if proba is not None:
            aucs.append(roc_auc_score(y_va, proba))

        print(f"[Fold {fold}] acc={accs[-1]:.4f} f1={f1s[-1]:.4f}")

    out = {
        "acc_mean": np.mean(accs), "acc_std": np.std(accs),
        "auc_mean": np.mean(aucs) if aucs else np.nan, "auc_std": np.std(aucs) if aucs else np.nan,
        "f1_mean": np.mean(f1s), "f1_std": np.std(f1s),
        "prec_mean": np.mean(pres), "prec_std": np.std(pres),
        "rec_mean": np.mean(recs), "rec_std": np.std(recs),
    }
    print("\nCV summary:")
    for k,v in out.items():
        try:
            print(f"  {k}: {v:.4f}")
        except TypeError:
            print(f"  {k}: {v}")
