import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
)

from features import assemble_features_from_battles, build_wr_map_from_battles

def cv_fold_safe_wr(
    train_data,
    n_splits=5,
    seed=42,
    model_kind="logreg",
    xgb_params=None,
    use_early_stopping=False,
    early_stopping_rounds=50,
):

    idx = np.arange(len(train_data))
    y = np.array([int(b.get("player_won", 0)) for b in train_data])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    accs, aucs, f1s, pres, recs = [], [], [], [], []

    print(f"Inizio CV {n_splits}-fold per {model_kind.upper()}...")

    for fold, (tr, va) in enumerate(skf.split(idx, y), start=1):
        tr_battles = [train_data[i] for i in tr]
        va_battles = [train_data[i] for i in va]

        wr_map_fold = build_wr_map_from_battles(tr_battles)

        df_tr = assemble_features_from_battles(
            tr_battles,
            wr_map_fold,
            include_target=True,
        )
        df_va = assemble_features_from_battles(
            va_battles,
            wr_map_fold,
            include_target=True,
        )

        feat_cols = [c for c in df_tr.columns if c not in ("battle_id", "player_won")]

        missing_in_va = set(feat_cols) - set(df_va.columns)
        for c in missing_in_va:
            df_va[c] = 0.0

        X_tr = df_tr[feat_cols].values
        y_tr = df_tr["player_won"].astype(int).values

        X_va = df_va[feat_cols].values
        y_va = df_va["player_won"].astype(int).values

        if model_kind == "logreg":
            model = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=20000, random_state=seed)),
                ]
            )
            model.fit(X_tr, y_tr)

        else:
            base_params = dict(
                n_estimators=1200,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.85,
                colsample_bytree=0.85,
                min_child_weight=2,
                gamma=0.0,
                reg_lambda=1.0,
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

        pred = model.predict(X_va)
        proba = model.predict_proba(X_va)[:, 1] if hasattr(model, "predict_proba") else None

        accs.append(accuracy_score(y_va, pred))
        f1s.append(f1_score(y_va, pred))
        pres.append(precision_score(y_va, pred, zero_division=0))
        recs.append(recall_score(y_va, pred, zero_division=0))

        if proba is not None:
            aucs.append(roc_auc_score(y_va, proba))

        print(f"[Fold {fold}] acc={accs[-1]:.4f} f1={f1s[-1]:.4f}")

    out = {
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs)),
        "auc_mean": float(np.mean(aucs)) if aucs else float("nan"),
        "auc_std": float(np.std(aucs)) if aucs else float("nan"),
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)),
        "prec_mean": float(np.mean(pres)),
        "prec_std": float(np.std(pres)),
        "rec_mean": float(np.mean(recs)),
        "rec_std": float(np.std(recs)),
    }

    print("\nCV summary:")
    for k, v in out.items():
        try:
            print(f"  {k}: {v:.4f}")
        except TypeError:
            print(f"  {k}: {v}")

    return out
