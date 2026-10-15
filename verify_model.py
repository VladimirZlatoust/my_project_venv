# -*- coding: utf-8 -*-
"""Точное воспроизведение пайплайна model.ipynb (без ydata-profiling и графиков)
для верификации метрик, используемых в Model Card."""

import json
import re
import sys
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

WORKDIR = r"D:\my_project\my_project_venv"
OUT = r"C:\Users\Владимир\AppData\Local\Temp\opencode\verify_metrics.json"
sys.path.insert(0, WORKDIR)


def load_fixed_csv(filename):
    records = []
    bad_lines = []
    with open(filename, encoding="utf-8-sig") as f:
        next(f)
        for i, raw in enumerate(f, start=2):
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            pos_comma = line.rfind(",")
            pos_semicolon = line.rfind(";")
            pos = max(pos_comma, pos_semicolon)
            if pos == -1:
                bad_lines.append((i, line))
                continue
            text = line[:pos].strip().strip('"')
            label_str = line[pos + 1:].strip().strip('"')
            try:
                label = int(label_str)
            except ValueError:
                bad_lines.append((i, line))
                continue
            records.append((text, label))
    return pd.DataFrame(records, columns=["text", "is_spam1_or_clickbait2"]), bad_lines


URL_RE = re.compile(r"https?://\S+|www\.\S+|<link>")
HTML_RE = re.compile(r"<[^>]+>")
MENTION_RE = re.compile(r"[@#]\w+")
DIGIT_RE = re.compile(r"\d+([.,]\d+)?")
SPACE_RE = re.compile(r"\s+")


def clean_text(text: str, fix_yo: bool = True) -> str:
    text = str(text)
    text = HTML_RE.sub(" ", text)
    text = URL_RE.sub(" URL ", text)
    text = MENTION_RE.sub(" ", text)
    text = text.lower()
    if fix_yo:
        text = text.replace("ё", "е")
    text = DIGIT_RE.sub(" NUM ", text)
    text = re.sub(r"[^a-zA-Zа-яА-Я\s]", " ", text)
    return SPACE_RE.sub(" ", text).strip()


res = {}
t0 = time.perf_counter()
df, bad_lines = load_fixed_csv(r"D:\my_project\my_project_venv\df.csv")
res["raw_rows"] = int(df.shape[0])
res["bad_lines"] = len(bad_lines)
res["raw_class_counts"] = {
    int(k): int(v) for k, v in df["is_spam1_or_clickbait2"].value_counts().sort_index().items()
}
res["nulls"] = int(df.isnull().sum().sum())
res["dupes_raw"] = int(df.duplicated().sum())

init = df.shape
df = df.drop_duplicates()
res["after_dedup"] = int(df.shape[0])
res["removed"] = int(init[0] - df.shape[0])
res["dedup_class_counts"] = {
    int(k): int(v) for k, v in df["is_spam1_or_clickbait2"].value_counts().sort_index().items()
}

# длина текста
lens = df["text"].astype(str).str.len()
res["len_mean"] = float(lens.mean())
res["len_max"] = int(lens.max())
res["len_min"] = int(lens.min())
res["len_median"] = float(lens.median())

y = df["is_spam1_or_clickbait2"].astype(int).values
X = df["text"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
res["n_train"] = int(X_train.shape[0])
res["n_test"] = int(X_test.shape[0])
res["train_class_counts"] = {int(k): int(v) for k, v in pd.Series(y_train).value_counts().sort_index().items()}

X_train_clean = X_train.apply(clean_text)
X_test_clean = X_test.apply(clean_text)
res["empty_after_clean_train"] = int((X_train_clean.str.len() == 0).sum())
res["empty_after_clean_test"] = int((X_test_clean.str.len() == 0).sum())
res["clean_time_sec"] = round(time.perf_counter() - t0, 1)

cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# ---------------- Модель 1 (baseline) ----------------
model1 = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=50000, sublinear_tf=True)),
    ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0, random_state=42, solver="lbfgs")),
])
model1.fit(X_train_clean, y_train)
pred1 = model1.predict(X_test_clean)
proba1 = model1.predict_proba(X_test_clean)

# ---------------- Модель 2 (champion) ----------------
advanced_vec = FeatureUnion([
    ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=30000, sublinear_tf=True)),
    ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=30000)),
])
model2_base = Pipeline([
    ("tfidf", advanced_vec),
    ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
])
param_grid = {"clf__C": [0.1, 0.5, 1.0, 5.0, 7.0, 10.0, 50.0, 100.0]}
grid = GridSearchCV(model2_base, param_grid, cv=cv_strategy, scoring="f1_weighted", n_jobs=-1, verbose=1)
grid.fit(X_train_clean, y_train)
best_model2 = grid.best_estimator_
pred2 = best_model2.predict(X_test_clean)
proba2 = best_model2.predict_proba(X_test_clean)
res["best_C"] = grid.best_params_["clf__C"]
res["grid_mean_test_f1w"] = {str(k): float(v) for k, v in zip(param_grid["clf__C"], grid.cv_results_["mean_test_score"])}
res["n_features_word_char"] = int(len(best_model2.named_steps["tfidf"].get_feature_names_out()))

target_names = ["Обычные", "Спам", "Кликбейт"]


def metrics(y_true, y_pred, y_proba):
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "roc_auc_ovr_weighted": float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")),
        "roc_auc_ovr_macro": float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")),
        "report": classification_report(y_true, y_pred, target_names=target_names, digits=4),
        "confusion": confusion_matrix(y_true, y_pred).tolist(),
        "support": {target_names[i]: int((y_true == i).sum()) for i in range(3)},
    }
    return out


res["model1"] = metrics(y_test, pred1, proba1)
res["model2"] = metrics(y_test, pred2, proba2)

# доля серой зоны при max P >= 0.60 (и >=0.80)
for name, proba, pred in [("model1", proba1, pred1), ("model2", proba2, pred2)]:
    maxp = proba.max(axis=1)
    res[name]["share_maxp_ge_060"] = float((maxp >= 0.60).mean())
    res[name]["share_maxp_ge_080"] = float((maxp >= 0.80).mean())
    res[name]["share_maxp_lt_060"] = float((maxp < 0.60).mean())
    # доля кликбейт-решений в диапазоне 0.60 <= P2 < 0.80
    p2 = proba[:, list(best_model2.named_steps["clf"].classes_).index(2)]
    res[name]["share_p2_band"] = float(((p2 >= 0.60) & (p2 < 0.80)).mean())

# ошибка на обычных (FP-rate по классу 0) и доля легальных в спам-карантине
cm1 = np.array(res["model1"]["confusion"])
cm2 = np.array(res["model2"]["confusion"])
res["model2"]["err_on_normal_fnr"] = float(cm2[0, 1] + cm2[0, 2]) / float(cm2[0].sum())
res["model1"]["err_on_normal_fnr"] = float(cm1[0, 1] + cm1[0, 2]) / float(cm1[0].sum())
res["model2"]["normal_in_spam_quarantine"] = float(cm2[0, 1]) / float(cm2[:, 1].sum())
res["model1"]["normal_in_spam_quarantine"] = float(cm1[0, 1]) / float(cm1[:, 1].sum())

# топ-20 признаков для кликбейта (модель 2)
vec = best_model2.named_steps["tfidf"]
clf = best_model2.named_steps["clf"]
fn = np.array(vec.get_feature_names_out())
cc = clf.coef_[list(clf.classes_).index(2)]
pos = cc.argsort()[-20:][::-1]
neg = cc.argsort()[:20]
res["top_clickbait_pos"] = [[str(fn[i]), float(cc[i])] for i in pos]
res["top_clickbait_neg"] = [[str(fn[i]), float(cc[i])] for i in neg]

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)
print("WROTE", OUT)
print("best_C", res["best_C"])
print("M1 report:\n", res["model1"]["report"])
print("M2 report:\n", res["model2"]["report"])
print("M2 CM:", res["model2"]["confusion"])
