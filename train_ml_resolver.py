# train_ml_resolver.py
# -*- coding: utf-8 -*-
import csv, argparse, os, joblib
from typing import Dict, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from extractor import parse_address  # mevcut parser'ı kullanıyoruz
from normalizer import normalize_text

def pick_address_field(row: Dict[str,str]) -> str:
    for k in ("address","Address","adres"):
        if k in row and row[k]:
            return row[k]
    return ""

def make_feat(parsed: Dict[str,str]) -> str:
    # Basit ama etkili: alan etiketleriyle birleştirip TF-IDF (char n-gram) besliyoruz
    parts = []
    for f in ("mahalle","cadde","sokak","site","apartman"):
        v = (parsed.get(f) or "").strip()
        if v:
            parts.append(f"{f}={normalize_text(v)}")
    txt = " | ".join(parts)
    return txt if txt else ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="cache/ml_resolver.joblib")
    ap.add_argument("--test-size", type=float, default=0.1)
    ap.add_argument("--min-samples", type=int, default=5, help="Seyrek sınıfları filtrelemek için alt sınır")
    args = ap.parse_args()

    X, y = [], []
    with open(args.input, "r", encoding="utf-8", newline="") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            addr = pick_address_field(row)
            if not addr: 
                continue
            p = parse_address(addr)
            # Etiket: (il|ilçe). İl olmak ZORUNLU, ilçe boş olabilir.
            il = (p.get("il") or "").strip()
            if not il: 
                continue
            ilce = (p.get("ilce") or "").strip()
            label = f"{il}|{ilce}"
            feat = make_feat(p)
            if not feat:
                continue
            X.append(feat); y.append(label)

    # Sınıf sayacı ve seyrek sınıf filtresi
    from collections import Counter
    cnt = Counter(y)
    keep_mask = [cnt[lab] >= args.min_samples for lab in y]
    X = [x for x, m in zip(X, keep_mask) if m]
    y = [lab for lab, m in zip(y, keep_mask) if m]

    if not X:
        print("Yeterli eğitim örneği yok.")
        return

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=args.test_size, random_state=42, stratify=y)

    pipe = Pipeline([
        ("vec", TfidfVectorizer(analyzer="char", ngram_range=(3,5), min_df=3)),
        ("clf", LogisticRegression(max_iter=300, n_jobs=-1, verbose=0, multi_class="auto")),
    ])
    pipe.fit(Xtr, ytr)

    ypred = pipe.predict(Xte)
    acc = accuracy_score(yte, ypred)
    print(f"[eval] pair(top-1) accuracy = {acc:.3f}  (sınıf sayısı: {len(set(y))})")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    joblib.dump(pipe, args.output)
    print(f"[save] {args.output}")

if __name__ == "__main__":
    main()
