# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import argparse, joblib, os
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier

# Bizim modüller
from extractor import parse_address
from resolver import LocationResolver
from normalizer import normalize_text

def enrich_text(addr: str, resolver: LocationResolver=None) -> str:
    p = parse_address(addr)
    # resolver ile il/ilçe doldurmayı dene (opsiyonel)
    if resolver:
        need_il  = not p.get("il")
        need_ilc = not p.get("ilce")
        if need_il or need_ilc:
            il_res, ilce_res, _ = resolver.infer(
                mahalle=p.get("mahalle"),
                sokak=p.get("sokak"),
                cadde=p.get("cadde"),
                site=p.get("site"),
                apartman=p.get("apartman"),
            )
            if need_il and il_res:  p["il"]   = il_res
            if need_ilc and ilce_res: p["ilce"] = ilce_res

    # Alanları tek metinde birleştir (feature text)
    parts = [
        normalize_text(addr),
        f"__il__ {p.get('il','')}",
        f"__ilce__ {p.get('ilce','')}",
        f"__mah__ {p.get('mahalle','')}",
        f"__cadde__ {p.get('cadde','')}",
        f"__sokak__ {p.get('sokak','')}",
        f"__bulvar__ {p.get('bulvar','')}",
        f"__no__ {p.get('no','')}",
        f"__kat__ {p.get('kat','')}",
        f"__daire__ {p.get('daire','')}",
        f"__blok__ {p.get('blok','')}",
        f"__site__ {p.get('site','')}",
        f"__apt__ {p.get('apartman','')}",
    ]
    return " ".join([x for x in parts if x and x.strip()])

def stream_rows(csv_path, chunksize=50000):
    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        # bazı datasetlerde address kolonu farklı adlandırılmış olabilir
        if "address" not in chunk.columns:
            if "Address" in chunk.columns: chunk["address"] = chunk["Address"]
            elif "adres" in chunk.columns: chunk["address"] = chunk["adres"]
            else: chunk["address"] = ""
        yield chunk

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="cache/label_clf.joblib")
    ap.add_argument("--kb", default="cache/gazetteer_index.json")
    ap.add_argument("--min-samples", type=int, default=10,
                    help="Sınıfta en az kaç örnek olsun (nadiren görülenleri filtrele)")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--chunksize", type=int, default=50000)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True) if os.path.dirname(args.output) else None

    # Resolver opsiyonel
    resolver = None
    if os.path.exists(args.kb):
        resolver = LocationResolver.load(args.kb)

    # LabelEncoder'ı kurmak için tüm label'ları birinci geçişte topla
    labels_all = []
    for df in stream_rows(args.input, chunksize=args.chunksize):
        if "label" in df.columns:
            labels_all.extend(df["label"].astype(str).tolist())

    if not labels_all:
        print("[warn] Girdi train.csv değil gibi (label yok). Eğitim yapılamaz.")
        return

    # nadir sınıfları filtrelemek için say
    vc = pd.Series(labels_all).value_counts()
    keep = set(vc[vc >= args.min_samples].index)
    print(f"[info] classes total: {vc.size}, kept: {len(keep)} (min_samples={args.min_samples})")

    le = LabelEncoder()
    le.fit(list(keep))
    n_classes = len(le.classes_)
    print(f"[info] n_classes for training: {n_classes}")

    # HashingVectorizer (stateless, RAM dostu)
    vect = HashingVectorizer(
        n_features=2**20,
        alternate_sign=False,
        analyzer="char",
        ngram_range=(3,5),
        norm="l2"
    )

    clf = SGDClassifier(
        loss="hinge",      # SVM benzeri; prob gerekmiyor
        alpha=1e-5,
        max_iter=5,
        tol=1e-3
    )

    # İlk partial_fit için sınıfları vermemiz gerekir
    clf_initialized = False

    for ep in range(args.epochs):
        print(f"[train] epoch {ep+1}/{args.epochs}")
        for df in stream_rows(args.input, chunksize=args.chunksize):
            df = df[ df["label"].astype(str).isin(keep) ]
            if df.empty: 
                continue
            # zenginleştirilmiş metin
            texts = [enrich_text(a, resolver) for a in df["address"].astype(str).tolist()]
            y = le.transform(df["label"].astype(str).tolist())

            X = vect.transform(texts)
            if not clf_initialized:
                clf.partial_fit(X, y, classes=np.arange(n_classes))
                clf_initialized = True
            else:
                clf.partial_fit(X, y)

    joblib.dump({"vectorizer":"hashing", "clf":clf, "label_encoder":le}, args.output)
    print(f"[ok] saved -> {args.output}")

if __name__ == "__main__":
    main()
