# -*- coding: utf-8 -*-
import argparse, os, joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import HashingVectorizer

from extractor import parse_address
from resolver import LocationResolver
from normalizer import normalize_text

def enrich_text(addr: str, resolver=None) -> str:
    p = parse_address(addr)
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

def main():
    ap = argparse.ArgumentParser(description="Evaluate or predict labels")
    ap.add_argument("--input", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--kb", default="cache/gazetteer_index.json")
    ap.add_argument("--output", default="preds.csv")
    ap.add_argument("--chunksize", type=int, default=50000)
    ap.add_argument("--topk", type=int, default=3)
    args = ap.parse_args()

    data = joblib.load(args.model)
    clf = data["clf"]
    le  = data["label_encoder"]

    resolver = LocationResolver.load(args.kb) if os.path.exists(args.kb) else None

    vect = HashingVectorizer(
        n_features=2**20,
        alternate_sign=False,
        analyzer="char",
        ngram_range=(3,5),
        norm="l2"
    )

    # test dosyası label içeriyorsa skor hesaplarız, yoksa sadece tahmin yazarız
    total, gold, pred, rows_out = 0, [], [], []
    has_label = False

    for chunk in pd.read_csv(args.input, chunksize=args.chunksize):
        if "address" not in chunk.columns:
            if "Address" in chunk.columns: chunk["address"] = chunk["Address"]
            elif "adres" in chunk.columns: chunk["address"] = chunk["adres"]
            else: chunk["address"] = ""

        if "label" in chunk.columns:
            has_label = True

        texts = [enrich_text(a, resolver) for a in chunk["address"].astype(str).tolist()]
        X = vect.transform(texts)

        # skorlar (decision_function) => top-k
        scores = clf.decision_function(X)    # (n, C) veya (n,) binary ise
        if scores.ndim == 1:
            # çok sınıflı olmalı; güvenlik için
            scores = scores[:, None]

        # top-k indeksleri
        topk = np.argsort(-scores, axis=1)[:, :args.topk]
        top1 = topk[:, 0]
        labels_top1 = le.inverse_transform(top1)

        # çıktı satırları
        if "id" in chunk.columns:
            ids = chunk["id"].tolist()
        else:
            ids = list(range(total, total + len(chunk)))

        for i, lid in enumerate(labels_top1):
            rows_out.append({"id": ids[i], "pred_label": lid})

        if has_label:
            gold.extend(chunk["label"].astype(str).tolist())
            pred.extend(labels_top1.tolist())

        total += len(chunk)

    # Skorlar
    if has_label and len(pred) == len(gold) and total > 0:
        acc = accuracy_score(gold, pred)
        f1m = f1_score(gold, pred, average="macro")
        print(f"[scores] top1-accuracy = {acc:.4f}")
        print(f"[scores] macro-F1      = {f1m:.4f}")
    else:
        print("[info] label kolonu yok; sadece tahmin dosyası yazılacak.")

    pd.DataFrame(rows_out).to_csv(args.output, index=False)
    print(f"[output] wrote -> {args.output}")

if __name__ == "__main__":
    main()
