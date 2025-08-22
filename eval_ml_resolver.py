# eval_ml_resolver.py
# -*- coding: utf-8 -*-
import argparse, csv, os, sys
from typing import Dict, List
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, f1_score

from normalizer import normalize
from extractor import parse_address
from resolver import LocationResolver

def pick_address_field(row: Dict[str,str]) -> str:
    for k in ("address","Address","adres"):
        if k in row and row[k]:
            return row[k]
    return ""

def truth_label_from_row(row: Dict[str,str]) -> str:
    il = (row.get("il") or "").strip().title()
    ilce = (row.get("ilce") or "").strip().title()
    if il and ilce:
        return f"{il}|{ilce}"
    t = (row.get("target") or row.get("label_true") or "").strip()
    return t

def read_csv(path: str) -> List[Dict[str,str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        rdr = csv.DictReader(f)
        return list(rdr)

def topk_acc(y_true: List[str], topk_preds: List[List[str]]) -> float:
    ok = 0; n = 0
    for yt, cand in zip(y_true, topk_preds):
        if not yt:
            continue
        n += 1
        if yt in cand:
            ok += 1
    return (ok / n) if n else float("nan")

def main():
    ap = argparse.ArgumentParser("ML il|ilçe değerlendirme")
    ap.add_argument("--input", required=True, help="test.csv")
    ap.add_argument("--model", default="cache/ml_resolver.joblib", help="joblib pipeline")
    ap.add_argument("--kb", help="resolver index json (hibrit için gerekli)", default=None)
    ap.add_argument("--hybrid", action="store_true", help="ML + resolver fallback değerlendir")
    ap.add_argument("--threshold", type=float, default=0.6, help="Hibritte ML güven eşiği")
    ap.add_argument("--output", help="tahmin çıktı CSV yolu (opsiyonel)")
    ap.add_argument("--batch", type=int, default=1000, help="tahmin batch boyutu (küçük tut)")
    args = ap.parse_args()

    rows = read_csv(args.input)
    if not rows:
        print("Boş veri.", file=sys.stderr); return

    print(f"[info] test rows: {len(rows):,}")

    pipe = joblib.load(args.model)
    classes = np.array(pipe.classes_)

    resolver = None
    if args.hybrid:
        if not args.kb:
            print("[warn] --hybrid için --kb verilmeli; hibrit kapatılıyor.", file=sys.stderr)
            args.hybrid = False
        else:
            resolver = LocationResolver.load(args.kb)

    texts_norm: List[str] = []
    parsed_list: List[Dict[str,str]] = []
    y_true: List[str] = []
    orig_addr: List[str] = []
    ids: List[str] = []

    for r in rows:
        addr = pick_address_field(r)
        orig_addr.append(addr)
        ids.append(r.get("id",""))
        y_true.append(truth_label_from_row(r))
        texts_norm.append(normalize(addr))
        parsed_list.append(parse_address(addr))

    # STREAMING tahmin: büyük matrisleri hiçbir yerde tutmuyoruz
    pmax_list: List[float] = []
    top1_labels_ml: List[str] = []
    top3_labels_ml: List[List[str]] = []

    for i in range(0, len(texts_norm), args.batch):
        batch = texts_norm[i:i+args.batch]
        P = pipe.predict_proba(batch)  # (b, C) dense; ama b küçük
        # p_max
        pmax_list.extend(P.max(axis=1).tolist())
        # top-3 (bellek güvenli; satır satır işleyelim)
        top_idx = np.argsort(-P, axis=1)[:, :3]  # ilk 3
        top1 = classes[top_idx[:, 0]]
        top1_labels_ml.extend(list(top1))
        for rix in range(P.shape[0]):
            top3_labels_ml.append(list(classes[top_idx[rix, :]]))
        # batch P'yi bırak
        del P

    # Hibrit top-1
    top1_labels_hybrid: List[str] = []
    if args.hybrid:
        for plabel, pscore, parsed in zip(top1_labels_ml, pmax_list, parsed_list):
            if pscore >= args.threshold:
                top1_labels_hybrid.append(plabel)
            else:
                il, ilce, score = resolver.infer(
                    mahalle=parsed.get("mahalle"),
                    sokak=parsed.get("sokak"),
                    cadde=parsed.get("cadde"),
                    site=parsed.get("site"),
                    apartman=parsed.get("apartman"),
                )
                if il or ilce:
                    top1_labels_hybrid.append(f"{il}|{ilce}")
                else:
                    top1_labels_hybrid.append(plabel)

    # Değerlendirme
    y_true_eval = [yt for yt in y_true if yt]
    idx_eval = [i for i, yt in enumerate(y_true) if yt]

    y_pred_ml_eval = [top1_labels_ml[i] for i in idx_eval]
    y_pred_top3_ml_eval = [top3_labels_ml[i] for i in idx_eval]

    acc_ml = accuracy_score(y_true_eval, y_pred_ml_eval) if y_true_eval else float("nan")
    f1_ml = f1_score(y_true_eval, y_pred_ml_eval, average="macro", zero_division=0) if y_true_eval else float("nan")
    acc3_ml = topk_acc(y_true_eval, y_pred_top3_ml_eval)

    print("[scores] Pure-ML")
    print(f"  top1-accuracy = {acc_ml:.4f}")
    print(f"  top3-accuracy = {acc3_ml:.4f}")
    print(f"  macro-F1      = {f1_ml:.4f}")

    if args.hybrid:
        y_pred_hybrid_eval = [top1_labels_hybrid[i] for i in idx_eval]
        acc_h = accuracy_score(y_true_eval, y_pred_hybrid_eval) if y_true_eval else float("nan")
        f1_h = f1_score(y_true_eval, y_pred_hybrid_eval, average="macro", zero_division=0) if y_true_eval else float("nan")
        print("[scores] Hybrid (ML + resolver)")
        print(f"  top1-accuracy = {acc_h:.4f}")
        print(f"  macro-F1      = {f1_h:.4f}")

    if args.output:
        fieldnames = ["id","address","y_true","y_pred_ml","p_max","top3_ml"]
        if args.hybrid:
            fieldnames += ["y_pred_hybrid"]
        with open(args.output, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for i in range(len(rows)):
                rowo = {
                    "id": ids[i],
                    "address": orig_addr[i],
                    "y_true": y_true[i],
                    "y_pred_ml": top1_labels_ml[i],
                    "p_max": f"{pmax_list[i]:.4f}",
                    "top3_ml": " | ".join(top3_labels_ml[i]),
                }
                if args.hybrid:
                    rowo["y_pred_hybrid"] = top1_labels_hybrid[i]
                w.writerow(rowo)
        print(f"[output] wrote predictions -> {args.output}")

if __name__ == "__main__":
    main()
