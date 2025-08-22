# train_ml_resolver.py  — parse tabanlı eğitim (opsiyonel resolver desteği)
import argparse, csv, joblib, numpy as np
from collections import Counter
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier

from normalizer import normalize
from extractor import parse_address
from resolver import LocationResolver

def rows(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        yield from csv.DictReader(f)

def pick_addr(r):
    return r.get("address") or r.get("Address") or r.get("adres") or ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV (address/adres sütunu olmalı)")
    ap.add_argument("--output", required=True, help="Kaydedilecek model yolu (joblib)")
    ap.add_argument("--kb", default="", help="(Opsiyonel) resolver index dosyası")
    ap.add_argument("--resolver-threshold", type=float, default=1.0,
                    help="Resolver skor eşiği (vars: 1.0)")
    ap.add_argument("--sample", type=int, default=0, help="İsteğe bağlı örnek sınırı")
    args = ap.parse_args()

    # (Opsiyonel) co-occurrence resolver
    resolver = LocationResolver.load(args.kb) if args.kb else None
    use_resolver = resolver is not None

    X, y = [], []
    total, used = 0, 0

    for r in rows(args.input):
        total += 1
        addr = pick_addr(r)
        if not addr:
            continue

        # 1) Adresi ayrıştır
        p = parse_address(addr)

        il   = (p.get("il") or "").strip().title()
        ilce = (p.get("ilce") or "").strip().title()

        # 2) Gerekirse resolver ile eksikleri tamamla (yüksek skor şart)
        if use_resolver and (not il or not ilce):
            il_res, ilce_res, score = resolver.infer(
                mahalle=p.get("mahalle"),
                sokak=p.get("sokak"),
                cadde=p.get("cadde"),
                site=p.get("site"),
                apartman=p.get("apartman"),
                il_hint=il or None,
                ilce_hint=ilce or None
            )
            if score >= args.resolver_threshold:
                il   = il   or il_res
                ilce = ilce or ilce_res

        # 3) Yine de ikisi de yoksa, bu satırı eğitimde kullanma
        if not il or not ilce:
            continue

        X.append(normalize(addr))
        y.append(f"{il}|{ilce}")
        used += 1

        if args.sample and used >= args.sample:
            break

    if not X:
        raise SystemExit(
            "Eğitim için örnek bulunamadı. Nedeni genelde: "
            "CSV'de adreslerden il/ilçe çıkarılamadı. "
            "Çözüm: --kb ile resolver ver veya verisetinde il/ilçe geçen adresleri kullan."
        )

    # Hafif & bellek dostu boru hattı
    pipe = Pipeline([
        ("vec", HashingVectorizer(
            analyzer="char_wb",
            ngram_range=(3,5),
            n_features=2**20,
            alternate_sign=False,
            norm="l2",
            dtype=np.float32
        )),
        ("clf", SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=1e-4,
            max_iter=25,
            n_jobs=-1,
            random_state=42
        )),
    ])

    pipe.fit(X, y)
    joblib.dump(pipe, args.output)

    # Kısa özet
    cls = Counter(y)
    print(f"[ok] saved -> {args.output}")
    print(f"[stats] total_rows={total}  used_for_training={used}  classes={len(cls)}")
    most = cls.most_common(10)
    if most:
        print("[top-classes]")
        for k, c in most:
            print(f"  {k}: {c}")

if __name__ == "__main__":
    main()
