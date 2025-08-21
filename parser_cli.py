# parser_cli.py
# -*- coding: utf-8 -*-
import argparse
import csv
import os
import sys
from typing import Dict, List

# Proje modülleri
from extractor import parse_address
from resolver import LocationResolver

# ML fallback opsiyonel
try:
    from ml_resolver import MLResolver  # train_ml_resolver.py ile eğitilen modelin yükleyicisi
except Exception:
    MLResolver = None  # type: ignore


def read_csv_rows(path: str):
    with open(path, "r", encoding="utf-8", newline="") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            yield row


def pick_address_field(row: Dict[str, str]) -> str:
    for key in ("address", "Address", "adres"):
        if key in row and row[key]:
            return row[key]
    return ""


def apply_resolver_if_needed(parsed: Dict[str, str],
                             resolver: LocationResolver,
                             score_threshold: float = 1.0,
                             ml_resolver=None,
                             ml_threshold: float = 0.55) -> Dict[str, str]:
    """
    Boş il/ilçe varsa:
      1) Co-occurrence (gazetteer) ile doldur.
      2) Hâlâ eksikse ve ML modeli yüklüyse, ML fallback ile tamamla.
    """
    need_il = not (parsed.get("il") or "").strip()
    need_ilce = not (parsed.get("ilce") or "").strip()
    if not (need_il or need_ilce):
        return parsed

    il_hint = (parsed.get("il") or None)
    ilce_hint = (parsed.get("ilce") or None)

    # --- 1) Gazetteer (co-occurrence) ---
    il_res, ilce_res, score = resolver.infer(
        mahalle=parsed.get("mahalle"),
        sokak=parsed.get("sokak"),
        cadde=parsed.get("cadde"),
        site=parsed.get("site"),
        apartman=parsed.get("apartman"),
        il_hint=il_hint,
        ilce_hint=ilce_hint,
    )

    if score >= score_threshold:
        if need_il and il_res:
            parsed["il"] = il_res
        if need_ilce and ilce_res:
            parsed["ilce"] = ilce_res

    # Durum güncelle
    need_il = not (parsed.get("il") or "").strip()
    need_ilce = not (parsed.get("ilce") or "").strip()

    # --- 2) ML fallback (opsiyonel) ---
    if (need_il or need_ilce) and ml_resolver is not None:
        try:
            il_m, ilce_m, p = ml_resolver.infer(parsed)
            if p >= ml_threshold:
                if need_il and il_m:
                    parsed["il"] = il_m
                if need_ilce and ilce_m:
                    parsed["ilce"] = ilce_m
        except Exception as e:
            print(f"[ml] WARN: tahmin sırasında hata: {e}", file=sys.stderr)

    return parsed


def build_index_from_csv(csv_path: str, kb_path: str) -> None:
    """CSV'yi tarayıp resolver index'ini oluşturur ve kaydeder."""
    print(f"[resolver] building index from: {csv_path}")
    resolver = LocationResolver()

    for i, row in enumerate(read_csv_rows(csv_path), start=1):
        addr = pick_address_field(row)
        if not addr:
            continue
        parsed = parse_address(addr)
        # Sadece il/ilçe anlamlı ise observe ekler (resolver.observe içinde filtre var)
        resolver.observe(parsed)
        if i % 200000 == 0:
            print(f"[resolver] observed: {i:,} rows...")

    os.makedirs(os.path.dirname(kb_path), exist_ok=True)
    resolver.save(kb_path)
    print(f"[resolver] saved index -> {kb_path}")


def write_output_csv(out_path: str, rows: List[Dict[str, str]]) -> None:
    fixed_fields = [
        "id", "address", "label",
        "normalized", "il", "ilce",
        "mahalle", "sokak", "cadde", "bulvar",
        "no", "kat", "daire", "blok", "site", "apartman"
    ]
    fieldnames = fixed_fields

    if os.path.dirname(out_path):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            out = {k: r.get(k, "") for k in fieldnames}
            w.writerow(out)
    print(f"[output] wrote {len(rows):,} rows -> {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Hepsiburada Hackathon - Address Matching/Resolution CLI"
    )
    parser.add_argument("--input", required=False, help="Girdi CSV yolu (id,address[,label])")
    parser.add_argument("--output", required=False, help="Çıktı CSV yolu")
    parser.add_argument("--dry-run", type=int, default=0,
                        help="İlk N kaydı sadece ekrana yaz (çıktı dosyası oluşturmaz)")
    parser.add_argument("--kb", "--knowledge-cache", dest="kb_path",
                        default="cache/gazetteer_index.json",
                        help="Resolver index dosyası (varsayılan: cache/gazetteer_index.json)")
    parser.add_argument("--build-index-from", dest="build_from", default=None,
                        help="Verilen CSV'den resolver index'i oluştur ve kaydet")
    parser.add_argument("--resolver-threshold", type=float, default=1.0,
                        help="Resolver skor eşiği (vars: 1.0). Daha düşük ise daha agresif doldurur.")

    # ML fallback opsiyonları
    parser.add_argument("--ml-model", dest="ml_model", default=None,
                        help="ML resolver model yolu (joblib). Örn: cache/ml_resolver.joblib")
    parser.add_argument("--ml-threshold", type=float, default=0.55,
                        help="ML tahmin olasılık eşiği (vars: 0.55)")

    args = parser.parse_args()

    # 1) İstenirse index oluştur
    if args.build_from:
        build_index_from_csv(args.build_from, args.kb_path)

    # 2) Resolver'ı yükle (boş da olabilir)
    resolver = LocationResolver.load(args.kb_path)

    # 2.5) ML model (opsiyonel)
    ml_resolver = None
    if args.ml_model:
        if MLResolver is None:
            print("[ml] WARN: ml_resolver import edilemedi; --ml-model yok sayılacak.", file=sys.stderr)
        else:
            try:
                ml_resolver = MLResolver(args.ml_model)
                print(f"[ml] loaded: {args.ml_model}")
            except Exception as e:
                print(f"[ml] WARN: model yüklenemedi: {e}", file=sys.stderr)

    # 3) Girdi yoksa burada bitir
    if not args.input:
        if not args.build_from:
            print("Hiç bir argüman çalışmadı. --input veya --build-index-from verin.", file=sys.stderr)
        return

    # 4) Girdiyi işle
    out_rows: List[Dict[str, str]] = []
    for i, row in enumerate(read_csv_rows(args.input)):
        addr = pick_address_field(row)
        if not addr:
            continue

        parsed = parse_address(addr)

        # Orijinal id/label’i ekle (varsa)
        if "id" in row:
            parsed["id"] = row["id"]
        if "label" in row:
            parsed["label"] = row["label"]
        # Orijinal metni de ekleyelim
        parsed["address"] = addr

        # İl/ilçe’yi co-occurrence + (opsiyonel) ML fallback ile doldur
        parsed = apply_resolver_if_needed(
            parsed,
            resolver,
            score_threshold=args.resolver_threshold,
            ml_resolver=ml_resolver,
            ml_threshold=args.ml_threshold
        )

        # Dry-run çıktı
        if args.dry_run and i < args.dry_run:
            preview = row.get("address", addr)
            print(f"[{i}] {preview}\n -> {parsed}\n")

        out_rows.append(parsed)

        # Sadece dry-run ise ilk N kaydı gösterip yazmadan çık
        if args.dry_run and (i + 1) >= args.dry_run and not args.output:
            break

    # 5) Çıktı dosyası
    if args.output:
        write_output_csv(args.output, out_rows)
    else:
        if not args.dry_run:
            print(f"[info] {len(out_rows):,} kayıt işlendi. "
                  f"Dosyaya yazmak için --output verin veya önizleme için --dry-run kullanın.")


if __name__ == "__main__":
    main()
