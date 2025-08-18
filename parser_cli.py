# parser_cli.py
import argparse
import csv
import os
import sys
from typing import Dict, List

# Proje modülleri
from extractor import parse_address
from resolver import LocationResolver


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
                             score_threshold: float = 1.0) -> Dict[str, str]:
    """Boş il/ilçe varsa co-occurrence tabanlı resolver ile doldur."""
    need_il = not (parsed.get("il") or "").strip()
    need_ilce = not (parsed.get("ilce") or "").strip()
    if not (need_il or need_ilce):
        return parsed

    il_hint = (parsed.get("il") or None)
    ilce_hint = (parsed.get("ilce") or None)

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
        # Sadece il/ilçe’yi gerçekten çıkarmışsak gözlem ekler (observe içinde filtre var)
        resolver.observe(parsed)
        if i % 200000 == 0:
            print(f"[resolver] observed: {i:,} rows...")

    os.makedirs(os.path.dirname(kb_path), exist_ok=True)
    resolver.save(kb_path)
    print(f"[resolver] saved index -> {kb_path}")


def write_output_csv(out_path: str, rows: List[Dict[str, str]]) -> None:
    # Sabit alan sırası; id/label varsa en başa alınır
    fixed_fields = [
        "id", "address", "label",
        "normalized", "il", "ilce",
        "mahalle", "sokak", "cadde", "bulvar",
        "no", "kat", "daire", "blok", "site", "apartman"
    ]
    # Gerçekte var olan alanlara göre filtrele
    # (id/label inputta yoksa yine başlıkta kalabilir ama boş yazarız)
    fieldnames = fixed_fields

    os.makedirs(os.path.dirname(out_path), exist_ok=True) if os.path.dirname(out_path) else None
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
    args = parser.parse_args()

    # 1) İstenirse index oluştur
    if args.build_from:
        build_index_from_csv(args.build_from, args.kb_path)

    # 2) Resolver'ı yükle (boş da olabilir)
    resolver = LocationResolver.load(args.kb_path)

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

        # Boş il/ilçe’yi co-occurrence ile doldur
        parsed = apply_resolver_if_needed(parsed, resolver, score_threshold=args.resolver_threshold)

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
            # Ne dry-run ne output verilmişse kullanıcıya hatırlatalım
            print(f"[info] {len(out_rows):,} kayıt işlendi. "
                  f"Dosyaya yazmak için --output verin veya önizleme için --dry-run kullanın.")


if __name__ == "__main__":
    main()
