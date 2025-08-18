# -*- coding: utf-8 -*-
import csv, argparse
from dataclasses import asdict
import re

from utils import Parsed
from normalizer import normalize
from extractor import (
    RE_NO, RE_KAT, RE_KAT_K, RE_DAIRE, RE_BLOK, RE_SITE, RE_APT, RE_MAH,
    extract_anchor_phrase, find_il, find_ilce
)
from gazetteer import load_gazetteer, infer_from_components
from extractor import clean_place_name, prune_street_phrase, trim_mahalle_tail

def parse_address(raw: str) -> Parsed:
    norm = normalize(raw)
    out = Parsed(normalized=norm)

    # İl / İlçe (desen tabanlı)
    out.il = find_il(norm)
    out.ilce = find_ilce(norm, out.il)

    # Temel alanlar
    m = RE_NO.search(norm);        out.no    = m.group(1) if m else ""
    m = RE_KAT.search(norm) or RE_KAT_K.search(norm)
    out.kat = m.group(1) if m else ""
    m = RE_DAIRE.search(norm);     out.daire = m.group(1) if m else ""
    m = RE_BLOK.search(norm);      out.blok  = (m.group(1).lower() if m else "")
    m = RE_SITE.search(norm);      out.site      = (m.group(1).title() if m else "").strip()
    m = RE_APT.search(norm);       out.apartman  = (m.group(1).title() if m else "").strip()
    m = RE_MAH.search(norm);       out.mahalle   = (m.group(1).title() if m else "").strip()
    # Mahalle son-kuyruk temizliği (sitesi/sanayi vb.)
    if out.mahalle:
        out.mahalle = trim_mahalle_tail(out.mahalle)

    # Anchor çıkarımı (sıra bağımsız)
    out.cadde  = extract_anchor_phrase(norm, "caddesi")
    out.sokak  = extract_anchor_phrase(norm, "sokak")
    out.bulvar = extract_anchor_phrase(norm, "bulvarı")

    # Sokak: içinde sayı varsa mevkî vb. gürültüyü at ve son sayıyı tut
    if out.sokak:
        out.sokak = prune_street_phrase(out.sokak)

    # Sayılı sokak fallback'i (anchor boş kaldıysa)
    if not out.sokak:
        m_num_sok = re.search(r"\b(\d{1,5})\s+sokak\b", norm)
        if m_num_sok:
            out.sokak = m_num_sok.group(1) + "."

    # Tamamen sayısal sokak "864" gibi geldiyse noktalı forma normalize et
    if out.sokak and out.sokak.isdigit():
        out.sokak = out.sokak + "."

    # Site/Apartman: 'no/sokak/cadde' artıkları ve sayısal kuyrukları temizle
    if out.site:
        out.site = clean_place_name(out.site)
    if out.apartman:
        out.apartman = clean_place_name(out.apartman)


    # Mahalleden il/ilçe tamamlama (sözlük tabakası)
    if out.mahalle and (not out.ilce or not out.il):
        out.ilce, out.il = infer_from_components(out.mahalle, out.sokak, out.cadde, out.ilce, out.il)

    return out

def run_cli(args: argparse.Namespace) -> None:
    load_gazetteer(args.gazetteer)

    if args.dry_run:
        n = int(args.dry_run)
        with open(args.input, "r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for i, row in enumerate(r):
                if i >= n: break
                raw = row.get(args.text_col, "")
                pid = row.get(args.id_col, i)
                parsed = parse_address(raw)
                out = {**row, **asdict(parsed)}
                print(f"[{pid}] {raw}\n -> {out}\n")
        return

    with open(args.input, "r", encoding="utf-8") as fin, \
         open(args.output, "w", encoding="utf-8", newline="") as fout:
        r = csv.DictReader(fin)
        base_fields = r.fieldnames or []
        new_fields = ["normalized","il","ilce","mahalle","sokak","cadde","bulvar","no","kat","daire","blok","site","apartman"]
        w = csv.DictWriter(fout, fieldnames=[*base_fields, *new_fields])
        w.writeheader()
        for row in r:
            raw = row.get(args.text_col, "")
            parsed = parse_address(raw)
            w.writerow({**row, **asdict(parsed)})
    print(f"✓ Parsed CSV written to: {args.output}")

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Input CSV path (must contain address text column)")
    p.add_argument("--output", default="parsed.csv", help="Output CSV path")
    p.add_argument("--text-col", default="address", help="Name of address text column")
    p.add_argument("--id-col", default="id", help="Name of id column (for logs)")
    p.add_argument("--dry-run", default=0, help="Print first N parsed rows instead of writing file")
    p.add_argument("--gazetteer", default="", help="Optional CSV with columns: mahalle,ilce,il")
    return p

if __name__ == "__main__":
    run_cli(build_argparser().parse_args())
