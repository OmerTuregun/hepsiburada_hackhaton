# inspect_index.py
import json, argparse, re

FIELDS = {"mahalle","cadde","sokak","site","apartman"}

ap = argparse.ArgumentParser()
ap.add_argument("--kb", required=True, help="cache/gazetteer_index.json")
ap.add_argument("--field", required=True, choices=FIELDS)
ap.add_argument("--query", required=True, help="anahtar (normalized) içinde arama; regex veya düz metin")
ap.add_argument("--top", type=int, default=10)
args = ap.parse_args()

with open(args.kb, "r", encoding="utf-8") as f:
    data = json.load(f)

bucket = data.get(args.field, {})
rx = re.compile(args.query, re.I)

matches = [(k, v) for k, v in bucket.items() if rx.search(k)]
if not matches:
    print("Eşleşme yok.")
    raise SystemExit(0)

for k, v in matches:
    # v, kaydederken list(cc.items()) şeklindeydi: [ [[il, ilçe], count], ... ]
    pairs = []
    for item in v:
        pair, cnt = item
        il, ilce = pair if isinstance(pair, list) else (pair[0], pair[1])
        pairs.append((cnt, il or "", ilce or ""))

    pairs.sort(reverse=True)
    print(f"\n== {k} ==")
    for cnt, il, ilce in pairs[:args.top]:
        print(f"  {il}/{ilce}: {cnt}")
