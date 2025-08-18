# -*- coding: utf-8 -*-
import os, csv
from typing import Dict, List, Optional, Tuple
from utils import tr_lower, clean_token, levenshtein

# Gömülü mini sözlük (istersen burada doldur)
EMBEDDED_GAZETTEER: Dict[str, List[Dict[str,str]]] = {
    # "akarca": [{"ilce": "Fethiye", "il": "Muğla"}],
    # "bitez": [{"ilce": "Bodrum", "il": "Muğla"}],
}

_gazetteer: Dict[str, List[Dict[str,str]]] = {}

def load_gazetteer(path: Optional[str]) -> None:
    global _gazetteer
    _gazetteer.clear()
    # 1) gömülü
    for mk, lst in EMBEDDED_GAZETTEER.items():
        _gazetteer[mk] = [{"ilce": x["ilce"].title(), "il": x["il"].title()} for x in lst]
    # 2) csv (varsa)
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                mah = tr_lower((r.get("mahalle") or "").strip())
                ilce = (r.get("ilce") or "").strip().title()
                il = (r.get("il") or "").strip().title()
                if not mah or not ilce or not il: 
                    continue
                _gazetteer.setdefault(mah, [])
                item = {"ilce": ilce, "il": il}
                if item not in _gazetteer[mah]:
                    _gazetteer[mah].append(item)

def infer_from_components(mahalle: str, sokak: str, cadde: str, cur_ilce: str, cur_il: str) -> Tuple[str,str]:
    ilce, il = cur_ilce, cur_il
    mah_key = tr_lower(mahalle or "")
    sok_key = tr_lower(sokak or "")
    cad_key = tr_lower(cadde or "")

    if not _gazetteer:
        return ilce, il

    # exact
    if mah_key in _gazetteer:
        candidates = _gazetteer[mah_key]
    else:
        # fuzzy (≤1)
        best_key, best_d = None, 10**9
        for mk in _gazetteer.keys():
            d = levenshtein(mah_key, mk)
            if d < best_d:
                best_key, best_d = mk, d
        candidates = _gazetteer.get(best_key, []) if best_d <= 1 else []

    if not candidates:
        return ilce, il

    if len(candidates) == 1:
        return (ilce or candidates[0]["ilce"], il or candidates[0]["il"])

    def score(c):
        s = 0
        if cad_key and tr_lower(c["ilce"]) in cad_key: s += 2
        if sok_key and tr_lower(c["ilce"]) in sok_key: s += 2
        if cad_key and tr_lower(c["il"])   in cad_key: s += 1
        if sok_key and tr_lower(c["il"])   in sok_key: s += 1
        return s

    ranked = sorted(candidates, key=score, reverse=True)
    top = ranked[0] if ranked else candidates[0]
    bests = [c for c in ranked if score(c) == score(top)]
    best = min(bests, key=lambda x: len(x["ilce"])) if len(bests) > 1 else top

    return (ilce or best["ilce"], il or best["il"])
