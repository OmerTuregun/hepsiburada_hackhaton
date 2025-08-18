# -*- coding: utf-8 -*-
import re
from typing import List, Tuple
from utils import tr_lower

ABBR_MAP: List[Tuple[str,str]] = [
    (r"\bmah\.?\b", "mahallesi"),
    (r"\bmh\.?\b", "mahallesi"),
    (r"\bm\b", "mahallesi"),
    (r"\bsok\.?\b", "sokak"),
    (r"\bsk\.?\b", "sokak"),
    (r"\bcad\.?\b", "caddesi"),
    (r"\bcd\.?\b", "caddesi"),
    (r"\bcadde\b", "caddesi"),
    (r"\bblv\.?\b", "bulvarı"),
    (r"\bbulv\.?\b", "bulvarı"),
    (r"\bapt\.?\b", "apartmanı"),
    (r"\bap\b", "apartmanı"),
    (r"\bdair\.?\b", "d"),
    (r"\bdaire\b", "d"),
    (r"\bd\.\b", "d"),
    (r"\bblok\b", "blok"),
    (r"\bno\b", "no"),
    (r"\bkat\b", "kat"),
    (r"\bk\.?\b", "kat"),
]

def normalize(text: str) -> str:
    t = tr_lower(text or "")
    # Ayırıcıları sadeleştir
    t = re.sub(r"[;,|]+", " ", t)

    # Kısaltma genişlet
    for pat, repl in ABBR_MAP:
        t = re.sub(pat, repl, t)

    # no/kat/d/k varyantlarını birleştir
    for key in ["no", "kat", "d", "k"]:
        t = t.replace(f"{key}:", f"{key} ").replace(f"{key}.", f"{key} ").replace(f"{key}/", f"{key} ")

     # --- SAYILI/BİTİŞİK YOLLARIN AYRIŞTIRILMASI ---

    # 864.sokak, 864sok, 147sok, 417.sk -> "864 sokak"
    t = re.sub(r"\b(\d+(?:/\d+)?)\.(?:sokak|sok|sk)\b", r"\1 sokak", t)
    t = re.sub(r"\b(\d+(?:/\d+)?)\s*(?:sok|sk)\b",          r"\1 sokak", t)

    # 120.cad, 120cad, 2716/2.cd -> "120 caddesi" / "2716/2 caddesi"
    t = re.sub(r"\b(\d+(?:/\d+)?)\.(?:caddesi|cadde|cad|cd)\b", r"\1 caddesi", t)
    t = re.sub(r"\b(\d+(?:/\d+)?)\s*(?:cadde|cad|cd)\b",        r"\1 caddesi", t)

    # 75.blv, 75blv, 9.bulv -> "75 bulvarı"
    t = re.sub(r"\b(\d+(?:/\d+)?)\.(?:bulvarı|bulvar|blv|bulv)\b", r"\1 bulvarı", t)
    t = re.sub(r"\b(\d+(?:/\d+)?)\s*(?:bulvarı|bulvar|blv|bulv)\b", r"\1 bulvarı", t)

    # kat/no/d bitişmeleri ayır
    t = re.sub(r"\b(kat|no|d)\s*([0-9])", r"\1 \2", t)

    # boşluk temizle
    t = re.sub(r"\s+", " ", t).strip()
    return t

# Tek tip boşluk
_SPACE_RE = re.compile(r"\s+")

def normalize_text(s: str) -> str:
    """
    Co-occurrence index ve parser için metni yalınlaştırır:
    - Türkçe lower (tr_lower)
    - Virgül/; gibi ayırıcıları boşluğa çevirir
    - Birden fazla boşluğu tek boşluğa indirir
    - Baştaki/sondaki boşlukları kırpar
    Not: nokta (.) ve slash (/) bilgilerini (ör. 417. / fethiye/muğla) bozmayız.
    """
    if not s:
        return ""
    s = tr_lower(s)
    s = s.replace(",", " ").replace(";", " ")
    # Bazı Unicode boşluk anomalileri vs.
    s = _SPACE_RE.sub(" ", s).strip()
    return s