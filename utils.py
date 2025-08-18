# -*- coding: utf-8 -*-
from dataclasses import dataclass
import re
from typing import List, Set

# ---- Sabitler ----
ILLER: Set[str] = {
    "adana","adiyaman","afyonkarahisar","agri","amasya","ankara","antalya","artvin","aydin",
    "balikesir","bilecik","bingol","bitlis","bolu","burdur","bursa","canakkale","cankiri",
    "corum","denizli","diyarbakir","edirne","elazig","erzincan","erzurum","eskisehir","gaziantep",
    "giresun","gumushane","hakkari","hatay","isparta","mersin","istanbul","izmir","kars","kastamonu",
    "kayseri","kirklareli","kirsehir","kocaeli","konya","kutahya","malatya","manisa","kahramanmaras",
    "mardin","mugla","mus","nevsehir","nigde","ordu","rize","sakarya","samsun","siirt","sinop","sivas",
    "tekirdag","tokat","trabzon","tunceli","sanliurfa","usak","van","yozgat","zonguldak","aksaray",
    "bayburt","karaman","kirikkale","batman","sirnak","bartin","ardahan","igdir","yalova","karabuk",
    "kilis","osmaniye","duzce"
}
ANCHOR_WORDS = {"sokak","caddesi","bulvarı"}
STOPWORDS_BACK = {
    "mahallesi","no","kat","k","d","blok","sitesi","apartmanı","apt","daire"
} | ANCHOR_WORDS | ILLER

# ---- Yardımcılar ----
def tr_lower(s: str) -> str:
    return (s or "").translate(str.maketrans("IİÇĞÖŞÜ", "ıiçğöşü")).lower()

def clean_token(tok: str) -> str:
    t = tr_lower(tok)
    return re.sub(r"[^\wçğıöşü]+", "", t)

def is_stop(tok: str) -> bool:
    return clean_token(tok) in STOPWORDS_BACK

def unique_collapse(words: List[str]) -> str:
    out = []
    for w in words:
        if not out or out[-1] != w:
            out.append(w)
    return " ".join(out)

def levenshtein(a: str, b: str) -> int:
    if a == b: return 0
    if not a: return len(b)
    if not b: return len(a)
    prev = list(range(len(b)+1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j]+1, cur[j-1]+1, prev[j-1]+cost))
        prev = cur
    return prev[-1]

TR_FOLD_MAP = str.maketrans("çğıöşü", "cgiosu")

def fold_tr(s: str) -> str:
    # önce mevcut clean_token filtresi ile sadeleştir, sonra aksanı kaldır
    from utils import clean_token as _ct  # circular import riskini önlemek için local import kullanma!
    # DİKKAT: burada utils.clean_token çağırmayalım; aynı dosyadayız.
    # Bu yüzden aşağıdaki satırı düzenleyelim:
    s2 = (s or "")
    s2 = s2.translate(str.maketrans("IİÇĞÖŞÜ", "ıiçğöşü")).lower()
    s2 = re.sub(r"[^\wçğıöşü]+", "", s2)  # clean_token ile aynı filtre
    return s2.translate(TR_FOLD_MAP)

# İllerin aksansız versiyon seti
ILLER_FOLDED = { fold_tr(x) for x in ILLER }

def is_il_token(tok: str) -> bool:
    return fold_tr(tok) in ILLER_FOLDED

@dataclass
class Parsed:
    normalized: str
    il: str = ""
    ilce: str = ""
    mahalle: str = ""
    sokak: str = ""
    cadde: str = ""
    bulvar: str = ""
    no: str = ""
    kat: str = ""
    daire: str = ""
    blok: str = ""
    site: str = ""
    apartman: str = ""
