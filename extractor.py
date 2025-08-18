# -*- coding: utf-8 -*-
import re
from utils import ILLER, ANCHOR_WORDS, STOPWORDS_BACK, clean_token, is_stop, unique_collapse

# Regexler
RE_NO    = re.compile(r"\bno\s*([0-9]+(?:/[0-9a-z]+)?|[0-9]+[a-z]?)\b", re.I)
RE_KAT   = re.compile(r"\bkat\s*([0-9]+)\b", re.I)
RE_KAT_K = re.compile(r"\bk\s*[:\.]?\s*([0-9]+)\b", re.I)
RE_DAIRE = re.compile(r"\bd\s*([0-9]+[a-z]?)\b", re.I)
RE_BLOK  = re.compile(r"\b([a-zçğıöşü]{1,2})\s*blok\b", re.I)
RE_SITE  = re.compile(r"\b([a-zçğıöşü0-9\s\-]+?)\s+sitesi\b", re.I)
RE_APT   = re.compile(r"\b([a-zçğıöşü0-9\s\-]+?)\s+apartman(?:ı|i)?\b", re.I)
RE_MAH = re.compile(
    r"\b([a-zçğıöşü0-9\.\-]+(?:\s+[a-zçğıöşü0-9\.\-]+)*)\s+mahallesi\b",
    re.I
)

def find_il(norm: str) -> str:
    tokens = norm.split()
    for tok in reversed(tokens):
        ct = clean_token(tok)
        if ct in ILLER:
            return ct.title()
    return ""

def find_ilce(norm: str, il: str) -> str:
    # parantez içini at
    base = re.sub(r"\([^)]*\)", " ", norm)

    # 1) "<ilçe>/<il>" ya da "<ilçe> / <il>"
    m = re.search(r"\b([a-zçğıöşü\-\. ]+?)\s*/\s*([a-zçığıöşü\-\. ]+)\b", base)
    if m:
        left_raw = m.group(1).strip()
        right = clean_token(m.group(2)).title().strip()
        if right.lower() in ILLER:
            left_tokens = [clean_token(x) for x in left_raw.split() if clean_token(x)]
            if left_tokens:
                last_left = left_tokens[-1].title()              # <- SOLDAN SON TOKEN
                if last_left.lower() != right.lower():
                    return last_left

    # 2) İl bulunduysa: il'in hemen öncesinden yalnızca **son** tokenı al
    if il:
        parts = base.split()
        il_l = il.lower()
        idx = -1
        for i, t in enumerate(parts):
            if clean_token(t) == il_l:
                idx = i
        if idx != -1:
            window = []
            j = idx - 1
            # stopword/iller/sayı gelene kadar geriye yürü (max 3 token topla)
            while j >= 0 and len(window) < 3:
                tok = parts[j]
                ct = clean_token(tok)
                if not ct or ct.isdigit() or ct in ILLER or ct in STOPWORDS_BACK:
                    break
                window.append(ct.title())
                j -= 1
            window.reverse()
            if window:
                # YALNIZCA SAĞDAKİ (SON) TOKENI İLÇE AL
                last_tok = window[-1]
                if last_tok.lower() != il_l:
                    return last_tok
    return ""


def extract_anchor_phrase(norm: str, anchor: str) -> str:
    tokens = norm.split()
    idx = -1
    for i, t in enumerate(tokens):
        if clean_token(t) == anchor:  # noktalı varyantları yakalar
            idx = i
            break
    if idx == -1:
        return ""
    seg = []
    j = idx - 1
    while j >= 0:
        w = tokens[j]
        if is_stop(w):
            break
        seg.append(w)
        j -= 1
    seg.reverse()
    phrase = " ".join(seg).strip()
    phrase = re.sub(r"\s+", " ", phrase)
    phrase = re.sub(r"[^\wçğıöşü\s\.\-]+$", "", phrase).strip()
    return phrase.title()

NOISE_BEFORE_STREET = {"mevkii", "mevkisi", "bolgesi", "bölgesi"}

def prune_street_phrase(phrase: str) -> str:
    # gürültü kelimelerini sil
    toks = phrase.split()
    toks = [t for t in toks if clean_token(t) not in NOISE_BEFORE_STREET]
    # içinde sayı varsa en sondaki sayıyı (opsiyonel nokta) bırak
    nums = [t for t in toks if clean_token(t).isdigit()]
    if nums:
        return nums[-1] + ("" if nums[-1].endswith(".") else ".")
    # yoksa olduğu gibi
    return " ".join(toks)

PLACE_STOP = {"no","sokak","caddesi","cadde","bulvarı","blok","d","kat"}

def clean_place_name(s: str) -> str:
    if not s: return s
    parts = s.split()
    out = []
    for p in parts:
        cp = clean_token(p)
        if not cp or cp.isdigit() or cp in PLACE_STOP:
            break  # bu ve sonrasını at
        out.append(p)
    name = " ".join(out).strip()
    # çok uzun kaçtıysa son 2-3 kelimeyi tut
    tokens = name.split()
    if len(tokens) > 3:
        tokens = tokens[-3:]
    return " ".join(tokens).title()

MAH_TAIL_NOISE = {"sitesi","site","sanayi","osb","organize"}
MAH_TWO_TOKEN_WHITELIST = {("yeni","sanayi"), ("eski","sanayi")}

def trim_mahalle_tail(name: str) -> str:
    if not name: 
        return name
    toks = name.split()
    low = [clean_token(t) for t in toks]

    # 0) "... sitesi <X>" kalıbı: "sitesi" geçiyorsa, ondan SONRAKİ kısım mahalle adayıdır
    if "sitesi" in low:
        idx = low.index("sitesi")
        tail = toks[idx+1:]
        if tail:
            toks = tail
            low = [clean_token(t) for t in toks]

    # 1) "Yeni Sanayi" / "Eski Sanayi" gibi ikilileri AYNI BIRAK
    if len(low) >= 2 and tuple(low[-2:]) in MAH_TWO_TOKEN_WHITELIST:
        return " ".join(toks[-2:]).title()

    # 2) "sanayi/site/osb/organize" kuyruklarını sadece 2’den FAZLA token varsa at
    while len(toks) > 2 and clean_token(toks[-1]) in MAH_TAIL_NOISE:
        toks.pop()

    # 3) Sonuç: en fazla son 2 token'ı tut (çok uzun isimleri sadeleştirmek için)
    if len(toks) >= 2:
        return " ".join(toks[-2:]).title()
    return " ".join(toks).title()

