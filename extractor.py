# -*- coding: utf-8 -*-
import re
from utils import ILLER, ANCHOR_WORDS, STOPWORDS_BACK, clean_token, is_stop, unique_collapse, is_il_token

# Regexler
RE_NO    = re.compile(r"\bno\s*([0-9]+(?:/[0-9a-z]+)?|[0-9]+[a-z]?)\b", re.I)
RE_KAT   = re.compile(r"\bkat\s*([0-9]+)\b", re.I)
RE_KAT_K = re.compile(r"\bk\s*[:\.]?\s*([0-9]+)\b", re.I)
RE_DAIRE = re.compile(r"\bd\s*([0-9]+[a-z]?)\b", re.I)
RE_BLOK  = re.compile(r"\b([a-zçğıöşü]{1,2})\s*blok\b", re.I)
RE_SITE  = re.compile(r"\b([a-zçğıöşü0-9\s\-]+?)\s+sitesi\b", re.I)
RE_APT = re.compile(r"\b([a-zçğıöşü0-9\s\-]+?)\s+apartman(?:ı|i)?\b", re.I)
RE_MAH = re.compile(
    r"\b([a-zçğıöşü0-9\.\-]+(?:\s+[a-zçğıöşü0-9\.\-]+)*)\s+mahallesi\b",
    re.I
)

def find_il(norm: str) -> str:
    tokens = norm.split()
    for tok in reversed(tokens):
        if is_il_token(tok):
            # Kullanıcıya TitleCase döndürürken aksanını koruyamayabiliriz;
            # en azından stringi normalize edip ilk harf büyük yapalım.
            ct = clean_token(tok)
            return ct.title()
    return ""

def find_ilce(norm: str, il: str) -> str:
    base = re.sub(r"\([^)]*\)", " ", norm)

    # 1) 'ilçe/il' kalıbı: sağ tarafı IL olan son eşleşmeyi seç
    cand = ""
    for m in re.finditer(r"\b([a-zçğıöşü\.\- ]+?)\s*/\s*([a-zçğıöşü\.\- ]+)\b", base):
        left_raw  = m.group(1).strip()
        right_raw = m.group(2).strip()
        if is_il_token(right_raw):
            left_tokens = [clean_token(x) for x in left_raw.split() if clean_token(x)]
            if left_tokens:
                last_left = left_tokens[-1].title()
                if not is_il_token(last_left):
                    cand = last_left
    if cand:
        return cand

    # 2) İl bulunduysa: 'il' kelimesinden 1–3 token sola bak, en yakını al
    if il:
        parts = base.split()
        il_norm = clean_token(il).lower()
        idxs = [i for i, t in enumerate(parts) if is_il_token(t)]
        for idx in reversed(idxs):
            if clean_token(parts[idx]).lower() != il_norm:
                continue
            j = idx - 1
            while j >= 0:
                ct = clean_token(parts[j])
                if ct and not ct.isdigit() and not is_il_token(ct) and ct not in STOPWORDS_BACK:
                    return ct.title()
                j -= 1
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

PLACE_STOP = {"no","sokak","caddesi","cadde","bulvarı","blok","d","kat"}

def clean_place_name(s: str) -> str:
    if not s: 
        return s
    parts = s.split()
    out = []
    # sağdan sola ilerle, ilk 1–3 anlamlı kelimeyi al
    for p in reversed(parts):
        cp = clean_token(p)
        if not cp or cp.isdigit() or cp in PLACE_STOP:
            if out:  # bir şey topladıysak, burada kes
                break
            else:
                continue
        out.append(p)
        if len(out) >= 3:
            break
    out.reverse()
    return " ".join(out).title()


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

