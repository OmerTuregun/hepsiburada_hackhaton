# -*- coding: utf-8 -*-
import re
from utils import ILLER, ANCHOR_WORDS, STOPWORDS_BACK, clean_token, is_stop, unique_collapse, is_il_token
from normalizer import normalize_text, normalize

# Regexler
RE_NO    = re.compile(r"\bno\s*([0-9]+(?:/[0-9a-z]+)?|[0-9]+[a-z]?)\b", re.I)
RE_KAT   = re.compile(r"\bkat\s*([0-9]+)\b", re.I)
RE_KAT_K = re.compile(r"\bk\s*[:\.]?\s*([0-9]+)\b", re.I)
RE_DAIRE = re.compile(r"\bd\s*([0-9]+[a-z]?)\b", re.I)
RE_BELEDIYE = re.compile(r"\b([a-zçğıöşü0-9\.\- ]+?)\s+belediyesi\b", re.I)
RE_BLOK  = re.compile(r"\b([a-zçğıöşü]{1,2})\s*blok\b", re.I)
RE_SITE  = re.compile(r"\b([a-zçğıöşü0-9\s\-]+?)\s+sitesi\b", re.I)
RE_APT = re.compile(r"\b([a-zçğıöşü0-9\s\-]+?)\s+apartman(?:ı|i)?\b", re.I)
RE_MAH = re.compile(
    r"\b([a-zçğıöşü0-9\.\-]+(?:\s+[a-zçğıöşü0-9\.\-]+)*)\s+mahallesi\b",
    re.I
)

# İlçe adayında kesinlikle istemediğimiz kelimeler
ILCE_NOISE = {
    "yeni","sanayi","osb","organize","mevkii","mevkisi","bölgesi","bolgesi",
    "daire","blok","site","sitesi","apartman","apartmanı","no","kat","d","k",
    "mahallesi","mah","mh","m"
}

# Gürültü (POI) kelimeleri: ilçe fallback'te atlanacak
POI_NOISE = {
    "foto", "fotograf", "işhanı", "ishanı", "market", "eczane", "ofis",
    "oto", "otomotiv", "sanayi", "sitesi", "site", "apt", "apartman",
    "blok"
}

def find_il(norm: str) -> str:
    tokens = norm.split()
    for tok in reversed(tokens):
        # "fethiye/muğla" gibi birleşik tokenları da tara
        for part in reversed(re.split(r"/", tok)):
            if is_il_token(part):
                return clean_token(part).title()
    return ""

def find_ilce(norm: str, il: str) -> str:
    base = re.sub(r"\([^)]*\)", " ", norm)
    
    m_bel = re.search(r"\b([a-zçğıöşü]+)\s+belediyesi\b", base)
    if m_bel:
        cand = clean_token(m_bel.group(1))
        if cand and not is_il_token(cand):
            return cand.title()
    # 0) Token bazlı hızlı tarama: "X/Y" veya "X / Y"
    parts = base.split()
    # a) Tek token içinde slash: "fethiye/muğla"
    for tok in reversed(parts):
        if "/" in tok:
            left, _, right = tok.rpartition("/")
            if is_il_token(right):
                lt = [clean_token(x) for x in left.split() if clean_token(x)]
                if lt:
                    cand = lt[-1].title()
                    if cand and not is_il_token(cand):
                        return cand
    # b) Ayrı token olarak slash: "... dikili / izmir ..."
    for i in range(1, len(parts)-1):
        if parts[i] == "/" and is_il_token(parts[i+1]):
            left = clean_token(parts[i-1])
            if left and not is_il_token(left):
                return left.title()

    # 1) Regex ile genel tarama (çeşitli boşluk varyantları)
    cand = ""
    pattern = r"([A-Za-zÇĞİÖŞÜçğıöşü0-9\.\- ]+?)\s*/\s*([A-Za-zÇĞİÖŞÜçğıöşü0-9\.\- ]+)"
    for m in re.finditer(pattern, base):
        left_raw  = m.group(1).strip()
        right_raw = m.group(2).strip()
        if is_il_token(right_raw):
            ltoks = [clean_token(x) for x in left_raw.split() if clean_token(x)]
            if ltoks:
                last_left = ltoks[-1].title()
                if not is_il_token(last_left):
                    cand = last_left
    if cand:
        return cand

        # 2) Fallback: sonda bulunan 'İl' tokenından sola yürü
    if il:
        il_norm = clean_token(il).lower()
        for idx in reversed(range(len(parts))):
            if is_il_token(parts[idx]) and clean_token(parts[idx]).lower() == il_norm:
                j = idx - 1
                while j >= 0:
                    raw  = parts[j]
                    ct   = clean_token(raw)
                    prev = clean_token(parts[j-1]) if j-1 >= 0 else ""

                    # filtre: kısa/tek harf/numara/stop/noise
                    if not ct or ct.isdigit() or len(ct) < 3:
                        j -= 1; continue
                    if ct in ILCE_NOISE or prev in ILCE_NOISE:
                        j -= 1; continue
                    if is_il_token(ct):
                        j -= 1; continue

                    # "fethiye/muğla" benzeri birleşikler
                    if "/" in raw:
                        subs = [clean_token(s) for s in raw.split("/")]
                        for sub in reversed(subs):
                            if sub and len(sub) >= 3 and sub not in ILCE_NOISE and not is_il_token(sub):
                                return sub.title()

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

def parse_address(text: str) -> dict:
    """
    CLI'nin beklediği arayüz:
    Girdi: ham adres (str)
    Çıktı: {
      'address','normalized','il','ilce','mahalle','sokak','cadde','bulvar',
      'no','kat','daire','blok','site','apartman'
    }
    """
    raw = (text or "").strip()
    norm = normalize(raw)

    out = {
        "address": raw,
        "normalized": norm,
        "il": "",
        "ilce": "",
        "mahalle": "",
        "sokak": "",
        "cadde": "",
        "bulvar": "",
        "no": "",
        "kat": "",
        "daire": "",
        "blok": "",
        "site": "",
        "apartman": "",
    }

    # --- Temel sayısal alanlar ---
    m = RE_NO.search(norm)
    if m: out["no"] = m.group(1).lower()

    m = RE_KAT.search(norm) or RE_KAT_K.search(norm)
    if m: out["kat"] = m.group(1)

    m = RE_DAIRE.search(norm)
    if m: out["daire"] = m.group(1).lower()

    m = RE_BLOK.search(norm)
    if m: out["blok"] = m.group(1).lower()

    m = RE_SITE.search(norm)
    if m:
        # "xxx sitesi" öncesini alıyoruz
        out["site"] = clean_place_name(m.group(1)).title()

    m = RE_APT.search(norm)
    if m:
        out["apartman"] = clean_place_name(m.group(1)).title()

    # --- Mahalle ---
    m = RE_MAH.search(norm)
    if m:
        out["mahalle"] = trim_mahalle_tail(m.group(1)).title()

    # --- Cadde / Sokak / Bulvar (anchor bazlı) ---
    # utils.ANCHOR_WORDS beklenen ör.: {"cadde": ["caddesi","cadde","cad.","cd."], "sokak": [...], "bulvar": [...]}
    for canon, anchors in ANCHOR_WORDS.items():
        for a in anchors:
            phrase = extract_anchor_phrase(norm, a)
            if not phrase:
                continue
            if canon == "sokak":
                out["sokak"] = prune_street_phrase(phrase)
            elif canon == "cadde":
                out["cadde"] = clean_place_name(phrase).title()
            elif canon == "bulvar":
                out["bulvar"] = clean_place_name(phrase).title()
            # aynı tür için ilk sağlam eşleşmede dur
            if out[canon]:
                break

    # --- İl / İlçe ---
    out["il"] = find_il(norm)
    out["ilce"] = find_ilce(norm, out["il"])

    return out
