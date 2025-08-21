# resolver.py
import os, json
from collections import Counter, defaultdict
from typing import Dict, Tuple, Optional
from normalizer import normalize_text
from utils import clean_token, is_il_token, STOPWORDS_BACK

# Şüpheli/ilçe olmaz kelimeler
ILCE_BLACKLIST = {
    "yeni","sanayi","osb","organize","mevkii","mevkisi","bölgesi","bolgesi",
    "daire","blok","site","sitesi","apartman","apartmanı","no","kat","d","k",
    "mahallesi","mah","mh","m"
}

# Index’lenecek alanlar
_KEYS = ("mahalle", "cadde", "sokak", "site", "apartman")


class LocationResolver:
    """
    Eşgörünüm tabanlı (co-occurrence) il/ilçe çıkarıcı.
    idx[field][normalized_key] => Counter({(il, ilçe): count})
    """
    def __init__(self):
        self.idx: Dict[str, Dict[str, Counter]] = {
            k: defaultdict(Counter) for k in _KEYS
        }

    @staticmethod
    def _is_good_ilce(s: str) -> bool:
        ct = clean_token(s)
        if not ct or ct.isdigit():
            return False
        if len(ct) < 3:
            return False
        if ct in STOPWORDS_BACK or ct in ILCE_BLACKLIST:
            return False
        if is_il_token(ct):  # il adının kendisi olamaz
            return False
        return True

    # --------- Modelin beslenmesi ----------
    def observe(self, row: Dict[str, str]):
        il = (row.get("il") or "").strip().title()
        ilce = (row.get("ilce") or "").strip().title()

        # İl geçerli değilse hiç ekleme
        if not il or not is_il_token(il):
            return

        # İlçe kötü/boşsa kaydı atma; boş olarak indexe yaz
        if not ilce or not self._is_good_ilce(ilce):
            ilce = ""

        pair = (il, ilce)

        # Alanları indexe işle
        for k in _KEYS:
            val = (row.get(k) or "").strip()
            if not val:
                continue
            key = normalize_text(val)
            if not key:
                continue
            self.idx[k][key][pair] += 1

    # --------- Kalıcı hale getirme ----------
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        serial = {k: {kk: list(cc.items()) for kk, cc in v.items()} for k, v in self.idx.items()}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(serial, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "LocationResolver":
        inst = cls()
        if not os.path.exists(path):
            return inst
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for k in _KEYS:
            for kk, items in data.get(k, {}).items():
                inst.idx[k][kk] = Counter({tuple(p): c for p, c in items})
        return inst

    # --------- Çıkarım ----------
    def infer(self,
              mahalle: Optional[str] = None,
              sokak: Optional[str] = None,
              cadde: Optional[str] = None,
              site: Optional[str] = None,
              apartman: Optional[str] = None,
              il_hint: Optional[str] = None,
              ilce_hint: Optional[str] = None) -> Tuple[str, str, float]:
        """
        Dönüş: (il, ilçe, skor). Boşsa "".
        Ağırlıklar: mahalle 3.0, cadde 2.0, site 2.5, sokak 1.5, apartman 1.0
        """
        weights = {"mahalle": 3.0, "cadde": 2.0, "site": 2.5, "sokak": 1.5, "apartman": 1.0}

        items = {
            "mahalle": mahalle,
            "cadde": cadde,
            "sokak": sokak,
            "site": site,
            "apartman": apartman,
        }

        def _collect(allow_relax: bool) -> Counter:
            cands = Counter()
            for field, val in items.items():
                if not val:
                    continue
                key = normalize_text(val)
                bucket = self.idx[field].get(key)
                if not bucket:
                    continue
                total = sum(bucket.values()) or 1
                w = weights[field]
                for (il, ilce), cnt in bucket.items():
                    # il ipucunu her zaman uygula
                    if il_hint and il and il.lower() != il_hint.lower():
                        continue
                    # ilçe ipucunu ilk geçişte uygula; sonuç yoksa gevşet
                    if not allow_relax and ilce_hint and ilce and ilce.lower() != ilce_hint.lower():
                        continue
                    cands[(il, ilce)] += w * (cnt / total)
            return cands

        cands = _collect(allow_relax=False)
        if not cands:
            cands = _collect(allow_relax=True)
        if not cands:
            return "", "", 0.0

        (il, ilce), score = cands.most_common(1)[0]
        return il or "", ilce or "", float(score)
