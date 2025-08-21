# ml_resolver.py
# -*- coding: utf-8 -*-
import os, joblib
from typing import Optional, Tuple
from normalizer import normalize_text
from extractor import parse_address  # sadece type için; dışarıdan parsed veriyoruz

def make_feat_for_parsed(parsed: dict) -> str:
    parts = []
    for f in ("mahalle","cadde","sokak","site","apartman"):
        v = (parsed.get(f) or "").strip()
        if v:
            parts.append(f"{f}={normalize_text(v)}")
    return " | ".join(parts)

class MLResolver:
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(model_path)
        self.pipe = joblib.load(model_path)

    def infer(self, parsed: dict) -> Tuple[str, str, float]:
        feat = make_feat_for_parsed(parsed)
        if not feat:
            return "", "", 0.0
        proba = None
        if hasattr(self.pipe, "predict_proba"):
            probs = self.pipe.predict_proba([feat])[0]
            idx = probs.argmax()
            label = self.pipe.classes_[idx]
            proba = float(probs[idx])
        else:
            label = self.pipe.predict([feat])[0]
            proba = 1.0  # proba desteği yoksa 1.0 varsay
        il, ilce = label.split("|", 1)
        return il or "", ilce or "", proba
