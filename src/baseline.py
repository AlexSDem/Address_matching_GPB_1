"""Baseline address matcher: TF-IDF char n-grams retrieval + RapidFuzz rerank
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from .baseline_normalize import normalize_ru_address


@dataclass
class MatchResult:
    query: str
    best: str
    cosine_sim: float
    fuzz_score: float
    final_score: float
    best_index: int


class AddressMatcher:
    def __init__(
        self,
        ngram_range: Tuple[int, int] = (2, 4),
        analyzer: str = "char_wb",
        top_k: int = 10,
        w_cosine: float = 0.6,
        w_fuzz: float = 0.4,
        do_normalize: bool = True,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        if w_cosine < 0 or w_fuzz < 0 or (w_cosine + w_fuzz) <= 0:
            raise ValueError("weights must be non-negative and sum > 0")

        self.ngram_range = ngram_range
        self.analyzer = analyzer
        self.top_k = top_k
        self.w_cosine = w_cosine
        self.w_fuzz = w_fuzz
        self.do_normalize = do_normalize

        self.vectorizer: Optional[TfidfVectorizer] = None
        self.nn: Optional[NearestNeighbors] = None
        self._ref: Optional[pd.Series] = None

    def _prep(self, x: str) -> str:
        return normalize_ru_address(x) if self.do_normalize else str(x)

    def fit(self, reference_addresses: Sequence[str]) -> "AddressMatcher":
        ref = pd.Series(list(reference_addresses), name="ref")
        if ref.empty:
            raise ValueError("reference_addresses is empty")

        ref_prep = ref.map(self._prep)

        self.vectorizer = TfidfVectorizer(
            analyzer=self.analyzer,
            ngram_range=self.ngram_range,
        )
        ref_vec = self.vectorizer.fit_transform(ref_prep)

        self.nn = NearestNeighbors(
            n_neighbors=min(self.top_k, ref.shape[0]),
            metric="cosine",
            n_jobs=-1,
        ).fit(ref_vec)

        self._ref = ref
        self._ref_prep = ref_prep
        self._ref_vec = ref_vec
        return self

    def _check_fitted(self) -> None:
        if self.vectorizer is None or self.nn is None or self._ref is None:
            raise RuntimeError("AddressMatcher is not fitted. Call .fit(reference_addresses) first.")

    def match_one(self, query: str) -> MatchResult:
        self._check_fitted()

        q_raw = str(query)
        q = self._prep(q_raw)
        q_vec = self.vectorizer.transform([q])

        distances, indices = self.nn.kneighbors(q_vec)
        distances = distances.flatten()
        indices = indices.flatten()

        # cosine similarity
        cos_sims = 1.0 - distances

        # rerank by final score (cosine + fuzz)
        best_final = -1.0
        best_i = int(indices[0])
        best_cos = float(cos_sims[0])
        best_fuzz = 0.0

        for idx, cos in zip(indices, cos_sims):
            ref_raw = str(self._ref.iloc[int(idx)])
            ref_prep = str(self._ref_prep.iloc[int(idx)])
            f = fuzz.token_sort_ratio(q, ref_prep) / 100.0
            final = (self.w_cosine * float(cos) + self.w_fuzz * float(f)) / (self.w_cosine + self.w_fuzz)
            if final > best_final:
                best_final = final
                best_i = int(idx)
                best_cos = float(cos)
                best_fuzz = float(f)

        best_raw = str(self._ref.iloc[best_i])
        return MatchResult(
            query=q_raw,
            best=best_raw,
            cosine_sim=best_cos,
            fuzz_score=best_fuzz,
            final_score=float(best_final),
            best_index=best_i,
        )

    def match_one_topk(self, query: str, k: Optional[int] = None) -> pd.DataFrame:
        """Return top-k candidates with component scores.

        This is intended for interactive demos: besides the best match, it returns
        a table of alternatives with cosine/fuzzy/final scores.

        Columns: candidate, cosine_sim, fuzz_score, final_score, ref_index
        """
        self._check_fitted()

        k = int(k or self.top_k)
        if k < 1:
            k = 1

        q_raw = str(query)
        q = self._prep(q_raw)
        q_vec = self.vectorizer.transform([q])

        n_neighbors = min(k, int(self._ref.shape[0]))
        distances, indices = self.nn.kneighbors(q_vec, n_neighbors=n_neighbors)

        distances = distances.flatten()
        indices = indices.flatten()
        cos_sims = 1.0 - distances

        rows = []
        for idx, cos in zip(indices, cos_sims):
            idx = int(idx)
            ref_raw = str(self._ref.iloc[idx])
            ref_prep = str(self._ref_prep.iloc[idx])
            f = fuzz.token_sort_ratio(q, ref_prep) / 100.0
            final = (self.w_cosine * float(cos) + self.w_fuzz * float(f)) / (self.w_cosine + self.w_fuzz)
            rows.append(
                {
                    "candidate": ref_raw,
                    "cosine_sim": float(cos),
                    "fuzz_score": float(f),
                    "final_score": float(final),
                    "ref_index": idx,
                }
            )

        return pd.DataFrame(rows).sort_values("final_score", ascending=False).reset_index(drop=True)

    def match_batch(self, queries: Sequence[str]) -> pd.DataFrame:
        """Match a batch. Returns a DataFrame with columns:
        query, best, cosine_sim, fuzz_score, final_score, best_index
        """
        rows = [self.match_one(q) for q in queries]
        return pd.DataFrame([r.__dict__ for r in rows])


def evaluate_at_k(
    ranked_indices: np.ndarray,
    true_indices: np.ndarray,
    k: int,
) -> float:
    """Recall@k where each query has exactly one true index."""
    if k < 1:
        raise ValueError("k must be >= 1")
    k = min(k, ranked_indices.shape[1])
    hits = 0
    for i in range(ranked_indices.shape[0]):
        if int(true_indices[i]) in set(map(int, ranked_indices[i, :k])):
            hits += 1
    return hits / float(ranked_indices.shape[0]) if ranked_indices.shape[0] else 0.0
