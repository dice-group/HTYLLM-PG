from __future__ import annotations

from collections import defaultdict
from typing import Dict


class _MetricStore:
    def __init__(self) -> None:
        self._totals: Dict[str, float] = defaultdict(float)
        self._weights: Dict[str, float] = defaultdict(float)

    def update(self, namespace: str, metrics: Dict[str, float], weight: float) -> None:
        weight = float(weight)
        if weight <= 0:
            weight = 1.0
        prefix = f"{namespace}/" if namespace else ""
        for key, value in metrics.items():
            if value is None:
                continue
            name = f"{prefix}{key}"
            self._totals[name] += float(value) * weight
            self._weights[name] += weight

    def pop(self) -> Dict[str, float]:
        aggregated = {}
        for name, total in self._totals.items():
            weight = self._weights.get(name, 0.0)
            if weight > 0:
                aggregated[name] = total / weight
        self._totals.clear()
        self._weights.clear()
        return aggregated


_STORE = _MetricStore()


def record_cola_metrics(metrics: Dict[str, float], weight: float) -> None:
    _STORE.update("cola", metrics, weight)


def record_hydralora_metrics(metrics: Dict[str, float], weight: float) -> None:
    _STORE.update("hydralora", metrics, weight)


def pop_tracked_metrics() -> Dict[str, float]:
    return _STORE.pop()
