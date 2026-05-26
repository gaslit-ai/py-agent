"""Usage + pricing — provider-agnostic metric aggregation and cost estimation."""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict, Field


class Usage(BaseModel):
    """Provider/model-tagged metric record. Metrics are an open string→number map."""

    model_config = ConfigDict(frozen=True)

    provider: str
    model: str
    metrics: dict[str, float] = Field(default_factory=dict)

    def __add__(self, other: Usage) -> Usage:
        if self.provider != other.provider or self.model != other.model:
            raise ValueError(
                f"cannot add Usage across (provider, model): "
                f"{(self.provider, self.model)} vs {(other.provider, other.model)}"
            )
        merged: dict[str, float] = dict(self.metrics)
        for k, v in other.metrics.items():
            merged[k] = merged.get(k, 0.0) + v
        return Usage(provider=self.provider, model=self.model, metrics=merged)


class PricingRule(BaseModel):
    """Cost per `scale` units of `metric`. Example: per-1M input tokens at $0.25.

    rule = PricingRule(metric="input_tokens", per_unit=0.25, scale=1_000_000)
    """

    model_config = ConfigDict(frozen=True)

    metric: str
    per_unit: float
    scale: float = 1.0
    provider: str | None = None  # if set, rule only applies to this provider
    model: str | None = None  # if set, rule only applies to this model


class CostLine(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: str
    model: str
    metric: str
    quantity: float
    cost: float


class CostBreakdown(BaseModel):
    model_config = ConfigDict(frozen=True)

    lines: tuple[CostLine, ...] = ()
    total: float = 0.0


def aggregate_usage(usages: Iterable[Usage]) -> list[Usage]:
    """Sum metrics across runs, grouped by (provider, model)."""
    buckets: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    for u in usages:
        bucket = buckets[(u.provider, u.model)]
        for k, v in u.metrics.items():
            bucket[k] += v
    return [
        Usage(provider=provider, model=model, metrics=dict(metrics))
        for (provider, model), metrics in buckets.items()
    ]


def estimate_cost(
    usages: Iterable[Usage], pricing_rules: Iterable[PricingRule]
) -> CostBreakdown:
    """Apply pricing rules to a set of usage records. Skips metrics with no matching rule."""
    rules = list(pricing_rules)
    lines: list[CostLine] = []
    for u in usages:
        for metric, qty in u.metrics.items():
            rule = _find_rule(rules, u.provider, u.model, metric)
            if rule is None:
                continue
            cost = (qty / rule.scale) * rule.per_unit
            lines.append(
                CostLine(
                    provider=u.provider,
                    model=u.model,
                    metric=metric,
                    quantity=qty,
                    cost=cost,
                )
            )
    total = sum(line.cost for line in lines)
    return CostBreakdown(lines=tuple(lines), total=total)


def _find_rule(
    rules: list[PricingRule], provider: str, model: str, metric: str
) -> PricingRule | None:
    """Most-specific match wins: (provider+model+metric) > (provider+metric) > (metric)."""
    best: PricingRule | None = None
    best_specificity = -1
    for rule in rules:
        if rule.metric != metric:
            continue
        if rule.provider is not None and rule.provider != provider:
            continue
        if rule.model is not None and rule.model != model:
            continue
        specificity = (
            (1 if rule.provider is not None else 0)
            + (1 if rule.model is not None else 0)
        )
        if specificity > best_specificity:
            best = rule
            best_specificity = specificity
    return best
