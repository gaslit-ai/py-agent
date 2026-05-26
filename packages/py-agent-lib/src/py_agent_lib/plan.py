"""Plan + PlanBuilder. Validation: unique ids, known deps, no self-deps, acyclic."""
from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from graphlib import CycleError, TopologicalSorter
from typing import Any

from .errors import PlanValidationError
from .types import Step


class Plan:
    """Immutable validated DAG of Steps."""

    __slots__ = ("_steps",)

    def __init__(self, steps: Mapping[str, Step] | Iterable[Step]) -> None:
        if isinstance(steps, Mapping):
            normalized: dict[str, Step] = {}
            for key, step in steps.items():
                if key != step.id:
                    raise PlanValidationError(
                        f"step key {key!r} does not match step.id {step.id!r}"
                    )
                normalized[step.id] = step
        else:
            normalized = {}
            for step in steps:
                if step.id in normalized:
                    raise PlanValidationError(f"duplicate step id {step.id!r}")
                normalized[step.id] = step

        _validate_dependencies(normalized)
        _validate_acyclic(normalized)
        self._steps: Mapping[str, Step] = normalized

    @property
    def steps(self) -> Mapping[str, Step]:
        return self._steps

    def get_step(self, step_id: str) -> Step:
        try:
            return self._steps[step_id]
        except KeyError:
            raise PlanValidationError(f"unknown step {step_id!r}") from None

    def __iter__(self) -> Iterator[Step]:
        return iter(self._steps.values())

    def __len__(self) -> int:
        return len(self._steps)

    def __contains__(self, step_id: object) -> bool:
        return isinstance(step_id, str) and step_id in self._steps


class PlanBuilder:
    """Mutable builder for Plans. Validation happens on `.build()`."""

    def __init__(self) -> None:
        self._steps: dict[str, Step] = {}

    def add_step(
        self,
        *,
        id: str,
        action: str,
        deps: Iterable[str] = (),
        payload: Any = None,
    ) -> Step:
        if id in self._steps:
            raise PlanValidationError(f"duplicate step id {id!r}")
        step = Step(id=id, action=action, deps=tuple(deps), payload=payload)
        self._steps[id] = step
        return step

    def build(self) -> Plan:
        return Plan(self._steps)


def _validate_dependencies(steps: Mapping[str, Step]) -> None:
    ids = set(steps)
    for step in steps.values():
        for dep in step.deps:
            if dep == step.id:
                raise PlanValidationError(f"step {step.id!r} cannot depend on itself")
            if dep not in ids:
                raise PlanValidationError(
                    f"step {step.id!r} depends on unknown step {dep!r}"
                )


def _validate_acyclic(steps: Mapping[str, Step]) -> None:
    ts: TopologicalSorter[str] = TopologicalSorter()
    for step in steps.values():
        ts.add(step.id, *step.deps)
    try:
        ts.prepare()
    except CycleError as e:
        cycle = " -> ".join(str(n) for n in e.args[1])
        raise PlanValidationError(f"cycle detected in plan: {cycle}") from None
