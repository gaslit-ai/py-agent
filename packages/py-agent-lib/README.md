# py-agent-lib

Minimal, domain-agnostic DAG scheduler/executor for step-based workflows. It runs a plan of steps with dependencies, supports cancellation and event streaming, and stays decoupled from any specific domain (LLM, agent, etc.).

## Install

```bash
pip install py-agent-lib
# or
uv add py-agent-lib
```

## 5-minute example

```python
import asyncio
from py_agent_lib import DagExecutor, PlanBuilder, StepStatus, StepResult

builder = PlanBuilder()
builder.add_step(id="pick-files", action="pick")
builder.add_step(id="edit-files", action="edit", deps=["pick-files"])
plan = builder.build()

async def pick(step, ctx):
    return StepResult(step_id=step.id, status=StepStatus.COMPLETED, output=["a.py", "b.py"])

async def edit(step, ctx):
    files = ctx.get_dependency_outputs()["pick-files"]
    return StepResult(step_id=step.id, status=StepStatus.COMPLETED, output={"edited": len(files)})

handlers = {"pick": pick, "edit": edit}

async def main():
    executor = DagExecutor(max_parallel_steps=4)
    state = await executor.execute(plan, handlers)
    print(state.steps["edit-files"].output)

asyncio.run(main())
```

## Status

Alpha. Core API stable, adapters and examples in progress.

## Design

- **Validation everywhere**: Pydantic v2 models at every boundary.
- **Structured concurrency**: `asyncio.TaskGroup` + `graphlib.TopologicalSorter` instead of a hand-rolled scheduler.
- **Retries**: `tenacity` for the heavy lifting.
- **Adapters live outside the core**: LLM clients (Pydantic AI, Instructor), observers (OpenTelemetry, NDJSON), persistence — all separate.
