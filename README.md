# py-agent

A monorepo for two related Python packages:

- **[py-agent-lib](packages/py-agent-lib/)** — minimal, domain-agnostic DAG scheduler/executor for step-based workflows. Pydantic v2 throughout. The runtime substrate.
- **[py-agent-ner](packages/py-agent-ner/)** — NER training-data generation built on `py-agent-lib`. Bring your own labels, LLM, and downstream trainer.

Both publish independently to PyPI; both share this repository.

## Layout

```
py-agent/
├── packages/
│   ├── py-agent-lib/        # the runtime: types, plan, executor, adapters
│   └── py-agent-ner/        # the NER toolkit: models, pipeline, BIO conversion
├── examples/
│   └── quickstart/          # end-to-end NER demo (gemma4 via Ollama)
├── pyproject.toml           # uv workspace root (not published)
└── LICENSE                  # MIT, covers both packages
```

`examples/` lives at the repo root, not inside any package, because it
demonstrates the cross-package integration — you wouldn't ship an example
inside the library it imports from.

## Develop

```bash
git clone https://github.com/gaslit-ai/py-agent.git
cd py-agent
uv sync                       # installs both packages editable in one .venv
uv run pytest                 # runs all tests across both packages
uv run python examples/quickstart/run.py   # live demo (needs Ollama + gemma4)
```

Within the workspace, `py-agent-ner`'s dependency on `py-agent-lib` resolves
to the in-tree package, so you can change `py-agent-lib`'s API and immediately
see how `py-agent-ner` reacts in tests.

## Publish

Each package is published independently from this repo. Tag with a
package-name prefix:

```bash
# When py-agent-lib is ready for 0.1.0:
git tag lib-v0.1.0 && git push --tags

# When py-agent-ner is ready for 0.1.0:
git tag ner-v0.1.0 && git push --tags
```

The matching workflow under `.github/workflows/` builds and publishes the
right package. Each package has its own version number — they evolve
independently.

## Quick links

- Library API → [packages/py-agent-lib/README.md](packages/py-agent-lib/README.md)
- NER toolkit API → [packages/py-agent-ner/README.md](packages/py-agent-ner/README.md)
- Working demo → [examples/quickstart/README.md](examples/quickstart/README.md)
