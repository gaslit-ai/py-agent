# Publishing

Both packages publish to PyPI from this monorepo, independently, via
**PyPI trusted publishing** (OIDC). No long-lived tokens.

## One-time PyPI setup

For each package, register the workflow as a pending publisher on PyPI:

1. Go to <https://pypi.org/manage/account/publishing/>.
2. Click **Add a new pending publisher**.
3. Fill in:

   | Field            | `py-agent-lib`        | `py-agent-ner`        |
   | ---------------- | --------------------- | --------------------- |
   | PyPI project     | `py-agent-lib`        | `py-agent-ner`        |
   | Owner            | `gaslit-ai`           | `gaslit-ai`           |
   | Repository       | `py-agent`            | `py-agent`            |
   | Workflow         | `publish-lib.yml`     | `publish-ner.yml`     |
   | Environment      | `pypi`                | `pypi`                |

4. Save.

Then in GitHub: **Settings → Environments → New environment → `pypi`**
(used by both publish workflows). No secrets are needed — OIDC handles auth.

## Cutting a release

Each package is versioned independently. Version lives in **one place**
per package: `src/<pkg>/__init__.py` (`__version__ = "..."`). Hatch reads
it dynamically; `pyproject.toml` has `dynamic = ["version"]`.

### `py-agent-lib`

```bash
# 1. Bump the version in packages/py-agent-lib/src/py_agent_lib/__init__.py
# 2. Commit
git commit -am "lib: 0.1.1"

# 3. Tag with the lib- prefix and push
git tag lib-v0.1.1
git push origin main lib-v0.1.1
```

The `publish-lib.yml` workflow fires on the tag, verifies the tag matches
`__version__`, builds sdist + wheel, and uploads to PyPI.

### `py-agent-ner`

Same, with the `ner-` prefix:

```bash
# 1. Bump packages/py-agent-ner/src/py_agent_ner/__init__.py
git commit -am "ner: 0.1.1"
git tag ner-v0.1.1
git push origin main ner-v0.1.1
```

## Local dry-run

Before tagging, build locally to make sure the package is healthy:

```bash
cd packages/py-agent-lib   # or py-agent-ner
uv build --sdist --wheel
ls dist/
```

You can also smoke-test the wheel in a clean venv:

```bash
uv venv /tmp/dryrun
/tmp/dryrun/bin/pip install dist/py_agent_lib-*.whl
/tmp/dryrun/bin/python -c "import py_agent_lib; print(py_agent_lib.__version__)"
```
