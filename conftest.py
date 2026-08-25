# Empty on purpose: its presence makes pytest add the repo root to sys.path
# during collection, so `tests/*.py` can `import developer_profile` /
# `import python_bridge` / `import wrapper` regardless of how pytest is
# invoked (`pytest`, `.venv\Scripts\pytest.exe`, or `python -m pytest`) or
# whether the venv is activated.
