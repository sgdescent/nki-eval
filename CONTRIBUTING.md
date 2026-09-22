# Contributing

Install `.[dev]`, run `ruff check .`, `ruff format --check .`, and `pytest -q` before opening a PR.
Add regression tests for changes to correctness, timing scope, failure handling, or SDK adapters.

Hardware changes should include the instance type, Neuron SDK and framework versions, command,
and a real integration result. Clearly distinguish mocked adapter tests from real SDK execution.
Never add CPU/simulator timings as a stand-in for Neuron device measurements.

Keep example kernels small and original. Document valid shape/dtype domains and numerical
tolerances. Preserve the correctness gate when adding backends, profilers, or agent integrations.
Discuss distributed execution and stronger isolation as separate features with explicit contracts.
