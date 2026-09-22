# Execution security

nki-eval executes supplied Python and compiler code. Workers inherit the OS user's privileges,
filesystem access, environment variables, and network access. Process isolation prevents ordinary
crashes from taking down a caller and enables deadlines; it is not a sandbox.

Do not run unknown kernels in an environment containing credentials or private files. For
generated/untrusted submissions, provide a disposable worker VM or appropriately restricted
container, unprivileged user, resource limits, restricted network, and a read-only source mount.
Hardware device access must be limited to the worker's assigned Neuron devices. This repository
does not configure those controls for you.

Static candidate lint is a convenience to reject obvious high-level fallbacks. Python permits
dynamic behavior that can bypass it, mutate the oracle/evaluator, or forge results. A stronger
evaluation service needs separately protected reference outputs, an external timing/verdict
authority, resource isolation, and runtime native-execution verification. Do not trust v0.1
reports from adversarial code as a secure reward signal.

Report bugs via GitHub issues using minimal reproductions. Do not include credentials, private
source, environment dumps, or private compiler artifacts in public reports.
