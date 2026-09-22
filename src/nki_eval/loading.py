import ast
import hashlib
import importlib.util
import sys
from pathlib import Path


def split_entry(entry):
    path, symbol = entry.rsplit(":", 1)
    path = Path(path).resolve()
    if not path.is_file() or not symbol.isidentifier():
        raise ValueError(f"Expected an existing file.py:symbol, got {entry}")
    return path, symbol


def source_info(entry):
    path, symbol = split_entry(entry)
    return {
        "file": str(path),
        "symbol": symbol,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def lint_candidate(entry):
    """Conservative direct-import/JIT check. Not a security or anti-cheat boundary."""
    path, symbol = split_entry(entry)
    tree = ast.parse(path.read_text(), filename=str(path))
    violations = []
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                aliases[item.asname or item.name.split(".")[0]] = (
                    item.name if item.asname else item.name.split(".")[0]
                )
                if item.name != "math" and not (
                    item.name == "nki"
                    or item.name.startswith("nki.")
                    or item.name == "neuronxcc.nki"
                    or item.name.startswith("neuronxcc.nki.")
                ):
                    violations.append(f"line {node.lineno}: import {item.name} is not allowed")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for item in node.names:
                full = f"{module}.{item.name}"
                aliases[item.asname or item.name] = full
                if (
                    node.level
                    or item.name == "*"
                    or not (
                        module == "math"
                        or full == "neuronxcc.nki"
                        or module == "nki"
                        or module.startswith("nki.")
                        or module == "neuronxcc.nki"
                        or module.startswith("neuronxcc.nki.")
                    )
                ):
                    violations.append(f"line {node.lineno}: import {full} is not allowed")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in {"eval", "exec", "__import__", "compile", "open"}:
                violations.append(f"line {node.lineno}: {node.func.id} is not allowed")
    function = next(
        (x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == symbol), None
    )
    decorators = []
    if function:
        for decorator in function.decorator_list:
            expr = decorator.func if isinstance(decorator, ast.Call) else decorator
            name = ast.unparse(expr)
            first, _, rest = name.partition(".")
            decorators.append(aliases.get(first, first) + (f".{rest}" if rest else ""))
    if not any(x in {"nki.jit", "neuronxcc.nki.jit"} for x in decorators):
        violations.append("Entry point must be a top-level @nki.jit function")
    return violations


def load(entry):
    path, symbol = split_entry(entry)
    name = "_nki_eval_" + hashlib.sha256(str(path).encode()).hexdigest()[:16]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    value = getattr(module, symbol)
    if not callable(value):
        raise TypeError(f"{entry} is not callable")
    return value
