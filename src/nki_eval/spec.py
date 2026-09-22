"""Strict, JSON-serializable workload descriptions and deterministic input generation."""

import math

import torch

DTYPES = {
    name: getattr(torch, name)
    for name in (
        "float16",
        "bfloat16",
        "float32",
        "float64",
        "int8",
        "int16",
        "int32",
        "int64",
        "bool",
    )
}
DISTRIBUTIONS = {"normal", "uniform", "zeros", "ones", "randint"}


def _keys(obj, allowed, where):
    if not isinstance(obj, dict):
        raise ValueError(f"{where} must be an object")
    extra = obj.keys() - allowed
    if extra:
        raise ValueError(f"Unknown fields in {where}: {sorted(extra)}")


def validate(spec):
    _keys(spec, {"name", "cases", "rtol", "atol", "equal_nan"}, "workload")
    if not isinstance(spec.get("cases"), list) or not spec["cases"]:
        raise ValueError("cases must be a nonempty list")
    for key, default in (("rtol", 1e-4), ("atol", 1e-5)):
        value = spec.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{key} must be a finite nonnegative number")
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be a finite nonnegative number")
    if not isinstance(spec.get("equal_nan", False), bool):
        raise ValueError("equal_nan must be boolean")
    names = set()
    for i, case in enumerate(spec["cases"]):
        _keys(case, {"name", "inputs", "flops", "bytes_moved"}, f"case {i}")
        name = case.get("name", f"case_{i}")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("Case names must be unique nonempty strings")
        names.add(name)
        if not isinstance(case.get("inputs"), list) or not case["inputs"]:
            raise ValueError("Each case needs a nonempty inputs list")
        tensor_count = 0
        for inp in case["inputs"]:
            if isinstance(inp, dict):
                _keys(inp, {"shape", "dtype", "distribution", "low", "high", "scale"}, "input")
                shape = inp.get("shape")
                if not isinstance(shape, list) or any(type(x) is not int or x < 0 for x in shape):
                    raise ValueError("Tensor shape must be a list of nonnegative integers")
                if inp.get("dtype", "float32") not in DTYPES:
                    raise ValueError("Unsupported input dtype")
                dtype = DTYPES[inp.get("dtype", "float32")]
                dist = inp.get("distribution", "normal" if dtype.is_floating_point else "randint")
                if dist not in DISTRIBUTIONS:
                    raise ValueError(f"Unsupported distribution: {dist}")
                if not dtype.is_floating_point and dist in {"normal", "uniform"}:
                    raise ValueError("Integer/bool inputs require randint, zeros, or ones")
                for key in ("low", "high", "scale"):
                    if key in inp and (
                        type(inp[key]) not in (int, float) or not math.isfinite(inp[key])
                    ):
                        raise ValueError(f"{key} must be finite")
                low, high = inp.get("low", 0), inp.get("high", 2 if dtype == torch.bool else 10)
                if dist == "randint":
                    if type(low) is not int or type(high) is not int or low >= high:
                        raise ValueError("randint needs integer low < high (high is exclusive)")
                    bounds = (
                        (0, 1)
                        if dtype == torch.bool
                        else (
                            (torch.iinfo(dtype).min, torch.iinfo(dtype).max)
                            if not dtype.is_floating_point
                            else (-(2**53), 2**53)
                        )
                    )
                    if low < bounds[0] or high - 1 > bounds[1]:
                        raise ValueError("randint bounds exceed dtype range")
                if dist == "uniform" and inp.get("low", -1) >= inp.get("high", 1):
                    raise ValueError("uniform needs low < high")
                if not dtype.is_floating_point and inp.get("scale", 1) != 1:
                    raise ValueError("scale is only supported for floating-point inputs")
                tensor_count += 1
            elif inp is not None and type(inp) not in (int, float, bool, str):
                raise ValueError("Inputs must be tensor specs or JSON scalar constants")
            elif isinstance(inp, float) and not math.isfinite(inp):
                raise ValueError("Scalar inputs must be finite")
        if not tensor_count:
            raise ValueError("Each case must include at least one tensor")
        for key in ("flops", "bytes_moved"):
            if key in case and (
                type(case[key]) not in (int, float) or not math.isfinite(case[key]) or case[key] < 0
            ):
                raise ValueError(f"{key} must be finite and nonnegative")
    return spec


def inputs_for(case, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    values = []
    for inp in case["inputs"]:
        if not isinstance(inp, dict):
            values.append(inp)
            continue
        dtype = DTYPES[inp.get("dtype", "float32")]
        shape = inp["shape"]
        dist = inp.get("distribution", "normal" if dtype.is_floating_point else "randint")
        if dist in ("zeros", "ones"):
            tensor = getattr(torch, dist)(shape, dtype=dtype)
        elif dist == "randint":
            tensor = torch.randint(
                inp.get("low", 0),
                inp.get("high", 2 if dtype == torch.bool else 10),
                shape,
                generator=generator,
            ).to(dtype)
        elif dist == "uniform":
            tensor = torch.rand(shape, generator=generator, dtype=torch.float64)
            low, high = inp.get("low", -1), inp.get("high", 1)
            tensor = (tensor * (high - low) + low).to(dtype)
        else:
            tensor = torch.randn(shape, generator=generator, dtype=torch.float64).to(dtype)
        if dtype.is_floating_point:
            tensor = tensor * inp.get("scale", 1)
        values.append(tensor)
    return tuple(values)


def clone_inputs(inputs):
    return tuple(x.clone() if isinstance(x, torch.Tensor) else x for x in inputs)
