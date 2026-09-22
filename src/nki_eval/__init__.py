"""Public evaluation API. Each evaluation runs in a separate worker process."""

__version__ = "0.1.0"


def evaluate(*args, **kwargs):
    from .runner import evaluate as run

    return run(*args, **kwargs)


__all__ = ["evaluate", "__version__"]
