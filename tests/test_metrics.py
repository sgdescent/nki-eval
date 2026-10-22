import torch

from nki_eval import metrics


def test_oracle_and_checks_are_outside_timer_and_pairs_share_values(monkeypatch):
    clock = [0]
    observed = {"candidate": [], "baseline": [], "oracle": []}
    check_output = metrics.check_output

    def slow_check(*args, **kwargs):
        clock[0] += 300_000_000
        return check_output(*args, **kwargs)

    def function(actor, duration):
        def call(x):
            observed[actor].append(x.clone())
            clock[0] += duration
            return -x

        return call

    monkeypatch.setattr(metrics.time, "perf_counter_ns", lambda: clock[0])
    monkeypatch.setattr(metrics, "check_output", slow_check)
    result = metrics.benchmark(
        function("candidate", 3_000_000),
        function("baseline", 5_000_000),
        function("oracle", 200_000_000),
        2,
        3,
        "test",
        {"inputs": [{"shape": [8]}]},
        99,
        {"rtol": 1e-4, "atol": 1e-5, "equal_nan": False},
    )
    assert result["candidate"]["samples"] == [3, 3, 3]
    assert result["reference"]["samples"] == [5, 5, 5]
    assert len(observed["oracle"]) == 5
    for candidate, baseline, oracle in zip(*observed.values()):
        assert torch.equal(candidate, baseline)
        assert torch.equal(candidate, oracle)
    assert not torch.equal(observed["candidate"][0], observed["candidate"][1])
