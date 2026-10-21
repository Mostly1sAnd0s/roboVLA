"""Pure core: no I/O, no network, no wall clock (time is injected).

Binding rules (design §11):
  * every module ships with unit tests, target >= 95% line coverage,
    whole unit suite < 60 s on CPU;
  * one JobSpec contract (spec.py), one config schema (config.py),
    one error type (errors.py), one state table (states.py);
  * no node names or IPs in code — hardware lives in fleet.yaml.
"""
