# E2E class scenario (design §11: scripted 12-student scenario — nightly)

Simulated-class runs against the service + FakeRunner (CPU):
12 uploads -> 12 tickets -> 12 submissions -> queue drains in expected
order -> kill/bump/requeue mid-drain -> demo-day 2-slot relay ->
backup + restore round-trip.

Not yet written — requires the service seams (tests/integration).
