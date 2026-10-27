# E2E class scenario (design section 11: scripted 12-student scenario, nightly)

Simulated-class runs against the service + FakeRunner (CPU). The flow:
12 uploads -> 12 tickets -> 12 submissions. The queue drains in expected
order. Kill/bump/requeue mid-drain. Demo-day 2-slot relay.
Backup + restore round-trip.

Not yet written. It requires the service seams (tests/integration).
