# Frozen SWE25 grouped and masked C16 candidate

Compare the frozen 25-task cohort used by control run40572176883 with all16
model slots exercised: resident batch16, block512, grouped and masked recurrence
enabled, C16. The immutable masked image is source543bac09167 and digest5f4e1cb6;
its isolated one-task serving smoke40579353560 completed normally with reward1,
9/9 complete HTTP responses and exact stable audits. This smoke is integration
evidence, not a general accuracy estimate.

The cohort, seed1234, medium reasoning, sampling, 16384-token request cap,
6300-second agent and3600-second request deadlines, isolation, model revision
and262144 native context match the frozen B8/C8 control. Only aggregate suite
timeout changes to14400 seconds for two6300-second waves plus1800 cleanup.
No individual budget is reduced. The bounded streaming audit exporter is
included; raw artifacts stay in CI/local storage.

Retain all25 attempted tasks, normal submissions, timeouts, verifier rewards,
actual isolation phases, complete/incomplete HTTP responses and output-token
accounting. C16 and model policies change together, so this measures a serving
configuration and cannot isolate one optimization's effect. The earlier B16/C8
SWE8 candidate had6/8 and two timeouts versus7/8 normal control submissions;
there is no existing SWE accuracy parity or wall-time improvement claim.
Flat prefill remains disabled and is absent from this immutable image.
