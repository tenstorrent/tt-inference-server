# Experimental Gemma prefill source overlay

These two source snapshots are byte-identical to TT-Metal commit
`e2c286b743dcfe60fe72af81f0e9a4285b045f15`, under
`models/autoports/google_gemma_4_26b_a4b_it/tt/`. The `.source` suffix keeps
repository formatting tools from silently rewriting an attested snapshot.
They are mounted read-only onto their corresponding `.py` files, not imported
from this directory. Native TTNN and Metal remain from the reused image.

| File | SHA256 |
|---|---|
| precision_policy.source | 85b938b62097460adb7e77047e2931c20a6ec5bf8e80f75437379420fec7283c |
| multichip_decoder.source | 3c99fe9418ad474b97fdd065ac18bd1aaec22ac09d279a35fbb1d0114e0b02bc |

Tested native image: `sha256:ad58effd178b9d8c7689a392159532d30aea0c1fe24bec82c06dbc0d3ebf4bbd`,
Metal `c9ec3469f1b875e7e5e505660c4421e5126e8dad`, plugin
`c9cfebcf0490066ff85e1e3fba2c7d456ce5ce42`.
The associated unselected precision policy is
`reference_config/precision/gemma4_eval_prefill_bfp8.json`, SHA256
`902bb30c0bc114645706941d7b31ff4815fbe96cff39a247d652f8eac65a8227`.
This is composite experimental provenance, not an unmodified image, a native
rebuild, or a release precision selection. Model/SWE evidence lives in the
TT-Metal autoport's `doc/eval_speed/`.
