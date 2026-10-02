# MiniMax-H3 single-galaxy instances on AI& (tyo2-ome)

One Helm release of `charts/tt-inference-server` per BH galaxy (32 chips, mesh `(4, 8)`), each with its own
single-pod Service so async video jobs stay sticky to the pod that accepted them.

- `h3-1.yaml` … `h3-4.yaml`, `h3-dev.yaml`: base values (image, hostPath devices to bypass the tt-dra-driver 0.0.29
  CDI collapse, NFS HF cache `/data_bh/h3`, shared DiT cache, API key from Secret `h3-api-key`).
- `h3-N-pin.yaml`: per-node pin — `fullnameOverride: h3-N` + `kubernetes.io/hostname` nodeSelector.

Secrets are never in these files: the API key comes from Secret `h3-api-key` (key `API_KEY`), and the HF token is
passed at install time.

```
helm -n wan upgrade --install h3-1 charts/tt-inference-server \
  --set model=MiniMax-H3 --set device=galaxy --set engine=media \
  -f charts/tt-inference-server/values/minimax-h3-aibang/h3-1.yaml \
  -f charts/tt-inference-server/values/minimax-h3-aibang/h3-1-pin.yaml \
  --set hfToken="$HF_TOKEN"
```

**Caveat:** on this branch's chart, `defaults.extraEnv` (the `API_KEY` entry) is not rendered into the pod env — neither
the original plaintext form nor the Secret reference produces an `API_KEY` env var in `helm template`. Before
reinstalling from these files, confirm the API key actually reaches the server (or add it with `--set`), otherwise the
endpoint runs without auth.
