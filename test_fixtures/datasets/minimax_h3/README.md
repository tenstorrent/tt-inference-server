# MiniMax-H3 benchmark input pack

The inputs the `MiniMaxH3BenchmarkTest` cases (`test_module/_test_common/minimax_h3_bench/cases.json`)
send to a deployment. The prompts (three lengths of one scene) and the sha256 manifests are
committed here; the media files (13 small images/clips/audio, 7.8 MB, plus five large ones,
121 MB) are not, because the t2va cases need only the prompts.

Staging the media, in the order the test looks (`host.resolve_assets_dir`):

1. `assets_dir` target / `H3_ASSETS` -- an explicit directory holding the whole pack;
2. `/mnt/MLPerf/tt-shield/persistent-volume/h3-assets` -- the shared volume on the hardware runners;
3. this directory.

Copy the pack from `quad-agent/h3-benchmark/h3-assets/` or regenerate it with `make_assets.sh`
(pinned Blender "Tears of Steel" and NASA sources; three files are not bit-reproducible across
ffmpeg builds, see `ATTRIBUTION.md`). Every file a selected case needs is checked against
`sha256s-bundle.txt` before a generation is started.
