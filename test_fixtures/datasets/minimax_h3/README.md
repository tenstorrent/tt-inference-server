# MiniMax-H3 benchmark input pack

The inputs the `MiniMaxH3BenchmarkTest` cases (`test_module/_test_common/minimax_h3_bench/cases.json`)
send to a deployment. In git: the three prompts (one scene at three lengths) and the two sha256
manifests. Not in git: the media -- 14 small files (8 images, 2 videos, 4 audio clips; 7.7 MB) and
5 large ones (121 MB). The media is staged, not regenerated from this repo; the t2va cases need only
the prompts.

## Staging the media

Copy the whole pack from the shared CI volume, `/mnt/MLPerf/tt-shield/persistent-volume/h3-assets`,
or from the team's h3-assets archive, into one directory (this one, or a directory you point the
test at, below). `make_assets.sh` is not a substitute: it regenerates only the 5 large files
(`img_max_27mb_astronaut.jpg`, `vid_max_8s_47mb_robot_street.mp4`, `vid_ns2_city.mp4`,
`vid_ns5_man.mp4`, `vid_ns7_robot.mp4`) from pinned public sources and needs the 14 small media
files already present next to it (`vid_min_2s_city_skyline.mp4` and `vid_std_5s_man_talking.mp4`
are its inputs for the two `vid_ns*` copies); it refuses to start otherwise. Three of the five are
not bit-reproducible across ffmpeg builds (see `ATTRIBUTION.md`).

## How the test finds a file

Every asset is looked up per file, in this order:

1. the explicit directory: the `assets_dir` target of the case / `--assets` on the command line;
2. `$H3_ASSETS`;
3. `/mnt/MLPerf/tt-shield/persistent-volume/h3-assets` (the shared volume on the hardware runners);
4. `$PERSISTENT_VOLUME_ROOT/h3-assets`;
5. `/localdev/persistent-volume/h3-assets` (tt-shield's volume where the shared mount is absent);
6. this directory.

Whichever copy a file comes from, its pin is this directory's `sha256s-bundle.txt`: every file a
selected case needs (and the smoke clip's, unless `skip_smoke`) is hashed against it before a
generation is started, and a missing or mismatching file fails the probe. A file in the explicit
directory is always the one used. Otherwise, when a file exists in more than one of places 2-6,
the first copy that matches its pin wins, so a stale staged copy never hides the pinned one further
down (for example media committed here); without the hash check (`verify_manifest: false` /
`--no-manifest`) it is simply the first copy found. The media is ignored by this repo's
`.gitignore`; committing it needs `git add -f` or a matching `!` line there.
