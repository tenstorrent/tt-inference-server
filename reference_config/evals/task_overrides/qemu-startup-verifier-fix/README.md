# QEMU verifier A/B override

This local Harbor dataset contains the `terminal-bench/qemu-startup` task from
`terminal-bench/terminal-bench-2-1` at immutable package digest
`sha256:8e58263747da7dc688ad470688fb72825c4ba2c1c40443fa0f52963645bbd999`.

All task content is copied from that package (apart from nonsemantic whitespace
normalization) except `tests/test.sh`. The override removes its failing Debian
`apt-get` step and downloads the same pinned uv installer with `wget`, which is
already present in the task image.
