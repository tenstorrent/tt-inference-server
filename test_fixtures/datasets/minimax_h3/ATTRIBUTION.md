# Attribution — MiniMax-H3 benchmark asset pack

Every image, video, and audio file in this directory (committed or regenerated
by `make_assets.sh`) is derived from one of two CC-BY / public-domain sources.
No AI-generated or synthetic media is used anywhere in this pack — every
still, video, and prompt image is real footage or a real photograph, per the
2026-08-18 direction on AIA-581 ("realistic (non-animated) media").

## Sources

**Tears of Steel** — Blender Foundation short film, licensed
[CC BY 3.0](https://creativecommons.org/licenses/by/3.0/).
Mezzanine file: `tears_of_steel_720p.mov`,
<https://download.blender.org/demo/movies/ToS/tears_of_steel_720p.mov>
(1280x534, CC BY (c) Blender Foundation | mango.blender.org).
sha256 of the exact source file used: `efa9062d9cdb7a338e40ad530dfdf234806743f29ae6a1a136b97ece4e588e8f`
(372,178,639 bytes) — pinned in `make_assets.sh`.

**NASA image library** — public domain (17 U.S.C. § 105 — U.S. government
works are not subject to copyright), <https://images.nasa.gov>. Three photos
used, by their NASA image ID:
  - `iss007e07306` — Earth limb photographed from the ISS.
  - `iss009e24868` — hurricane photographed from the ISS.
  - `jsc2024e062537` — astronaut portrait, 6084x4867 (source for the large
    reference image; sha256 of the source file used:
    `3211a9da2b393e4bd080f667d40d3506f17a2946714a93fc703e42e7c5be73cb`,
    pinned in `make_assets.sh`).

## Per-file provenance

All video/image crops target 16:9 (the case matrix's fixed `aspect_ratio`).
Frame grabs and crops from Tears of Steel use a centered crop
(`crop=ih*16/9:ih`) before scaling, since the source is 1280x534 ("scope",
~2.39:1) — cropping the width down to a 16:9 box centered on the frame avoids
the horizontal stretch a bare aspect-ratio-changing scale would introduce.

| File | Committed or generated | Source | Derivation |
|---|---|---|---|
| `img_std_city_skyline.jpg` | committed | Tears of Steel | frame @ t=70s, centered-crop -> 1344x768 |
| `img_std_old_man_portrait.jpg` | committed | Tears of Steel | frame @ t=175s, centered-crop -> 1344x768 |
| `img_std_three_people.jpg` | committed | Tears of Steel | frame @ t=240s, centered-crop -> 1344x768 |
| `img_std_robot_street.jpg` | committed | Tears of Steel | frame @ t=300s, centered-crop -> 1344x768 |
| `img_std_young_man_canal.jpg` | committed | Tears of Steel | frame @ t=520s, centered-crop -> 1344x768 |
| `img_min_256px_scientist.jpg` | committed | Tears of Steel | frame @ t=400s, centered-crop, downscaled -> 456x256 |
| `img_std_earth_from_iss.jpg` | committed | NASA `iss007e07306` | centered-crop -> 1344x768 |
| `img_std_hurricane_from_iss.jpg` | committed | NASA `iss009e24868` | centered-crop -> 1344x768 |
| `img_max_27mb_astronaut.jpg` | **generated** | NASA `jsc2024e062537` | pure centered crop 6084x4867 -> 5760x3240 (no upscale) @ offset (162, 814), film-grain pass, JPEG q1 4:4:4 — see `make_assets.sh`'s `gen_img_max` |
| `vid_min_2s_city_skyline.mp4` | committed | Tears of Steel | 2.0s clip, centered-crop+scale -> 1280x720, H.264+AAC (same "city" scene as the still) |
| `vid_std_5s_man_talking.mp4` | committed | Tears of Steel | 5.0s clip, centered-crop+scale -> 1280x720, H.264+AAC |
| `vid_max_8s_47mb_robot_street.mp4` | **generated** | Tears of Steel | 8.0s clip @ t=298.0s, centered-crop+scale -> 1280x720, CBR ~48 Mbps H.264 + AAC 195 kbps — see `make_assets.sh`'s `gen_vid_max` |
| `vid_ns2_city.mp4` | **generated** | `vid_min_2s_city_skyline.mp4` | exact `-c:v copy -an` audio strip (byte-identical reproduction, no re-encoding) |
| `vid_ns5_man.mp4` | **generated** | `vid_std_5s_man_talking.mp4` | exact `-c:v copy -an` audio strip (byte-identical reproduction, no re-encoding) |
| `vid_ns7_robot.mp4` | **generated** | Tears of Steel | 7.083333s clip @ t=298.0s (same start point as `vid_max`), centered-crop+scale -> 1280x720, CBR ~48 Mbps H.264, silent — see `make_assets.sh`'s `gen_vid_ns7` |
| `aud_min_2s_score.wav` | committed | Tears of Steel soundtrack | 2s clip @ t=100s |
| `aud_std_5s_score.wav` | committed | Tears of Steel soundtrack | 5s clip @ t=360s |
| `aud_max_8s_score.wav` | committed | Tears of Steel soundtrack | 8s clip @ t=700s |
| `prompt_min.txt` / `prompt_std.txt` / `prompt_max_7000chars.txt` | committed | hand-written | T2VA/FL2VA/REF2VA text prompts at the min/std/max length bars |

## Reproducibility note on the "generated" files

`img_max_27mb_astronaut.jpg`, `vid_max_8s_47mb_robot_street.mp4`, and
`vid_ns7_robot.mp4` are re-encoded (JPEG / H.264+AAC) from the sources above.
`make_assets.sh`'s recipe for these three reproduces the correct crop,
resolution, timestamp, and nominal bitrate/quality — confirmed by decoding and
diffing the regenerated output against the pinned originals — but the exact
compressed bytes are NOT guaranteed to match `sha256s.txt` across different
ffmpeg builds/versions/platforms, since JPEG and CBR H.264 encoding are not
specified to be bit-reproducible across encoder implementations. This pack's
pinned hashes were produced with **ffmpeg 7.1.1** (`configuration:
--enable-libx264 ...`, Apple clang / macOS arm64 build). `make_assets.sh`
treats a hash mismatch on exactly these three files as an expected-drift
WARNING (not a failure) as long as an independent ffprobe resolution/duration
check still passes; a resolution or duration mismatch on any of them means the
recipe itself is wrong and is escalated to a hard failure.

`vid_ns2_city.mp4` and `vid_ns5_man.mp4` are plain stream copies
(`-c:v copy -an`, no re-encoding) of already-committed files and ARE
guaranteed byte-identical — verified against the pinned originals during
development of this pack.

## Manifest note

`sha256s.txt` in this directory was extended (2026-08-19, during the AIA-581
packaging work) to add `vid_ns2_city.mp4`, `vid_ns5_man.mp4`, and
`vid_ns7_robot.mp4` — the original pack manifest predated the "ns" (no-sound)
video variants that REF2VA-H needs (see the `_note` on the `REF2VA-H` case in
`st_engine/h3video_engine/cases.json` for why silent reference videos exist at
all: MiniMax-H3's "<=15s total audio" cap also counts audio embedded in
reference *videos*, not just standalone audio files). The three added hashes
are the real, verified hashes of the original proven asset pack's files, not
of this repo's regenerated approximations.
