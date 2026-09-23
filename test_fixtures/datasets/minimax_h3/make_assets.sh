#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# make_assets.sh — regenerate the large MiniMax-H3 benchmark assets that are NOT
# committed to the repo, then verify every asset (committed + regenerated)
# against sha256s.txt.
#
# Why some assets aren't committed: img_max_27mb_astronaut.jpg (~27 MB),
# vid_max_8s_47mb_robot_street.mp4 (~47 MB), and the three silent reference
# videos vid_ns{2,5,7}_*.mp4 (~46 MB combined, dominated by vid_ns7_robot.mp4
# at ~42 MB) would roughly double this repo's data/ directory for files that
# are mechanically derivable from two CC-BY / public-domain sources. Every
# other asset the AIA-581 case matrix needs (7 standard 1344x768 images,
# img_min, the 2s/5s reference videos, all 3 audio clips, all 3 prompts) is
# small and committed as-is next to this script.
#
# Sources (see ATTRIBUTION.md in this directory for full attribution text):
#   - Tears of Steel (Blender Foundation, CC BY 3.0) — 720p mezzanine file,
#     used for every video-derived asset below.
#   - NASA image jsc2024e062537 (public domain, images.nasa.gov) — used for
#     img_max_27mb_astronaut.jpg.
#
# Usage:
#   ./make_assets.sh            # generate + verify everything
#   ./make_assets.sh --verify-only   # skip generation, just check hashes
#
# Requires: ffmpeg + ffprobe on PATH, curl, python3 (stdlib only, for the
# sha256 comparison table). No cluster access, no kubectl, no GPU.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

CACHE_DIR="$HERE/.cache"
SHA_FILE="$HERE/sha256s.txt"

TOS_URL="https://download.blender.org/demo/movies/ToS/tears_of_steel_720p.mov"
TOS_FILE="$CACHE_DIR/tears_of_steel_720p.mov"
# Pinned sha256 of the source mezzanine file, captured 2026-08-19. If Blender's
# mirror ever re-encodes this file, this check fails LOUDLY rather than silently
# cropping/scaling different footage at the same timestamps.
TOS_SHA256="efa9062d9cdb7a338e40ad530dfdf234806743f29ae6a1a136b97ece4e588e8f"

NASA_URL="https://images-assets.nasa.gov/image/jsc2024e062537/jsc2024e062537~orig.jpg"
NASA_FILE="$CACHE_DIR/jsc2024e062537.jpg"
NASA_SHA256="3211a9da2b393e4bd080f667d40d3506f17a2946714a93fc703e42e7c5be73cb"

VERIFY_ONLY="${1:-}"

log() { printf '[make_assets] %s\n' "$*"; }
warn() { printf '[make_assets] WARN: %s\n' "$*" >&2; }
die() { printf '[make_assets] ERROR: %s\n' "$*" >&2; exit 1; }

require_tool() {
    command -v "$1" >/dev/null 2>&1 || die "required tool '$1' not found on PATH"
}

sha256_of() {
    # macOS has shasum, not sha256sum; Linux (the python:3.12-slim pod target)
    # has sha256sum, not shasum. Support both without depending on either.
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | awk '{print $1}'
    else
        die "neither sha256sum nor shasum is available"
    fi
}

fetch_and_pin() {
    # fetch_and_pin <url> <dest> <expected_sha256> <label>
    local url="$1" dest="$2" expected="$3" label="$4"
    if [ -f "$dest" ]; then
        local have
        have="$(sha256_of "$dest")"
        if [ "$have" = "$expected" ]; then
            log "$label: already cached and verified ($dest)"
            return 0
        fi
        warn "$label: cached file at $dest has unexpected sha256 ($have) — re-downloading"
        rm -f "$dest"
    fi
    log "$label: downloading from $url"
    mkdir -p "$(dirname "$dest")"
    curl -fSL --retry 3 --retry-delay 5 -o "$dest" "$url"
    local got
    got="$(sha256_of "$dest")"
    if [ "$got" != "$expected" ]; then
        die "$label: downloaded file's sha256 ($got) does not match the pinned source hash " \
            "($expected) — the upstream file may have changed. Refusing to build assets from " \
            "an unverified source. See ATTRIBUTION.md."
    fi
    log "$label: downloaded and verified ($dest)"
}

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
# All three ToS-derived clips below start at the SAME timestamp (298.0s into
# tears_of_steel_720p.mov — the "MEMORY OVERWRITE" HUD shot that immediately
# precedes the canal-bridge robot close-up used for img_std_robot_street.jpg)
# and use the SAME crop: the source is 1280x534 ("scope", 2.39:1); cropping the
# width to a centered 16:9 box (crop=ih*16/9:ih, i.e. ih*16/9 x ih, x/y default
# to centered) before scaling to the 1280x720 target avoids stretching the
# image the way a bare scale from 2.39:1 to 16:9 would. This was confirmed by
# extracting vid_max_8s_47mb_robot_street.mp4's actual first frame and
# decoding-and-diffing it against candidate crop/scale/timestamp combinations
# from the pinned ToS source (mean abs pixel diff ~2.4/255 at t=298.0 — i.e.
# the ordinary lossy-recompression noise floor, not a wrong frame).
#
# CBR ~48 Mbps video (nominal target per the AIA-581 asset-pack notes; actual
# encoded bitrate lands a little under, same shape as the pinned files) + AAC
# 195 kbps 48 kHz stereo (resampled/re-encoded from the movie's own 44.1 kHz
# MP3 track — confirmed by decoding both to PCM and correlating: r=0.9997).
generate_video_clip() {
    # generate_video_clip <start_s> <duration_s> <out> [extra_ffmpeg_args...]
    local start="$1" dur="$2" out="$3"; shift 3
    ffmpeg -y -v error \
        -ss "$start" -i "$TOS_FILE" -t "$dur" \
        -vf "crop=ih*16/9:ih,scale=1280:720" \
        -c:v libx264 -b:v 48M -minrate 48M -maxrate 48M -bufsize 1M \
        -x264-params "nal-hrd=cbr:force-cfr=1" -pix_fmt yuv420p -r 24 \
        "$@" \
        "$out"
}

gen_vid_max() {
    [ -f "$HERE/vid_max_8s_47mb_robot_street.mp4" ] && { log "vid_max_8s_47mb_robot_street.mp4 exists, skipping generation"; return 0; }
    fetch_and_pin "$TOS_URL" "$TOS_FILE" "$TOS_SHA256" "Tears of Steel source"
    log "generating vid_max_8s_47mb_robot_street.mp4 (8.0s @ t=298.0, with audio)"
    generate_video_clip 298.0 8.0 "$HERE/vid_max_8s_47mb_robot_street.mp4" \
        -c:a aac -b:a 195k -ar 48000 -ac 2
}

gen_vid_ns7() {
    [ -f "$HERE/vid_ns7_robot.mp4" ] && { log "vid_ns7_robot.mp4 exists, skipping generation"; return 0; }
    fetch_and_pin "$TOS_URL" "$TOS_FILE" "$TOS_SHA256" "Tears of Steel source"
    log "generating vid_ns7_robot.mp4 (7.083333s @ t=298.0, silent — REF2VA-H's audio-total cap counts audio embedded in reference VIDEOS too, see docs/H3_VIDEO_BENCHMARK.md)"
    generate_video_clip 298.0 7.083333 "$HERE/vid_ns7_robot.mp4" -an
}

gen_vid_ns_from_committed() {
    # vid_ns2_city.mp4 and vid_ns5_man.mp4 are EXACT `-c:v copy -an` audio-strips
    # of the already-committed vid_min_2s_city_skyline.mp4 / vid_std_5s_man_talking.mp4
    # — verified byte-for-byte identical to the pinned assets, no download needed.
    if [ ! -f "$HERE/vid_ns2_city.mp4" ]; then
        log "generating vid_ns2_city.mp4 (audio-stripped copy of vid_min_2s_city_skyline.mp4)"
        ffmpeg -y -v error -i "$HERE/vid_min_2s_city_skyline.mp4" -c:v copy -an "$HERE/vid_ns2_city.mp4"
    else
        log "vid_ns2_city.mp4 exists, skipping generation"
    fi
    if [ ! -f "$HERE/vid_ns5_man.mp4" ]; then
        log "generating vid_ns5_man.mp4 (audio-stripped copy of vid_std_5s_man_talking.mp4)"
        ffmpeg -y -v error -i "$HERE/vid_std_5s_man_talking.mp4" -c:v copy -an "$HERE/vid_ns5_man.mp4"
    else
        log "vid_ns5_man.mp4 exists, skipping generation"
    fi
}

gen_img_max() {
    [ -f "$HERE/img_max_27mb_astronaut.jpg" ] && { log "img_max_27mb_astronaut.jpg exists, skipping generation"; return 0; }
    fetch_and_pin "$NASA_URL" "$NASA_FILE" "$NASA_SHA256" "NASA jsc2024e062537 source"
    # Source is 6084x4867 (already 4:4:4 chroma). Target is 5760x3240 (16:9) —
    # a PURE CENTERED CROP, no upscale: x=(6084-5760)/2=162, y=(4867-3240)/2
    # rounds to 814. Confirmed by brute-force pixel-diff search across a
    # +/-20px window around the naive centered offset: (162, 814) is the
    # unambiguous minimum. A film-grain pass (noise=alls=73) is applied before
    # JPEG re-encoding at quality 1 (max) — this is what the "+ grain" in the
    # asset-pack notes refers to, and pushes the file up toward the pinned
    # ~26.8 MB size (alls=73 landed within 0.3% of the pinned file's size in
    # testing; the exact noise seed/alls value used for the ORIGINAL pinned
    # file could not be recovered, so this is a faithful but best-effort
    # reproduction — see the verification step below).
    log "generating img_max_27mb_astronaut.jpg (centered crop 5760x3240 @ (162,814) + grain)"
    ffmpeg -y -v error -i "$NASA_FILE" \
        -vf "crop=5760:3240:162:814,noise=alls=73:allf=t+u,format=yuvj444p" \
        -q:v 1 -pix_fmt yuvj444p \
        "$HERE/img_max_27mb_astronaut.jpg"
}

# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
# Files that are committed to git as-is: a hash mismatch here means something
# is genuinely wrong (a corrupted checkout, an accidental edit) and is a hard
# failure. The 5 regenerated files are best-effort reproductions from public
# CC-BY / public-domain sources: img_max/vid_max/vid_ns7 depend on ffmpeg's
# exact JPEG/H.264 encoder implementation, which is NOT guaranteed to produce
# byte-identical output across ffmpeg versions or platforms even from an
# identical command line (this pack was built with ffmpeg 7.1.1). A hash
# mismatch on exactly those 3 files is expected drift, not corruption, and is
# reported as a WARNING as long as an independent ffprobe resolution/duration
# check (sanity_check_best_effort) still comes back correct — if THAT also
# fails, it is escalated to a hard failure, since that means the recipe
# itself is wrong, not just encoder-version hash drift. vid_ns2_city.mp4/
# vid_ns5_man.mp4 ARE deterministic (a plain stream
# copy with no re-encoding) and are held to the same hard-failure standard as
# the committed files.
BEST_EFFORT_FILES="img_max_27mb_astronaut.jpg vid_max_8s_47mb_robot_street.mp4 vid_ns7_robot.mp4"
# vid_ns2_city.mp4/vid_ns5_man.mp4 are NOT in this list: they are exact,
# deterministic `-c:v copy -an` strips of committed files with no re-encoding
# involved, so they are held to the same hard-failure standard as a committed
# file (see gen_vid_ns_from_committed()).

is_best_effort() {
    local f="$1" x
    for x in $BEST_EFFORT_FILES; do
        [ "$x" = "$f" ] && return 0
    done
    return 1
}

# expected_wh_dur <fname> -> prints "WIDTHxHEIGHT DURATION_S" (DURATION_S is
# "-" for the still image). Used to sanity-check a best-effort file that
# failed the hash check: a resolution/duration mismatch means the recipe is
# actually wrong (a real bug), not just encoder-version hash drift, and is
# escalated to a hard failure.
expected_wh_dur() {
    case "$1" in
        img_max_27mb_astronaut.jpg)      echo "5760x3240 -" ;;
        vid_max_8s_47mb_robot_street.mp4) echo "1280x720 8.0" ;;
        vid_ns7_robot.mp4)                echo "1280x720 7.083333" ;;
        *)                                echo "" ;;
    esac
}

sanity_check_best_effort() {
    # sanity_check_best_effort <path> <fname> -> 0 if resolution (and, for
    # video, duration within 0.5s) match; 1 otherwise. Never touches the hash.
    local path="$1" fname="$2" expected got_wh got_dur exp_wh exp_dur
    expected="$(expected_wh_dur "$fname")"
    [ -z "$expected" ] && return 0   # no spec on file — nothing to check
    exp_wh="${expected%% *}"
    exp_dur="${expected##* }"
    got_wh="$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height \
        -of csv=s=x:p=0 "$path" 2>/dev/null || true)"
    if [ "$got_wh" != "$exp_wh" ]; then
        printf '           SANITY FAIL: resolution %s != expected %s\n' "${got_wh:-<unreadable>}" "$exp_wh"
        return 1
    fi
    if [ "$exp_dur" != "-" ]; then
        got_dur="$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$path" 2>/dev/null || true)"
        # integer-seconds comparison (awk) is enough to catch "wrong clip length
        # entirely" without fighting float precision/locale issues in pure bash.
        if ! awk -v a="$got_dur" -v b="$exp_dur" 'BEGIN{d=a-b; if(d<0)d=-d; exit !(d<0.5)}' 2>/dev/null; then
            printf '           SANITY FAIL: duration %ss != expected %ss (+/-0.5s)\n' "${got_dur:-<unreadable>}" "$exp_dur"
            return 1
        fi
    fi
    return 0
}

verify_all() {
    [ -f "$SHA_FILE" ] || die "sha256s.txt not found at $SHA_FILE"
    local fail=0 warn_count=0 pass=0
    log "verifying every asset in $SHA_FILE ..."
    while read -r expected fname; do
        [ -z "${fname:-}" ] && continue
        local path="$HERE/$fname"
        if [ ! -f "$path" ]; then
            printf '  MISSING  %s\n' "$fname"
            fail=$((fail + 1))
            continue
        fi
        local got
        got="$(sha256_of "$path")"
        if [ "$got" = "$expected" ]; then
            printf '  PASS     %s\n' "$fname"
            pass=$((pass + 1))
        elif is_best_effort "$fname"; then
            local have_size
            have_size=$(wc -c < "$path" | tr -d ' ')
            if sanity_check_best_effort "$path" "$fname"; then
                printf '  WARN     %s  (hash differs from the pinned original; size on disk: %s bytes.\n' "$fname" "$have_size"
                printf '           This file is regenerated from a public source with ffmpeg — see the\n'
                printf '           comment above gen_img_max/gen_vid_max in this script for why an exact\n'
                printf '           byte match across ffmpeg versions/platforms is not guaranteed. Resolution\n'
                printf '           and, for video, duration were independently re-verified with ffprobe and\n'
                printf '           are correct — this is encoder-version hash drift, not a wrong recipe.)\n'
                warn_count=$((warn_count + 1))
            else
                printf '  FAIL     %s  (hash differs AND resolution/duration sanity check failed — this is\n' "$fname"
                printf '           a real recipe problem, not just encoder-version hash drift. See above.)\n'
                fail=$((fail + 1))
            fi
        else
            printf '  FAIL     %s  (hash differs — this file is expected to be committed verbatim\n' "$fname"
            printf '           or produced by a deterministic stream copy; a mismatch here means\n'
            printf '           something is genuinely wrong, not just encoder drift)\n'
            fail=$((fail + 1))
        fi
    done < "$SHA_FILE"
    log "verification summary: $pass pass, $warn_count best-effort warning(s), $fail failure(s)"
    if [ "$fail" -gt 0 ]; then
        die "$fail file(s) missing or unexpectedly mismatched — see above"
    fi
    if [ "$warn_count" -gt 0 ]; then
        log "note: $warn_count regenerated large file(s) did not hash-match the pinned originals exactly, but passed the resolution/duration sanity check (expected hash drift — see WARN detail above)."
    fi
    return 0
}

# ---------------------------------------------------------------------------
main() {
    require_tool ffmpeg
    require_tool ffprobe
    require_tool curl

    if [ "$VERIFY_ONLY" != "--verify-only" ]; then
        gen_vid_max
        gen_vid_ns7
        gen_vid_ns_from_committed
        gen_img_max
    else
        log "--verify-only: skipping generation"
    fi

    verify_all
    log "done."
}

main "$@"
