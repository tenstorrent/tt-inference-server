#!/usr/bin/env bash
# Blocks until the 1G hugepages that device DMA buffers pin into are actually allocated.
#
# Ordering After=tenstorrent-hugepages.service is not enough on its own: that unit is
# Type=simple, so systemd considers it started the moment hugepages-setup.sh is forked,
# not when the pages exist. UMD does not degrade gracefully on a short allocation -- it
# fails device init with "Failed to pin pages for hugepage at virtual address 0x0".
set -uo pipefail

HP=/sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages
MOUNT=/dev/hugepages-1G
DEADLINE=${HUGEPAGE_WAIT_SECONDS:-120}

# One 1G page per ASIC. Falling back to 1 keeps this a liveness check rather than a hard
# failure on hosts where lspci is not on the service user's PATH.
want=$(lspci -d 1e52: 2>/dev/null | wc -l)
[ "$want" -gt 0 ] || want=1

deadline=$((SECONDS + DEADLINE))
while [ "$SECONDS" -lt "$deadline" ]; do
  have=$(cat "$HP" 2>/dev/null || echo 0)
  if mountpoint -q "$MOUNT" && [ "$have" -ge "$want" ]; then
    echo "hugepages ready: ${have} x 1G, ${MOUNT} mounted"
    exit 0
  fi
  sleep 2
done

printf 'hugepages not ready after %ss: want %s, have %s, %s %s\n' \
  "$DEADLINE" "$want" "$(cat "$HP" 2>/dev/null || echo 0)" "$MOUNT" \
  "$(mountpoint -q "$MOUNT" && echo mounted || echo 'NOT mounted')" >&2
exit 1
