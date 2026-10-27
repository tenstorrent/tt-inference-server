# Stable audit export for larger SWE cohorts

The SWE8 recurrence audit was about 55 MB. The previous 128 MiB aggregate
input limit and whole-file buffering could prevent a valid larger evaluation
from retaining its final evidence. Export now validates JSONL and streams each
of the two exact files directly into the compressed ZIP, hashes the original
bytes, then independently rereads the source and verifies its identity, size,
timestamps and digest. Host archive validation also hashes members in chunks.
No records are filtered or rewritten.

Bounds are 2 GiB aggregate uncompressed source, 16 MiB per JSONL line, and
64 MiB compressed ZIP. The helper timeout is 300 seconds. Exceeding any bound
fails the export before emitting an archive; it does not truncate evidence.
These bounds are finite capacity, not a guarantee every SWE500 audit will fit.
Prefer independently attributed deterministic shards when total run time or
artifact volume exceeds a single job's capacity.

The helper retains its pinned base image, exact serving-image allowlist,
read-only source volume and root filesystem, no network or devices, exact owner
and Docker daemon identity checks, and verified cleanup. No device operation
is added. File stability does not prove that every request completed; the
serving interval and HTTP/trajectory accounting still require separate review.

Validation includes the actual stdlib collector on over 136 MiB of JSONL under
96 MiB of address space, source growth/truncation between reads, sparse input
above the limit, oversized line, compressed-size failure, missing files,
symlinks/FIFO, incomplete or invalid JSONL, tampered archive and attribution.
Generated fixtures, ZIPs and runtime logs remain outside Git.

Observed capacity check: stable audit archives for clean SWE8 control40554052639
contain55,672,460 raw bytes in2,218,555 compressed bytes; grouped candidate
40559423498 contains48,168,108 raw bytes in1,922,572 compressed bytes. Linear
100-task forecasts are respectively663.67/574.21 MiB raw and26.45/22.92 MiB
compressed, within the new bounds. These are volume forecasts, not guarantees:
check the frozen25 evidence before launching100, retain failures rather than
truncating records, and use independently attributed100-task shards for500.
