#!/usr/bin/env bash
# Fingerprint the compiled artifacts of a MiniMax-H3 tt-metal deployment so two
# machines (e.g. C12 bare metal vs Quad3 k8s) can be compared with `diff`.
#   usage: h3_fingerprint.sh <TT_METAL_HOME> <dir containing libtt_metal.so/_ttnn.so> [cache-key-dir]
#   cache-key-dir defaults to the newest ~/.cache/tt-metal-cache/<key>/ on this host.
# Run it ON the rank-0 host (the JIT cache is per host). Output is plain text on stdout.
set -u
MH=${1:?TT_METAL_HOME}; LIB=${2:?lib dir}; K=${3:-$(ls -td ~/.cache/tt-metal-cache/*/ 2>/dev/null | head -1)}
md5() { md5sum "$1" 2>/dev/null | cut -c1-32; }
echo "# host=$(hostname) date=$(date -u +%FT%TZ) TT_METAL_HOME=$MH lib=$LIB cache=$K"
echo "## host libraries (size md5)"
for f in libtt_metal.so libtt_stl.so _ttnncpp.so _ttnn.so _ttnn.cpython-*.so; do
  for p in $LIB/$f $LIB/../ttnn/ttnn/$f; do [ -f "$p" ] && printf "%-32s %10s %s\n" "$(basename $p)" "$(stat -c %s $p)" "$(md5 $p)"; done
done
echo "## compiler recorded in libtt_metal.so"
readelf -p .comment $LIB/libtt_metal.so 2>/dev/null | grep -oE '\[ *[0-9]+\] .*' | sed 's/\[ *[0-9]*\] *//' | sort -u | sed 's/^/  /'
echo "## sfpi"
grep -hoE 'SFPI_(RELEASE|VERSION)[^)]*|7\.[0-9]+\.[0-9]+' $MH/runtime/sfpi-version.cmake 2>/dev/null | head -2 | tr '\n' ' '; echo
$MH/runtime/sfpi/compiler/bin/riscv-tt-elf-g++ --version 2>/dev/null | head -1
echo "## runtime/hw/lib/blackhole objects"
(cd $MH/runtime/hw/lib/blackhole 2>/dev/null && for o in *.o; do printf "  %-20s %s\n" $o "$(md5 $o)"; done)
echo "## runtime/hw/toolchain/blackhole linker scripts"
(cd $MH/runtime/hw/toolchain/blackhole 2>/dev/null && for l in *.ld; do printf "  %-34s %s\n" $l "$(md5 $l)"; done)
echo "## JIT cache key dir: $(basename $K)"
echo "## firmware (<core>: md5 of <core>.elf | dephash of compile inputs)"
for d in $K/firmware/*/; do c=$(basename $d); printf "  %-28s elf=%s deps=%s\n" $c "$(md5 $d/$c.elf)" "$(tr -d '"' < $d/$c.elf.dephash 2>/dev/null | awk -F'\t' '{n=$1; sub(/.*\//,"",n); printf "%s=%s ", n, $2}')"; done
echo "## kernels: <kernel> <hash-dir> <md5 over all *.elf inside> <n elf>"
for name in exp_ring_joint_sdpa ring_joint_sdpa exp_ring_joint_reader ring_joint_reader exp_ring_joint_writer ring_joint_writer; do
  d=$K/kernels/$name; [ -d $d ] || { echo "  $name: (none)"; continue; }
  for h in $(ls $d | sort); do
    elfs=$(find $d/$h -name '*.elf' | sort)
    printf "  %-24s %-22s %s %s\n" $name $h "$(cat $elfs 2>/dev/null | md5sum | cut -c1-32)" "$(echo "$elfs" | grep -c elf)"
  done
done
echo "## kernel families present: $(ls $K/kernels | wc -l) dirs, $(find $K/kernels -mindepth 2 -maxdepth 2 -type d | wc -l) hash dirs total"
