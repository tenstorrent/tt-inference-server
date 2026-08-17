#!/bin/bash
# Patches the Dynamo frontend binary to change owned_by from "nvidia" to "TT Inc"
# Run after: pip install ai-dynamo
# Limitation: replacement must be exactly 6 characters (same as "nvidia")

SO_FILE=".venv/lib/python3.12/site-packages/dynamo/_core.abi3.so"

if [ ! -f "$SO_FILE" ]; then
    echo "ERROR: $SO_FILE not found"
    exit 1
fi

python3 -c "
import sys
data = bytearray(open('$SO_FILE', 'rb').read())
old = b'\x66\xc7\x40\x04\x69\x61\xc7\x00\x6e\x76\x69\x64'
new = b'\x66\xc7\x40\x04\x6e\x63\xc7\x00\x54\x54\x20\x49'
count = 0
pos = 0
while True:
    idx = data.find(old, pos)
    if idx == -1: break
    data[idx:idx+len(old)] = new
    count += 1
    pos = idx + len(old)
if count:
    open('$SO_FILE', 'wb').write(data)
    print(f'Patched {count} occurrences: owned_by nvidia -> TT Inc')
else:
    # Check if already patched
    if data.find(new) != -1:
        print('Already patched.')
    else:
        print('ERROR: pattern not found', file=sys.stderr)
        sys.exit(1)
"
