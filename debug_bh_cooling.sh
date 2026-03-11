#!/bin/bash
# Debug BH-QB-GE thermal and clock issues

echo "================================"
echo "BH Device Thermal Diagnostics"
echo "================================"
echo ""

echo "1. Check if tt-smi is available"
echo "--------------------------------"
which tt-smi || echo "tt-smi not found in PATH"
echo ""

echo "2. Get device temperatures (if tt-smi available)"
echo "------------------------------------------------"
if command -v tt-smi &> /dev/null; then
    tt-smi -s || echo "Failed to get device status"
else
    echo "Skipping - tt-smi not available"
fi
echo ""

echo "3. Check fan speeds and cooling"
echo "--------------------------------"
echo "TODO: Add your specific cooling check commands here"
echo "Examples:"
echo "  - ipmitool sensor list | grep -i fan"
echo "  - sensors | grep -i fan"
echo ""

echo "4. Check AICLK stability test"
echo "-----------------------------"
echo "Run a simple device open/close to check for AICLK warnings:"
python3 << 'PYEOF'
import os
import sys

# Suppress most logs
os.environ["TT_METAL_LOGGER_LEVEL"] = "INFO"

try:
    import ttnn
    print("Opening device to check AICLK stability...")
    devices = ttnn.get_device_ids()
    print(f"Found devices: {devices}")

    if devices:
        # Just open device 0 to trigger AICLK init
        print("Opening device 0...")
        mesh = ttnn.open_mesh_device(
            mesh_shape=ttnn.MeshShape(1, 1),
            device_ids=ttnn.DeviceGrid(y=slice(0, 1), x=slice(0, 1)),
        )
        print("✓ Device opened successfully")
        print("Closing device...")
        ttnn.close_mesh_device(mesh)
        print("✓ Device closed successfully")
        print("\nNo AICLK issues detected (check for warnings above)")
    else:
        print("✗ No devices found!")
        sys.exit(1)

except Exception as e:
    print(f"✗ Error during device init: {e}")
    print("\nThis likely indicates thermal or clock stability issues")
    sys.exit(1)
PYEOF
echo ""

echo "5. Check for thermal throttling in dmesg"
echo "----------------------------------------"
dmesg | grep -i "thermal\|throttl" | tail -20 || echo "No recent thermal events"
echo ""

echo "6. Recommendations"
echo "------------------"
echo "If you see AICLK warnings:"
echo "  1. Improve cooling (check fans, airflow)"
echo "  2. Reduce ambient temperature"
echo "  3. Wait for chips to cool between runs"
echo "  4. Consider reducing clock speeds if persistent"
echo ""
echo "================================"
