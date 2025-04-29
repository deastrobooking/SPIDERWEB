#!/bin/bash
set -e

echo "=== Building VST2 format for macOS ARM ==="
./build_macos_arm.sh

echo ""
echo "=== Building VST3 format for macOS ARM ==="
./build_macos_arm_vst3.sh

echo ""
echo "=== Creating combined distribution package ==="

PLUGIN_NAME="BasicSynthVST"
TARGET="aarch64-apple-darwin"
OUTPUT_DIR="target/$TARGET/release"
DIST_DIR="dist/macos-arm"

# Create distribution directory
mkdir -p "$DIST_DIR"

# Copy VST2 and VST3 bundles to dist directory
cp -r "$OUTPUT_DIR/$PLUGIN_NAME.vst" "$DIST_DIR/"
cp -r "$OUTPUT_DIR/$PLUGIN_NAME.vst3" "$DIST_DIR/"

# Copy README
cp README.md "$DIST_DIR/"

# Create a simple installation guide
cat > "$DIST_DIR/INSTALL.txt" << EOF
BasicSynthVST - Installation Guide
==============================

This package contains both VST2 (.vst) and VST3 (.vst3) formats of the plugin.

Installation:
------------

1. VST2 Format:
   Copy the "$PLUGIN_NAME.vst" folder to one of these locations:
   - /Library/Audio/Plug-Ins/VST/ (system-wide)
   - ~/Library/Audio/Plug-Ins/VST/ (user-only)

2. VST3 Format:
   Copy the "$PLUGIN_NAME.vst3" folder to one of these locations:
   - /Library/Audio/Plug-Ins/VST3/ (system-wide)
   - ~/Library/Audio/Plug-Ins/VST3/ (user-only)

After installation, restart your DAW to detect the new plugin.

Enjoy creating music with BasicSynthVST!
EOF

# Create distribution ZIP
cd dist
zip -r "$PLUGIN_NAME-macOS-ARM-Complete.zip" "macos-arm"

echo "=== Build Complete ==="
echo "VST2 and VST3 bundles are available in the $DIST_DIR directory"
echo "Complete distribution package: dist/$PLUGIN_NAME-macOS-ARM-Complete.zip"