#!/bin/bash
set -e

# Define variables
PLUGIN_NAME="BasicSynthVST"
TARGET_ARM="aarch64-apple-darwin"
TARGET_INTEL="x86_64-apple-darwin"
OUTPUT_DIR="target/universal-apple-darwin/release"
VST_BUNDLE_NAME="$PLUGIN_NAME.vst"
VST_BUNDLE_DIR="$OUTPUT_DIR/$VST_BUNDLE_NAME"
CONTENTS_DIR="$VST_BUNDLE_DIR/Contents"
MACOS_DIR="$CONTENTS_DIR/MacOS"
RESOURCES_DIR="$CONTENTS_DIR/Resources"

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$MACOS_DIR"
mkdir -p "$RESOURCES_DIR"

# Ensure Rust targets are installed
echo "Installing Rust targets..."
rustup target add $TARGET_ARM
rustup target add $TARGET_INTEL

# Build for ARM macOS
echo "Building for ARM (M1/M2) macOS..."
cargo build --release --target $TARGET_ARM

# Build for Intel macOS
echo "Building for Intel macOS..."
cargo build --release --target $TARGET_INTEL

# Create universal binary using lipo
echo "Creating universal binary..."
lipo -create -output "$OUTPUT_DIR/libbasic_synth_vst.dylib" \
    "target/$TARGET_ARM/release/libbasic_synth_vst.dylib" \
    "target/$TARGET_INTEL/release/libbasic_synth_vst.dylib"

# Copy the universal library to the bundle
echo "Copying universal binary to VST bundle..."
cp "$OUTPUT_DIR/libbasic_synth_vst.dylib" "$MACOS_DIR/$PLUGIN_NAME"

# Create Info.plist file
echo "Creating Info.plist..."
cat > "$CONTENTS_DIR/Info.plist" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>English</string>
    <key>CFBundleExecutable</key>
    <string>$PLUGIN_NAME</string>
    <key>CFBundleIdentifier</key>
    <string>com.yourcompany.$PLUGIN_NAME</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>$PLUGIN_NAME</string>
    <key>CFBundlePackageType</key>
    <string>BNDL</string>
    <key>CFBundleSignature</key>
    <string>????</string>
    <key>CFBundleVersion</key>
    <string>1.0</string>
    <key>VSTPluginCategory</key>
    <string>Synth</string>
</dict>
</plist>
EOF

# Create PkgInfo file
echo "Creating PkgInfo..."
echo "BNDL????" > "$CONTENTS_DIR/PkgInfo"

# ZIP the VST bundle for distribution
echo "Creating ZIP archive for distribution..."
cd "$OUTPUT_DIR"
zip -r "$PLUGIN_NAME-macOS-Universal.zip" "$VST_BUNDLE_NAME"

echo "Done! Universal macOS VST bundle created at $VST_BUNDLE_DIR"
echo "ZIP archive available at $OUTPUT_DIR/$PLUGIN_NAME-macOS-Universal.zip"