#!/bin/bash
set -e

# Define variables
PLUGIN_NAME="BasicSynthVST"
TARGET="aarch64-apple-darwin"
OUTPUT_DIR="target/$TARGET/release"
VST3_BUNDLE_NAME="$PLUGIN_NAME.vst3"
VST3_BUNDLE_DIR="$OUTPUT_DIR/$VST3_BUNDLE_NAME"
CONTENTS_DIR="$VST3_BUNDLE_DIR/Contents"
MACOS_DIR="$CONTENTS_DIR/MacOS"
RESOURCES_DIR="$CONTENTS_DIR/Resources"

# Ensure Rust target is installed
echo "Installing Rust target for $TARGET..."
rustup target add $TARGET

# Build the plugin for macOS ARM
echo "Building plugin for $TARGET..."
cargo build --release --target $TARGET

# Create VST3 bundle directories
echo "Creating VST3 bundle structure..."
mkdir -p "$MACOS_DIR"
mkdir -p "$RESOURCES_DIR"

# Copy the compiled library to the bundle
echo "Copying binary to VST3 bundle..."
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
    <key>CSResourcesFileMapped</key>
    <true/>
    <key>VSTPluginCategory</key>
    <string>Synth</string>
</dict>
</plist>
EOF

# Create PkgInfo file
echo "Creating PkgInfo..."
echo "BNDL????" > "$CONTENTS_DIR/PkgInfo"

# ZIP the VST3 bundle for distribution
echo "Creating ZIP archive for distribution..."
cd "$OUTPUT_DIR"
zip -r "$PLUGIN_NAME-macOS-ARM-VST3.zip" "$VST3_BUNDLE_NAME"

echo "Done! macOS ARM VST3 bundle created at $VST3_BUNDLE_DIR"
echo "ZIP archive available at $OUTPUT_DIR/$PLUGIN_NAME-macOS-ARM-VST3.zip"