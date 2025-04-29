#!/bin/bash
set -e

# Define variables
PLUGIN_NAME="BasicSynthVST"
OUTPUT_DIR="target/release"
VST_BUNDLE_NAME="$PLUGIN_NAME.vst"
VST_BUNDLE_DIR="$OUTPUT_DIR/$VST_BUNDLE_NAME"
CONTENTS_DIR="$VST_BUNDLE_DIR/Contents"
MACOS_DIR="$CONTENTS_DIR/MacOS"
RESOURCES_DIR="$CONTENTS_DIR/Resources"

# Build the plugin for the current platform
echo "Building plugin (using Replit default cargo)..."
/nix/store/k38hhkbf0g3xgzsvnlabjhj3fyniz6za-cargo-1.77.1/bin/cargo build --release

# Create VST bundle directories
echo "Creating VST bundle structure..."
mkdir -p "$MACOS_DIR"
mkdir -p "$RESOURCES_DIR"

# Copy the compiled library to the bundle 
echo "Copying binary to VST bundle..."
# The exact filename depends on the platform
if [ -f "$OUTPUT_DIR/libbasic_synth_vst.so" ]; then
    cp "$OUTPUT_DIR/libbasic_synth_vst.so" "$MACOS_DIR/$PLUGIN_NAME"
elif [ -f "$OUTPUT_DIR/libbasic_synth_vst.dylib" ]; then
    cp "$OUTPUT_DIR/libbasic_synth_vst.dylib" "$MACOS_DIR/$PLUGIN_NAME"
elif [ -f "$OUTPUT_DIR/basic_synth_vst.dll" ]; then
    cp "$OUTPUT_DIR/basic_synth_vst.dll" "$MACOS_DIR/$PLUGIN_NAME"
else
    echo "Could not find compiled library. Please check the build output."
    exit 1
fi

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
zip -r "$PLUGIN_NAME.zip" "$VST_BUNDLE_NAME"

echo "Done! VST bundle created at $VST_BUNDLE_DIR"
echo "ZIP archive available at $OUTPUT_DIR/$PLUGIN_NAME.zip"