#!/bin/bash
# Script to create a complete downloadable package for installation

set -e  # Exit on error

echo "Creating downloadable installation package..."

# Step 1: Build the release version for current platform
cargo build --release
echo "✓ Release build completed"

# Step 2: Create directory structure for the package
PACKAGE_DIR="./dist"
VERSION="1.0.0"
PACKAGE_NAME="BasicSynthVST-${VERSION}"
FULL_PACKAGE_DIR="${PACKAGE_DIR}/${PACKAGE_NAME}"

mkdir -p "${FULL_PACKAGE_DIR}"
mkdir -p "${FULL_PACKAGE_DIR}/VST2"
mkdir -p "${FULL_PACKAGE_DIR}/VST3"
mkdir -p "${FULL_PACKAGE_DIR}/Standalone"
mkdir -p "${FULL_PACKAGE_DIR}/Documentation"

echo "✓ Package directory structure created"

# Step 3: Copy the VST2 plugin
if [ -f ./target/release/libbasic_synth_vst.so ]; then
    # Linux
    mkdir -p "${FULL_PACKAGE_DIR}/VST2/Linux"
    cp ./target/release/libbasic_synth_vst.so "${FULL_PACKAGE_DIR}/VST2/Linux/"
    echo "✓ Linux VST2 plugin copied"
elif [ -f ./target/release/libbasic_synth_vst.dylib ]; then
    # macOS - create proper bundle
    mkdir -p "${FULL_PACKAGE_DIR}/VST2/macOS/BasicSynthVST.vst/Contents/MacOS"
    mkdir -p "${FULL_PACKAGE_DIR}/VST2/macOS/BasicSynthVST.vst/Contents/Resources"
    
    cp ./target/release/libbasic_synth_vst.dylib "${FULL_PACKAGE_DIR}/VST2/macOS/BasicSynthVST.vst/Contents/MacOS/BasicSynthVST"
    
    # Create Info.plist
    cat > "${FULL_PACKAGE_DIR}/VST2/macOS/BasicSynthVST.vst/Contents/Info.plist" << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>English</string>
    <key>CFBundleExecutable</key>
    <string>BasicSynthVST</string>
    <key>CFBundleIdentifier</key>
    <string>com.rustsynthbuilder.BasicSynthVST</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>BasicSynthVST</string>
    <key>CFBundlePackageType</key>
    <string>BNDL</string>
    <key>CFBundleSignature</key>
    <string>????</string>
    <key>CFBundleVersion</key>
    <string>1.0.0</string>
    <key>VSTPluginCategory</key>
    <string>Synth</string>
</dict>
</plist>
EOF
    
    # Create PkgInfo
    echo "BNDL????" > "${FULL_PACKAGE_DIR}/VST2/macOS/BasicSynthVST.vst/Contents/PkgInfo"
    
    echo "✓ macOS VST2 plugin bundle created"
elif [ -f ./target/release/basic_synth_vst.dll ]; then
    # Windows
    mkdir -p "${FULL_PACKAGE_DIR}/VST2/Windows"
    cp ./target/release/basic_synth_vst.dll "${FULL_PACKAGE_DIR}/VST2/Windows/"
    echo "✓ Windows VST2 plugin copied"
fi

# Step 4: Copy the standalone tester application
if [ -f ./target/release/test_synth ]; then
    cp ./target/release/test_synth "${FULL_PACKAGE_DIR}/Standalone/"
    echo "✓ Standalone tester application copied"
elif [ -f ./target/release/test_synth.exe ]; then
    cp ./target/release/test_synth.exe "${FULL_PACKAGE_DIR}/Standalone/"
    echo "✓ Standalone tester application copied"
fi

# Step 5: Copy documentation
cp README.md "${FULL_PACKAGE_DIR}/Documentation/"
cp TESTER_GUIDE.md "${FULL_PACKAGE_DIR}/Documentation/"

# Create installation instructions
cat > "${FULL_PACKAGE_DIR}/INSTALL.md" << 'EOF'
# BasicSynth VST Installation Guide

This package contains the BasicSynth VST plugin in various formats and a standalone tester application.

## VST2 Plugin Installation

### Windows
- Copy the `.dll` file from the `VST2/Windows` folder to your VST plugins folder
  (typically `C:\Program Files\VSTPlugins` or your DAW's custom VST folder)

### macOS
- Copy the `.vst` bundle from the `VST2/macOS` folder to:
  - `/Library/Audio/Plug-Ins/VST/` (system-wide installation)
  - `~/Library/Audio/Plug-Ins/VST/` (user-only installation)

### Linux
- Copy the `.so` file from the `VST2/Linux` folder to your VST plugins folder
  (typically `/usr/lib/vst` or `~/.vst`)

## Standalone Tester Usage

The standalone tester application in the `Standalone` folder allows you to test the synthesizer without a DAW.

### Basic Usage
```
./test_synth play
```

For more options and detailed usage instructions, refer to the Documentation folder.
EOF

echo "✓ Documentation copied"

# Step 6: Create a compressed archive
cd "${PACKAGE_DIR}"
if command -v zip >/dev/null 2>&1; then
    zip -r "${PACKAGE_NAME}.zip" "${PACKAGE_NAME}"
    echo "✓ Created ZIP archive: ${PACKAGE_DIR}/${PACKAGE_NAME}.zip"
elif command -v tar >/dev/null 2>&1; then
    tar -czf "${PACKAGE_NAME}.tar.gz" "${PACKAGE_NAME}"
    echo "✓ Created TAR archive: ${PACKAGE_DIR}/${PACKAGE_NAME}.tar.gz"
else
    echo "! Warning: Neither zip nor tar commands found. Archive not created."
fi

echo
echo "Installation package created successfully!"
echo "You can distribute the package from: ${PACKAGE_DIR}"
echo "Users can download and extract the archive to install the plugin."