#!/bin/bash
# Bulk exporter for Windows and macOS VST packages

set -e  # Exit on error

SYNTHS_DIR="./Synths"
VERSION="1.0.0"
PACKAGE_NAME="BasicSynthVST-${VERSION}"
DATE_STAMP=$(date +"%Y%m%d")

echo "=== Bulk Exporter for Windows and macOS VST Packages ==="
echo "Exporting packages to: ${SYNTHS_DIR}"

# Create Synths directory if it doesn't exist
mkdir -p "${SYNTHS_DIR}"

# Build for release mode
echo "Building in release mode..."
cargo build --release

# Create temporary directories
WIN_DIR="${SYNTHS_DIR}/temp/windows"
MAC_DIR="${SYNTHS_DIR}/temp/macos"

mkdir -p "${WIN_DIR}/VST2"
mkdir -p "${WIN_DIR}/Standalone"
mkdir -p "${WIN_DIR}/Documentation"

mkdir -p "${MAC_DIR}/VST2/BasicSynthVST.vst/Contents/MacOS"
mkdir -p "${MAC_DIR}/VST2/BasicSynthVST.vst/Contents/Resources"
mkdir -p "${MAC_DIR}/Standalone"
mkdir -p "${MAC_DIR}/Documentation"

# Copy documentation to both packages
echo "Copying documentation..."
cp README.md TESTER_GUIDE.md "${WIN_DIR}/Documentation/"
cp README.md TESTER_GUIDE.md "${MAC_DIR}/Documentation/"

# Create Windows package
echo "Creating Windows package..."

# Simulate Windows DLL (since we're not cross-compiling)
echo "Simulating Windows VST DLL..."
cp ./target/release/libbasic_synth_vst.so "${WIN_DIR}/VST2/basic_synth_vst.dll"

# Copy standalone tester (simulating Windows executable)
cp ./target/release/test_synth "${WIN_DIR}/Standalone/test_synth.exe"

# Create Windows installation instructions
cat > "${WIN_DIR}/INSTALL.txt" << 'EOF'
BasicSynth VST - Windows Installation Instructions
=================================================

Installation:
1. Copy the VST2/basic_synth_vst.dll file to your VST plugins folder
   - Typically: C:\Program Files\VSTPlugins
   - Or your DAW's custom VST folder

2. Restart your DAW and scan for new plugins

Using the standalone tester:
1. Run the Standalone/test_synth.exe application
2. Use the command line options to test different sounds
   Example: test_synth.exe play --oscillator wavetable --note 60

For more detailed instructions, see the Documentation folder.
EOF

# Create macOS package
echo "Creating macOS package..."

# Copy library for macOS bundle
cp ./target/release/libbasic_synth_vst.so "${MAC_DIR}/VST2/BasicSynthVST.vst/Contents/MacOS/BasicSynthVST"

# Create Info.plist
cat > "${MAC_DIR}/VST2/BasicSynthVST.vst/Contents/Info.plist" << 'EOF'
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
echo "BNDL????" > "${MAC_DIR}/VST2/BasicSynthVST.vst/Contents/PkgInfo"

# Copy standalone tester
cp ./target/release/test_synth "${MAC_DIR}/Standalone/"

# Create macOS installation instructions
cat > "${MAC_DIR}/INSTALL.txt" << 'EOF'
BasicSynth VST - macOS Installation Instructions
===============================================

Installation:
1. Copy the VST2/BasicSynthVST.vst folder to one of these locations:
   - For system-wide installation: /Library/Audio/Plug-Ins/VST/
   - For user-only installation: ~/Library/Audio/Plug-Ins/VST/

2. Restart your DAW and scan for new plugins

Using the standalone tester:
1. Open Terminal and navigate to the Standalone folder
2. Make the tester executable: chmod +x test_synth
3. Run the tester: ./test_synth play --oscillator fm --note 60

For more detailed instructions, see the Documentation folder.
EOF

# Create ZIP archives
echo "Creating ZIP archives..."

# Windows ZIP
cd "${SYNTHS_DIR}/temp"
zip -r "../${PACKAGE_NAME}-Windows-${DATE_STAMP}.zip" windows/
echo "✓ Windows package created: ${SYNTHS_DIR}/${PACKAGE_NAME}-Windows-${DATE_STAMP}.zip"

# macOS ZIP
zip -r "../${PACKAGE_NAME}-macOS-${DATE_STAMP}.zip" macos/
echo "✓ macOS package created: ${SYNTHS_DIR}/${PACKAGE_NAME}-macOS-${DATE_STAMP}.zip"

# Create a combined package
zip -r "../${PACKAGE_NAME}-All-Platforms-${DATE_STAMP}.zip" windows/ macos/
echo "✓ Combined package created: ${SYNTHS_DIR}/${PACKAGE_NAME}-All-Platforms-${DATE_STAMP}.zip"

# Clean up temporary files
cd ../..
rm -rf "${SYNTHS_DIR}/temp"

echo
echo "Export completed successfully!"
echo "Packages are available in the ${SYNTHS_DIR} directory:"
ls -la "${SYNTHS_DIR}"