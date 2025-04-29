#!/bin/bash
# Build script for multiple platforms

echo "Building VST for current platform..."
cargo build --release

# Create VST bundle structure
mkdir -p ./target/release/BasicSynthVST.vst/Contents/MacOS
mkdir -p ./target/release/BasicSynthVST.vst/Contents/Resources

# Copy the binary to the bundle
if [ -f ./target/release/libbasic_synth_vst.so ]; then
    cp ./target/release/libbasic_synth_vst.so ./target/release/BasicSynthVST.vst/Contents/MacOS/BasicSynthVST
elif [ -f ./target/release/libbasic_synth_vst.dylib ]; then
    cp ./target/release/libbasic_synth_vst.dylib ./target/release/BasicSynthVST.vst/Contents/MacOS/BasicSynthVST
elif [ -f ./target/release/basic_synth_vst.dll ]; then
    cp ./target/release/basic_synth_vst.dll ./target/release/BasicSynthVST.vst/Contents/MacOS/BasicSynthVST
else
    echo "Could not find compiled library. Build may have failed."
    exit 1
fi

# Create Info.plist
cat > ./target/release/BasicSynthVST.vst/Contents/Info.plist << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>English</string>
    <key>CFBundleExecutable</key>
    <string>BasicSynthVST</string>
    <key>CFBundleIdentifier</key>
    <string>com.yourcompany.BasicSynthVST</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>BasicSynthVST</string>
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

# Create PkgInfo
echo "BNDL????" > ./target/release/BasicSynthVST.vst/Contents/PkgInfo

echo "VST bundle created at ./target/release/BasicSynthVST.vst"
echo "You can copy this bundle to your VST plugins folder."

# Create a ready-to-distribute zip file
cd ./target/release
zip -r BasicSynthVST.zip BasicSynthVST.vst
echo "Distribution package created at ./target/release/BasicSynthVST.zip"
