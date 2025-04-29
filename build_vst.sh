#!/bin/bash
# Comprehensive VST build script for multiple platforms

# Set error handling
set -e

# Function to display help
show_help() {
    echo "Basic Synth VST Build Script"
    echo "============================="
    echo "Usage: ./build_vst.sh [OPTIONS]"
    echo ""
    echo "OPTIONS:"
    echo "  --help                 Show this help message"
    echo "  --clean                Clean before building"
    echo "  --format               Format the code"
    echo "  --release              Build in release mode (default is debug)"
    echo "  --package              Package for distribution"
    echo "  --platform PLATFORM    Build for specific platform (default: current)"
    echo "                         Valid platforms: macos, macos-arm, linux, windows"
    echo "  --vst3                 Build VST3 instead of VST2"
    echo ""
    echo "Examples:"
    echo "  ./build_vst.sh --release --package          # Build release and package for current platform"
    echo "  ./build_vst.sh --platform macos-arm --vst3  # Build VST3 for macOS ARM"
}

# Default values
BUILD_TYPE="debug"
CLEAN=0
FORMAT=0
PACKAGE=0
PLATFORM="current"
VST_VERSION="vst2"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --help)
            show_help
            exit 0
            ;;
        --clean)
            CLEAN=1
            shift
            ;;
        --format)
            FORMAT=1
            shift
            ;;
        --release)
            BUILD_TYPE="release"
            shift
            ;;
        --package)
            PACKAGE=1
            shift
            ;;
        --platform)
            PLATFORM="$2"
            shift 2
            ;;
        --vst3)
            VST_VERSION="vst3"
            shift
            ;;
        *)
            echo "Error: Unknown option $1"
            show_help
            exit 1
            ;;
    esac
done

# Format the code if requested
if [ $FORMAT -eq 1 ]; then
    echo "Formatting code..."
    cargo fmt
fi

# Clean if requested
if [ $CLEAN -eq 1 ]; then
    echo "Cleaning build..."
    cargo clean
fi

# Determine target platform
TARGET=""
case $PLATFORM in
    macos)
        TARGET="--target x86_64-apple-darwin"
        ;;
    macos-arm)
        TARGET="--target aarch64-apple-darwin"
        ;;
    linux)
        TARGET="--target x86_64-unknown-linux-gnu"
        ;;
    windows)
        TARGET="--target x86_64-pc-windows-msvc"
        ;;
    current)
        TARGET=""
        ;;
    *)
        echo "Error: Unknown platform $PLATFORM"
        exit 1
        ;;
esac

# Set up build flags
BUILD_FLAGS=""
if [ "$BUILD_TYPE" = "release" ]; then
    BUILD_FLAGS="--release"
fi

# Build the VST plugin
echo "Building VST plugin for $PLATFORM platform..."
if [ -n "$TARGET" ]; then
    cargo build $BUILD_FLAGS $TARGET
else
    cargo build $BUILD_FLAGS
fi

# Package for distribution if requested
if [ $PACKAGE -eq 1 ]; then
    echo "Packaging VST plugin for distribution..."
    
    # Ensure the distribution directory exists
    mkdir -p dist
    
    # Determine platform-specific binary name and extension
    BINARY_PATH="./target/$BUILD_TYPE"
    BINARY_NAME="basic_synth_vst"
    EXTENSION=""
    
    case $PLATFORM in
        windows)
            EXTENSION=".dll"
            ;;
        macos | macos-arm)
            EXTENSION=".dylib"
            BINARY_NAME="lib$BINARY_NAME"
            ;;
        linux | current)
            EXTENSION=".so"
            BINARY_NAME="lib$BINARY_NAME"
            ;;
    esac
    
    # For macOS, create a bundle
    if [[ $PLATFORM == macos* || ($PLATFORM == "current" && $(uname) == "Darwin") ]]; then
        echo "Creating macOS VST bundle..."
        
        VST_BUNDLE_NAME="BasicSynthVST.vst"
        VST3_BUNDLE_NAME="BasicSynthVST.vst3"
        
        if [ "$VST_VERSION" = "vst3" ]; then
            BUNDLE_NAME=$VST3_BUNDLE_NAME
            BUNDLE_PATH="./dist/$BUNDLE_NAME"
            
            # Create VST3 bundle structure
            mkdir -p "$BUNDLE_PATH/Contents/MacOS"
            mkdir -p "$BUNDLE_PATH/Contents/Resources"
            
            # Copy binary
            cp "$BINARY_PATH/$BINARY_NAME$EXTENSION" "$BUNDLE_PATH/Contents/MacOS/BasicSynthVST"
            
            # Create Info.plist
            cat > "$BUNDLE_PATH/Contents/Info.plist" << EOF
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
            echo "BNDL????" > "$BUNDLE_PATH/Contents/PkgInfo"
            
        else
            BUNDLE_NAME=$VST_BUNDLE_NAME
            BUNDLE_PATH="./dist/$BUNDLE_NAME"
            
            # Create VST2 bundle structure
            mkdir -p "$BUNDLE_PATH/Contents/MacOS"
            mkdir -p "$BUNDLE_PATH/Contents/Resources"
            
            # Copy binary
            cp "$BINARY_PATH/$BINARY_NAME$EXTENSION" "$BUNDLE_PATH/Contents/MacOS/BasicSynthVST"
            
            # Create Info.plist
            cat > "$BUNDLE_PATH/Contents/Info.plist" << EOF
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
            echo "BNDL????" > "$BUNDLE_PATH/Contents/PkgInfo"
        fi
        
        echo "Created bundle at $BUNDLE_PATH"
        
        # Create a zip archive
        (cd ./dist && zip -r "BasicSynthVST-$PLATFORM-$VST_VERSION.zip" "$(basename "$BUNDLE_NAME")")
        echo "Created zip archive at ./dist/BasicSynthVST-$PLATFORM-$VST_VERSION.zip"
        
    else
        # For non-macOS platforms, just copy the binary to dist
        cp "$BINARY_PATH/$BINARY_NAME$EXTENSION" "./dist/"
        
        # Create a zip archive
        (cd ./dist && zip -r "BasicSynthVST-$PLATFORM-$VST_VERSION.zip" "$BINARY_NAME$EXTENSION")
        echo "Created zip archive at ./dist/BasicSynthVST-$PLATFORM-$VST_VERSION.zip"
    fi
fi

echo "Build completed successfully!"