#!/bin/bash

# This script validates macOS VST2 and VST3 bundles

PLUGIN_NAME="BasicSynthVST"
TARGET="aarch64-apple-darwin"
OUTPUT_DIR="target/$TARGET/release"
VST2_BUNDLE="$OUTPUT_DIR/$PLUGIN_NAME.vst"
VST3_BUNDLE="$OUTPUT_DIR/$PLUGIN_NAME.vst3"

echo "=== Validating macOS VST Bundles ==="

# Check if VST2 bundle exists
if [ -d "$VST2_BUNDLE" ]; then
    echo "✅ VST2 bundle exists: $VST2_BUNDLE"
    
    # Check for executable
    if [ -f "$VST2_BUNDLE/Contents/MacOS/$PLUGIN_NAME" ]; then
        echo "✅ VST2 executable exists"
        
        # Check if executable is a valid Mach-O binary
        file_output=$(file "$VST2_BUNDLE/Contents/MacOS/$PLUGIN_NAME")
        if echo "$file_output" | grep -q "Mach-O"; then
            echo "✅ VST2 executable is a valid Mach-O binary"
            echo "   Architecture: $(echo "$file_output" | grep -o 'arm64\|x86_64')"
        else
            echo "❌ VST2 executable is not a valid Mach-O binary"
            echo "   File type: $file_output"
        fi
    else
        echo "❌ VST2 executable missing"
    fi
    
    # Check for Info.plist
    if [ -f "$VST2_BUNDLE/Contents/Info.plist" ]; then
        echo "✅ VST2 Info.plist exists"
    else
        echo "❌ VST2 Info.plist missing"
    fi
else
    echo "❌ VST2 bundle not found at: $VST2_BUNDLE"
fi

echo ""

# Check if VST3 bundle exists
if [ -d "$VST3_BUNDLE" ]; then
    echo "✅ VST3 bundle exists: $VST3_BUNDLE"
    
    # Check for executable
    if [ -f "$VST3_BUNDLE/Contents/MacOS/$PLUGIN_NAME" ]; then
        echo "✅ VST3 executable exists"
        
        # Check if executable is a valid Mach-O binary
        file_output=$(file "$VST3_BUNDLE/Contents/MacOS/$PLUGIN_NAME")
        if echo "$file_output" | grep -q "Mach-O"; then
            echo "✅ VST3 executable is a valid Mach-O binary"
            echo "   Architecture: $(echo "$file_output" | grep -o 'arm64\|x86_64')"
        else
            echo "❌ VST3 executable is not a valid Mach-O binary"
            echo "   File type: $file_output"
        fi
    else
        echo "❌ VST3 executable missing"
    fi
    
    # Check for Info.plist
    if [ -f "$VST3_BUNDLE/Contents/Info.plist" ]; then
        echo "✅ VST3 Info.plist exists"
    else
        echo "❌ VST3 Info.plist missing"
    fi
else
    echo "❌ VST3 bundle not found at: $VST3_BUNDLE"
fi

echo ""
echo "Validation complete!"