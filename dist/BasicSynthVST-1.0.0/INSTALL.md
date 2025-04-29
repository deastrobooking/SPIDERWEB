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
