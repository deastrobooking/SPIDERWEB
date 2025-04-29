# Basic Synth VST - Virtual Tester & Export Guide

This document provides instructions on how to use the virtual tester feature for testing the synth without a DAW, and how to export the VST plugin for distribution.

## Virtual Tester

The virtual tester allows you to test and experiment with the synthesizer without requiring a Digital Audio Workstation (DAW). It's useful for development, debugging, and quick sound design experiments.

### Command-Line Interface

The tester provides a command-line interface with the following options:

```
Usage: test_synth play [OPTIONS]

Options:
  -n, --note <note>              MIDI note number (0-127, 60 = middle C) [default: 60]
  -v, --velocity <velocity>      Note velocity (0.0-1.0) [default: 0.8]
  -d, --duration <duration>      Note duration in seconds [default: 2.0]
  -o, --oscillator <oscillator>  Oscillator type (sine, square, saw, triangle, wavetable, fm) [default: sine]
  -f, --filter <filter>          Filter type (lowpass, highpass, bandpass) [default: lowpass]
  -c, --cutoff <cutoff>          Filter cutoff (0.0-1.0) [default: 1.0]
  -r, --resonance <resonance>    Filter resonance (0.0-1.0) [default: 0.1]
      --attack <attack>          Attack time in seconds (0.0-5.0) [default: 0.1]
      --release <release>        Release time in seconds (0.0-5.0) [default: 0.3]
      --record <record>          Record output to WAV file
  -h, --help                     Print help
```

### Usage Examples

1. Play a simple middle C note:
   ```
   ./target/debug/test_synth play
   ```

2. Play a specific note with different oscillator:
   ```
   ./target/debug/test_synth play --note 69 --oscillator wavetable
   ```

3. Create a custom sound with filter and envelope settings:
   ```
   ./target/debug/test_synth play --oscillator fm --filter highpass --cutoff 0.3 --resonance 0.7 --attack 0.5 --release 1.0
   ```

4. Record a sound to a WAV file:
   ```
   ./target/debug/test_synth play --oscillator saw --record output.wav
   ```

## VST Export and Packaging

The synthesizer can be packaged as a VST plugin for use in different DAWs across multiple platforms.

### Using the Export Command

The export command generates a script that builds and packages the VST plugin:

```
./target/debug/test_synth export
```

This will create an `export_plugin.sh` script in the project directory.

### Running the Export Script

After generating the export script, make it executable and run it:

```
chmod +x export_plugin.sh
./export_plugin.sh
```

This will:
1. Build the VST plugin in release mode
2. Create the appropriate bundle structure (for macOS)
3. Package it into a zip file for distribution

### Building for Different Platforms

To build for specific platforms, use the `build_vst.sh` script with the appropriate options:

```
./build_vst.sh --release --platform macos-arm --package
```

Available options:
- `--clean`: Clean build files before building
- `--format`: Format the code
- `--release`: Build in release mode
- `--package`: Create distributable package
- `--platform`: Specify target platform (macos, macos-arm, linux, windows)
- `--vst3`: Build VST3 instead of VST2

### Installation in a DAW

After building, you can install the VST plugin in your DAW:

1. For macOS:
   - Copy `./target/release/BasicSynthVST.vst` to `~/Library/Audio/Plug-Ins/VST/`

2. For Windows:
   - Copy `./target/release/basic_synth_vst.dll` to your VST plugin directory

3. For Linux:
   - Copy `./target/release/libbasic_synth_vst.so` to your VST plugin directory

## Troubleshooting

1. Audio not playing:
   - Ensure audio output device is properly configured
   - Try different oscillator types or parameter settings

2. Build errors:
   - Make sure all dependencies are installed
   - Run `cargo clean` and try building again

3. VST loading issues in DAW:
   - Verify the plugin is in the correct format for your OS
   - Check DAW plugin scanning logs for errors