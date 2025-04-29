# Basic Synth VST

A basic VST synthesizer plugin built with Rust featuring oscillators, ADSR envelope, and filters.

## Features

- Multiple oscillator types (Sine, Square, Saw, Triangle, Wavetable, FM)
- ADSR envelope for amplitude modulation
- Filter with multiple types (Low Pass, High Pass, Band Pass)
- Parameter controls for all synth components
- MIDI input support
- Virtual tester for DAW-free experimentation
- WAV recording capabilities
- Enhanced build tools for multiple platforms

## Building

### Prerequisites

- Rust and Cargo
- For cross-compilation: Rust targets for your target platforms

### Using the Makefile

The project includes a Makefile for easier building. Here are the available targets:

```bash
# Show all available targets
make help

# Build for current platform
make

# Build VST2 format for macOS ARM
make vst-arm

# Build VST3 format for macOS ARM
make vst3-arm

# Build both VST2 and VST3 formats for macOS ARM
make arm

# Build universal macOS binary (Intel + ARM)
make universal

# Validate the VST2 and VST3 bundles for macOS
make validate

# Clean all build artifacts
make clean

# Install the VST2 plugin locally for testing (macOS)
make install-macos

# Install the VST3 plugin locally for testing (macOS)
make install-macos-vst3

# Install both VST2 and VST3 plugins locally
make install-macos-all
```

### Building for macOS (ARM/Apple Silicon)

To build for macOS running on Apple Silicon (M1/M2):

```bash
# Install the Rust target
rustup target add aarch64-apple-darwin

# Run the build script
./build_macos_arm.sh
```

The VST plugin will be packaged as a .vst bundle in `target/aarch64-apple-darwin/release/` and also zipped for distribution.

#### Building Both VST2 and VST3 Formats

To build both VST2 (.vst) and VST3 (.vst3) formats for macOS ARM:

```bash
./build_all_macos_arm.sh
```

This will create a complete distribution package in `dist/BasicSynthVST-macOS-ARM-Complete.zip`.

### Building Universal macOS Binary (Intel + ARM)

To build a universal binary that works on both Intel and Apple Silicon Macs:

```bash
# Install the Rust targets
rustup target add aarch64-apple-darwin
rustup target add x86_64-apple-darwin

# Run the universal build script
./build_macos_universal.sh
```

The universal VST plugin will be packaged as a .vst bundle in `target/universal-apple-darwin/release/` and also zipped for distribution.

## Installation

### macOS

#### VST2 Format

Copy the `.vst` bundle to one of these locations:

- `/Library/Audio/Plug-Ins/VST/` (system-wide installation)
- `~/Library/Audio/Plug-Ins/VST/` (user-only installation)

#### VST3 Format

Copy the `.vst3` bundle to one of these locations:

- `/Library/Audio/Plug-Ins/VST3/` (system-wide installation)
- `~/Library/Audio/Plug-Ins/VST3/` (user-only installation)

After installation, restart your DAW to detect the new plugin.

## Usage

### Using as a VST Plugin in a DAW

1. Open your VST-compatible DAW
2. Add the "Basic Synth VST" as an instrument
3. Play notes via MIDI to generate sound
4. Adjust parameters to customize your sound:
   - Oscillator Type: Choose between sine, square, saw, triangle, wavetable, or FM waveforms
   - ADSR: Adjust attack, decay, sustain, and release settings
   - Filter: Select filter type and adjust cutoff and resonance
   - Master Gain: Control the overall volume

### Using the Virtual Tester (without a DAW)

The synthesizer includes a virtual tester that allows you to experiment with sounds without using a DAW:

```bash
# Play a simple middle C note with default settings
./target/debug/test_synth play

# Play a specific note with a specific oscillator type
./target/debug/test_synth play --note 69 --oscillator saw

# Record output to a WAV file
./target/debug/test_synth play --oscillator fm --record output.wav
```

For a complete guide on using the virtual tester, refer to the [TESTER_GUIDE.md](TESTER_GUIDE.md) file.

### Using the Improved Build and Export Tools

The project includes enhanced build and export tools for packaging the VST plugin:

```bash
# Generate the export script
./target/debug/test_synth export

# Run the comprehensive build script with options
./build_vst.sh --release --platform macos-arm --package
```

For detailed instructions on packaging and exporting, see the [TESTER_GUIDE.md](TESTER_GUIDE.md) file.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Development

To add new features or fix bugs:

1. Clone the repository
2. Make your changes
3. Run `cargo build` to test compilation
4. Use the provided build scripts to package for your target platforms