# Basic Synthesizer VST Plugin

## Overview

This project is a VST synthesizer plugin written in Rust, designed to provide basic synthesis capabilities with oscillators, ADSR envelope, and filters. The project includes both a VST plugin library and a standalone test binary for development and testing purposes.

## System Architecture

The system follows a modular architecture with separate components for different aspects of audio synthesis:

- **Plugin Architecture**: Built using the VST 2.4 framework with Rust bindings
- **Audio Processing**: Real-time audio synthesis with configurable parameters
- **Testing Framework**: Standalone binary for testing synthesis without a DAW
- **GUI Support**: Planned integration with iced-based user interface (dependencies present but implementation pending)

## Key Components

### Core Synthesis Modules
- **Oscillators** (`src/oscillator.rs`): Generate basic waveforms (sine, square, sawtooth, triangle)
- **Envelope Generator** (`src/envelope.rs`): ADSR envelope for amplitude and filter modulation
- **Filters** (`src/filter.rs`): Audio filtering capabilities for tone shaping
- **DSP Utilities** (`src/dsp_utils.rs`): Common digital signal processing functions

### Plugin Infrastructure
- **Main Library** (`src/lib.rs`): VST plugin entry point and host interface
- **Parameters** (`src/parameters.rs`): Plugin parameter management and automation
- **Tester Module** (`src/tester.rs`): Testing utilities for audio output validation

### Audio Output and Testing
- **Test Binary** (`src/bin/test_synth.rs`): Standalone application for testing synthesis
- **Audio Backend**: CPAL for cross-platform audio output
- **File Export**: Hound library for WAV file generation

## Data Flow

1. **MIDI Input**: VST host sends MIDI events to the plugin
2. **Parameter Processing**: Plugin parameters are processed and applied to synthesis modules
3. **Audio Generation**: Oscillators generate basic waveforms based on MIDI note data
4. **Envelope Application**: ADSR envelope shapes the amplitude over time
5. **Filtering**: Audio signal passes through configurable filters
6. **Audio Output**: Processed audio is sent back to the VST host or audio device

## External Dependencies

### Audio Processing
- **VST Framework**: `vst` crate for VST 2.4 plugin development
- **Audio Output**: `cpal` for cross-platform audio device access
- **DSP Libraries**: `realfft` for frequency domain processing
- **File I/O**: `hound` for WAV file operations

### GUI Framework (Planned)
- **UI Framework**: `iced` and related crates for native GUI
- **Window Management**: `baseview` for plugin window embedding
- **Graphics**: `iced_wgpu` for GPU-accelerated rendering

### Utilities
- **Command Line**: `clap` for CLI argument parsing in test binary
- **Serialization**: `serde` and `serde_json` for preset management
- **Threading**: `atomic_float` for thread-safe parameter handling
- **Logging**: `log` crate for debug information

## Deployment Strategy

The project is configured for development in the Replit environment with:

- **Build System**: Cargo-based Rust build system
- **Target Platforms**: Primary focus on Linux (ALSA audio system)
- **Distribution**: VST plugin as dynamic library (.so file)
- **Testing**: Standalone binary for development testing

### Build Configuration
- **Library Output**: Both `cdylib` (for VST host) and `rlib` (for internal use)
- **Optimization**: Release builds use full optimization with LTO
- **Audio Dependencies**: ALSA system integration for Linux audio

The project includes comprehensive error suppression utilities (`src/suppress_warnings.rs`) to maintain clean build output during development.

## Changelog
- June 18, 2025. Initial setup

## User Preferences

Preferred communication style: Simple, everyday language.