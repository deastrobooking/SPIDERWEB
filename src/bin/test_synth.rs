extern crate basic_synth_vst;
extern crate clap;
extern crate hound;

use basic_synth_vst::tester::VirtualTester;
use basic_synth_vst::oscillator::OscillatorType;
use basic_synth_vst::filter::FilterType;

use clap::{Arg, Command};
use std::thread;
use std::time::Duration;
use std::fs::File;
use std::io::Write;

const VERSION: &str = env!("CARGO_PKG_VERSION");

fn main() {
    let matches = Command::new("Basic Synth VST Tester")
        .version(VERSION)
        .author("Your Name")
        .about("Test the Basic Synth VST without a DAW")
        .subcommand(Command::new("play")
            .about("Play a note")
            .arg(Arg::new("note")
                .short('n')
                .long("note")
                .help("MIDI note number (0-127, 60 = middle C)")
                .default_value("60"))
            .arg(Arg::new("velocity")
                .short('v')
                .long("velocity")
                .help("Note velocity (0.0-1.0)")
                .default_value("0.8"))
            .arg(Arg::new("duration")
                .short('d')
                .long("duration")
                .help("Note duration in seconds")
                .default_value("2.0"))
            .arg(Arg::new("oscillator")
                .short('o')
                .long("oscillator")
                .help("Oscillator type (sine, square, saw, triangle, wavetable, fm)")
                .default_value("sine"))
            .arg(Arg::new("filter")
                .short('f')
                .long("filter")
                .help("Filter type (lowpass, highpass, bandpass)")
                .default_value("lowpass"))
            .arg(Arg::new("cutoff")
                .short('c')
                .long("cutoff")
                .help("Filter cutoff (0.0-1.0)")
                .default_value("1.0"))
            .arg(Arg::new("resonance")
                .short('r')
                .long("resonance")
                .help("Filter resonance (0.0-1.0)")
                .default_value("0.1"))
            .arg(Arg::new("attack")
                .long("attack")
                .help("Attack time in seconds (0.0-5.0)")
                .default_value("0.1"))
            .arg(Arg::new("release")
                .long("release")
                .help("Release time in seconds (0.0-5.0)")
                .default_value("0.3"))
            .arg(Arg::new("record")
                .long("record")
                .help("Record output to WAV file")))
        .subcommand(Command::new("export")
            .about("Export VST plugin to different platforms"))
        .get_matches();
    
    match matches.subcommand() {
        Some(("play", play_matches)) => {
            play_synth(play_matches);
        },
        Some(("export", _)) => {
            export_plugin();
        },
        _ => {
            println!("No command specified. Use --help for more information.");
        }
    }
}

fn play_synth(matches: &clap::ArgMatches) {
    let note = matches.get_one::<String>("note").unwrap().parse::<u8>().unwrap_or(60);
    let velocity = matches.get_one::<String>("velocity").unwrap().parse::<f32>().unwrap_or(0.8);
    let duration = matches.get_one::<String>("duration").unwrap().parse::<f32>().unwrap_or(2.0);
    
    let osc_type = match matches.get_one::<String>("oscillator").unwrap().as_str() {
        "sine" => OscillatorType::Sine,
        "square" => OscillatorType::Square,
        "saw" => OscillatorType::Saw,
        "triangle" => OscillatorType::Triangle,
        "wavetable" => OscillatorType::Wavetable,
        "fm" => OscillatorType::FM,
        _ => OscillatorType::Sine,
    };
    
    let filter_type = match matches.get_one::<String>("filter").unwrap().as_str() {
        "lowpass" => FilterType::LowPass,
        "highpass" => FilterType::HighPass,
        "bandpass" => FilterType::BandPass,
        _ => FilterType::LowPass,
    };
    
    let cutoff = matches.get_one::<String>("cutoff").unwrap().parse::<f32>().unwrap_or(1.0);
    let resonance = matches.get_one::<String>("resonance").unwrap().parse::<f32>().unwrap_or(0.1);
    
    println!("Playing note {} with {} oscillator", note, matches.get_one::<String>("oscillator").unwrap());
    println!("Filter: {} with cutoff {} and resonance {}", 
        matches.get_one::<String>("filter").unwrap(), cutoff, resonance);
    
    let mut tester = VirtualTester::new();
    tester.set_oscillator_type(osc_type);
    tester.set_filter_type(filter_type);
    tester.set_filter_cutoff(cutoff);
    tester.set_filter_resonance(resonance);
    
    // If recording is requested, start recording
    if let Some(record_file) = matches.get_one::<String>("record") {
        tester.start_recording();
        println!("Recording to file: {}", record_file);
    }
    
    // Play the note
    tester.play_note(note, velocity);
    
    // Start audio output
    match tester.start_audio() {
        Ok(_) => println!("Audio started successfully"),
        Err(e) => println!("Error starting audio: {}", e),
    }
    
    // Wait for the specified duration
    thread::sleep(Duration::from_secs_f32(duration));
    
    // Stop the note
    tester.stop_note();
    
    // Wait for release to complete
    let release_time = matches.get_one::<String>("release").unwrap().parse::<f32>().unwrap_or(0.3);
    thread::sleep(Duration::from_secs_f32(release_time * 1.5)); // a bit extra to ensure release completes
    
    // If recording, save to WAV file
    if let Some(record_file) = matches.get_one::<String>("record") {
        match tester.save_recording_to_file(record_file) {
            Ok(_) => println!("Recording saved to: {}", record_file),
            Err(e) => println!("Error saving recording: {}", e),
        }
    }
}

fn export_plugin() {
    println!("Exporting plugin...");
    
    // Platform-specific export logic would go here
    // For now, we'll just create a script to build for different platforms
    
    let script = r#"#!/bin/bash
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
"#;
    
    // Write the script to a file
    let mut file = File::create("export_plugin.sh").expect("Failed to create export script");
    file.write_all(script.as_bytes()).expect("Failed to write export script");
    
    // Make the script executable
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata("export_plugin.sh").expect("Failed to get file metadata").permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions("export_plugin.sh", perms).expect("Failed to set permissions");
    }
    
    println!("Export script created: export_plugin.sh");
    println!("Run this script to build and package the VST plugin for distribution.");
}