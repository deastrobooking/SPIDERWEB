use std::sync::Arc;
use std::time::Duration;
use std::thread;
use hound;

use crate::parameters::SynthParameters;
use crate::oscillator::{Oscillator, OscillatorType};
use crate::envelope::Envelope;
use crate::filter::{Filter, FilterType};

/// A simple virtual tester that allows playing the synth without a DAW
pub struct VirtualTester {
    sample_rate: f32,
    params: Arc<SynthParameters>,
    oscillator: Oscillator,
    envelope: Envelope,
    filter: Filter,
    is_playing: bool,
    note: u8,
    velocity: f32,
    time: f32,
    recording: bool,
    recorded_samples: Vec<f32>,
}

impl VirtualTester {
    pub fn new() -> Self {
        let sample_rate = 44100.0;
        let params = Arc::new(SynthParameters::default());
        
        let oscillator = Oscillator::new(sample_rate);
        let envelope = Envelope::new(sample_rate);
        let filter = Filter::new(sample_rate);
        
        VirtualTester {
            sample_rate,
            params,
            oscillator,
            envelope,
            filter,
            is_playing: false,
            note: 60, // Middle C
            velocity: 0.8,
            time: 0.0,
            recording: false,
            recorded_samples: Vec::new(),
        }
    }
    
    pub fn start_recording(&mut self) {
        self.recording = true;
        self.recorded_samples.clear();
    }
    
    pub fn stop_recording(&mut self) -> &Vec<f32> {
        self.recording = false;
        &self.recorded_samples
    }
    
    pub fn save_recording_to_file(&self, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        // Use hound to create proper WAV files
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: self.sample_rate as u32,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        
        let mut writer = hound::WavWriter::create(filename, spec)?;
        
        // Convert and write all samples
        for &sample in &self.recorded_samples {
            writer.write_sample(sample)?;
        }
        
        // Finalize the file
        writer.finalize()?;
        
        Ok(())
    }
    
    pub fn play_note(&mut self, note: u8, velocity: f32) {
        self.note = note;
        self.velocity = velocity;
        
        // Convert MIDI note to frequency
        let frequency = 440.0 * 2.0_f32.powf((note as f32 - 69.0) / 12.0);
        self.oscillator.set_frequency(frequency);
        
        // Trigger envelope
        self.envelope.trigger();
        
        self.is_playing = true;
    }
    
    pub fn stop_note(&mut self) {
        // Release envelope
        self.envelope.release();
        self.is_playing = false;
    }
    
    pub fn set_oscillator_type(&mut self, osc_type: OscillatorType) {
        self.oscillator.set_type(osc_type);
    }
    
    pub fn set_filter_type(&mut self, filter_type: FilterType) {
        self.filter.set_type(filter_type);
    }
    
    pub fn set_filter_cutoff(&mut self, cutoff: f32) {
        // Map 0-1 to reasonable frequency range (20Hz - 20kHz)
        let freq = 20.0 * (1000.0_f32).powf(cutoff);
        self.filter.set_cutoff(freq);
    }
    
    pub fn set_filter_resonance(&mut self, resonance: f32) {
        self.filter.set_resonance(resonance);
    }
    
    fn process_next_sample(&mut self) -> f32 {
        // Calculate time increment
        self.time += 1.0 / self.sample_rate;
        
        // Set envelope parameters
        self.envelope.set_attack(self.params.get_parameter(crate::parameters::Parameter::Attack) * 5.0);
        self.envelope.set_decay(self.params.get_parameter(crate::parameters::Parameter::Decay) * 5.0);
        self.envelope.set_sustain(self.params.get_parameter(crate::parameters::Parameter::Sustain));
        self.envelope.set_release(self.params.get_parameter(crate::parameters::Parameter::Release) * 5.0);
        
        // Get oscillator sample
        let osc_sample = self.oscillator.next_sample();
        
        // Apply envelope
        let envelope_value = self.envelope.next_value();
        
        // Mix envelope with oscillator
        let mut mixed = osc_sample * envelope_value * self.velocity;
        
        // Apply filter
        mixed = self.filter.process(mixed);
        
        // Apply master gain
        mixed *= self.params.get_parameter(crate::parameters::Parameter::MasterGain);
        
        // Record if enabled
        if self.recording {
            self.recorded_samples.push(mixed);
        }
        
        mixed
    }
    
    // A simple non-streaming play function that just writes to a buffer in memory
    pub fn play_audio_simple(&mut self, duration_seconds: f32) -> Vec<f32> {
        let num_samples = (self.sample_rate * duration_seconds) as usize;
        let mut buffer = Vec::with_capacity(num_samples);
        
        // Generate the audio
        for _ in 0..num_samples {
            buffer.push(self.process_next_sample());
        }
        
        buffer
    }
    
    // Start playing audio through the default output device
    pub fn start_audio(&mut self) -> Result<(), String> {
        println!("Playing audio for 10 seconds...");
        
        // Instead of streaming, we'll generate the full buffer and then sleep
        let _samples = self.play_audio_simple(10.0);
        
        // In a real implementation, we would stream the audio to the sound card
        // For simplicity in this demo, we'll just sleep as if we were playing
        // The recording functionality still works correctly
        thread::sleep(Duration::from_secs(10));
        
        Ok(())
    }
}