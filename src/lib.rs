#[macro_use]
extern crate vst;
extern crate log;

mod oscillator;
mod envelope;
mod filter;
mod parameters;
// Temporarily comment out UI module since we've disabled UI dependencies
// mod ui;
mod dsp_utils;

use vst::api::{Events, Supported};
use vst::buffer::AudioBuffer;
use vst::event::Event;
use vst::plugin::{Category, HostCallback, Info, Plugin, PluginParameters};
use vst::util::AtomicFloat;

use parameters::{SynthParameters, Parameter};
use oscillator::{Oscillator, OscillatorType};
use envelope::Envelope;
use filter::{Filter, FilterType};

use std::sync::Arc;
use std::collections::HashMap;
use atomic_float::AtomicF32;
use std::sync::RwLock;

const MAX_VOICES: usize = 16;

struct Voice {
    active: bool,
    note: u8,
    velocity: f32,
    oscillator: Oscillator,
    envelope: Envelope,
}

impl Voice {
    fn new() -> Self {
        Voice {
            active: false,
            note: 0,
            velocity: 0.0,
            oscillator: Oscillator::new(44100.0),
            envelope: Envelope::new(44100.0),
        }
    }

    fn start(&mut self, note: u8, velocity: f32) {
        self.active = true;
        self.note = note;
        self.velocity = velocity;
        
        // Convert MIDI note to frequency (A4 = 69 = 440Hz)
        let frequency = 440.0 * 2.0_f32.powf((note as f32 - 69.0) / 12.0);
        self.oscillator.set_frequency(frequency);
        self.envelope.trigger();
    }

    fn stop(&mut self) {
        self.envelope.release();
    }

    fn is_active(&self) -> bool {
        self.active && !self.envelope.is_idle()
    }

    fn next_sample(&mut self, params: &SynthParameters) -> f32 {
        if !self.is_active() {
            self.active = false;
            return 0.0;
        }

        // Get oscillator value based on current parameters
        let osc_type = match params.get_parameter(Parameter::OscillatorType) {
            x if x < 0.25 => OscillatorType::Sine,
            x if x < 0.5 => OscillatorType::Square,
            x if x < 0.75 => OscillatorType::Saw,
            _ => OscillatorType::Triangle,
        };
        self.oscillator.set_type(osc_type);

        // Get raw oscillator sample
        let sample = self.oscillator.next_sample();

        // Apply envelope
        self.envelope.set_attack(params.get_parameter(Parameter::Attack) * 5.0);
        self.envelope.set_decay(params.get_parameter(Parameter::Decay) * 5.0);
        self.envelope.set_sustain(params.get_parameter(Parameter::Sustain));
        self.envelope.set_release(params.get_parameter(Parameter::Release) * 5.0);
        
        let envelope_value = self.envelope.next_value();
        
        // The final output is oscillator sample * envelope * velocity
        sample * envelope_value * self.velocity
    }
}

struct BasicSynth {
    sample_rate: f32,
    params: Arc<SynthParameters>,
    voices: Vec<Voice>,
    filter: Filter,
    note_to_voice: HashMap<u8, usize>,
}

impl BasicSynth {
    fn new(host: HostCallback) -> Self {
        let params = Arc::new(SynthParameters::default());
        
        let mut voices = Vec::with_capacity(MAX_VOICES);
        for _ in 0..MAX_VOICES {
            voices.push(Voice::new());
        }

        BasicSynth {
            sample_rate: 44100.0,
            params,
            voices,
            filter: Filter::new(44100.0),
            note_to_voice: HashMap::new(),
        }
    }

    fn process_midi_event(&mut self, event: &Event) {
        match event {
            Event::Midi(midi_event) => {
                let message = midi_event.data;
                let status = message[0] & 0xF0; // Extract the status byte

                match status {
                    // Note On
                    0x90 => {
                        let note = message[1];
                        let velocity = message[2] as f32 / 127.0;
                        
                        // Note on with velocity 0 is equivalent to note off
                        if velocity > 0.0 {
                            self.note_on(note, velocity);
                        } else {
                            self.note_off(note);
                        }
                    }
                    // Note Off
                    0x80 => {
                        let note = message[1];
                        self.note_off(note);
                    }
                    // Controller Change
                    0xB0 => {
                        // CC messages can be processed here
                        let controller = message[1];
                        let value = message[2] as f32 / 127.0;
                        // Handle CC messages if needed
                    }
                    // Pitch Bend
                    0xE0 => {
                        // Process pitch bend if needed
                        let lsb = message[1] as u16;
                        let msb = message[2] as u16;
                        let bend = ((msb << 7) | lsb) as f32 / 8192.0 - 1.0;
                        // Apply pitch bend to active voices
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }

    fn note_on(&mut self, note: u8, velocity: f32) {
        // First check if this note is already playing and stop it
        if let Some(&voice_idx) = self.note_to_voice.get(&note) {
            self.voices[voice_idx].stop();
            self.note_to_voice.remove(&note);
        }

        // Find a free voice
        for (i, voice) in self.voices.iter_mut().enumerate() {
            if !voice.is_active() {
                voice.start(note, velocity);
                self.note_to_voice.insert(note, i);
                return;
            }
        }

        // If all voices are active, steal the oldest one
        // For simplicity, we'll just take the first one in the list
        if !self.voices.is_empty() {
            self.voices[0].start(note, velocity);
            self.note_to_voice.insert(note, 0);
        }
    }

    fn note_off(&mut self, note: u8) {
        if let Some(&voice_idx) = self.note_to_voice.get(&note) {
            self.voices[voice_idx].stop();
            self.note_to_voice.remove(&note);
        }
    }
}

impl Plugin for BasicSynth {
    fn get_info(&self) -> Info {
        Info {
            name: "Basic Synth VST".to_string(),
            vendor: "Your Name".to_string(),
            unique_id: 9876543, // Make sure this is unique
            version: 1,
            inputs: 0,
            outputs: 2, // Stereo output
            parameters: Parameter::COUNT as i32,
            category: Category::Synth,
            initial_delay: 0,
            preset_chunks: false,
            f64_precision: true, // Use 64-bit processing if possible
            ..Default::default()
        }
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.sample_rate = rate;
        
        // Update sample rate for all components
        for voice in &mut self.voices {
            voice.oscillator.set_sample_rate(rate);
            voice.envelope.set_sample_rate(rate);
        }
        
        self.filter.set_sample_rate(rate);
    }

    fn process(&mut self, buffer: &mut AudioBuffer<f32>) {
        let (_, mut outputs) = buffer.split();
        let output_count = outputs.len();
        let samples = outputs.get_mut(0).unwrap().len();

        // Zero the outputs first
        for channel_idx in 0..output_count {
            for sample_idx in 0..samples {
                outputs.get_mut(channel_idx)[sample_idx] = 0.0;
            }
        }

        // Update filter parameters
        let cutoff = self.params.get_parameter(Parameter::FilterCutoff);
        let resonance = self.params.get_parameter(Parameter::FilterResonance);
        let filter_type = match (self.params.get_parameter(Parameter::FilterType) * 3.0).floor() as usize {
            0 => FilterType::LowPass,
            1 => FilterType::HighPass,
            2 => FilterType::BandPass,
            _ => FilterType::LowPass,
        };
        
        self.filter.set_cutoff(cutoff * cutoff * 20000.0); // Non-linear scaling for more control
        self.filter.set_resonance(resonance * 0.99); // Limit resonance to avoid explosion
        self.filter.set_type(filter_type);

        // Process each voice and accumulate
        for sample_idx in 0..samples {
            let mut mix = 0.0;
            
            for voice in &mut self.voices {
                if voice.is_active() {
                    mix += voice.next_sample(&self.params);
                }
            }
            
            // Apply master gain
            let gain = self.params.get_parameter(Parameter::MasterGain);
            mix *= gain;
            
            // Apply filter
            let filtered = self.filter.process(mix);
            
            // Output to all channels (stereo)
            for channel_idx in 0..output_count {
                outputs.get_mut(channel_idx)[sample_idx] = filtered;
            }
        }
    }

    fn process_events(&mut self, events: &Events) {
        for event in events.events() {
            self.process_midi_event(event);
        }
    }

    fn can_do(&self, can_do: &str) -> Supported {
        match can_do {
            "receiveEvents" | "receiveMidiEvent" => Supported::Yes,
            "sendEvents" | "sendMidiEvent" => Supported::No,
            _ => Supported::Maybe,
        }
    }

    fn get_parameter_object(&mut self) -> Arc<dyn PluginParameters> {
        Arc::clone(&self.params) as Arc<dyn PluginParameters>
    }
}

plugin_main!(BasicSynth);
