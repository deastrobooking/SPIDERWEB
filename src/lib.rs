#[macro_use]
extern crate vst;
extern crate log;

pub mod oscillator;
pub mod envelope;
pub mod filter;
pub mod parameters;
pub mod ui;
mod dsp_utils;
mod suppress_warnings; // Keeping this as empty module for future use
pub mod tester;        // Add tester module for standalone testing

use vst::api::{Events, Supported};
use vst::buffer::AudioBuffer;
use vst::event::Event;
use vst::plugin::{Category, HostCallback, Info, Plugin, PluginParameters, CanDo};

use parameters::{SynthParameters, Parameter};
use oscillator::{Oscillator, OscillatorType};
use envelope::Envelope;
use filter::{Filter, FilterType};

use std::sync::Arc;
use std::collections::HashMap;

const MAX_VOICES: usize = 16;

struct Voice {
    active: bool,
    note: u8,
    velocity: f32,
    oscillator: Oscillator,
    envelope: Envelope,
    pitch_bend: f32,
    modulation: f32,
}

impl Voice {
    fn new() -> Self {
        Voice {
            active: false,
            note: 0,
            velocity: 0.0,
            oscillator: Oscillator::new(44100.0),
            envelope: Envelope::new(44100.0),
            pitch_bend: 0.0,    // No pitch bend initially
            modulation: 0.0,    // No modulation initially
        }
    }

    fn start(&mut self, note: u8, velocity: f32) {
        self.active = true;
        self.note = note;
        self.velocity = velocity;
        
        // Set frequency with current pitch bend
        self.update_frequency();
        self.envelope.trigger();
    }

    fn stop(&mut self) {
        self.envelope.release();
    }

    fn is_active(&self) -> bool {
        self.active && !self.envelope.is_idle()
    }

    fn set_pitch_bend(&mut self, bend: f32) {
        self.pitch_bend = bend;
        self.update_frequency();
    }

    fn set_modulation(&mut self, mod_value: f32) {
        self.modulation = mod_value;
    }

    fn update_frequency(&mut self) {
        // Convert MIDI note to frequency (A4 = 69 = 440Hz)
        // Apply pitch bend: range of +/- 2 semitones (can be adjusted)
        let bend_semitones = self.pitch_bend * 2.0; // -1.0 to 1.0 -> -2 to +2 semitones
        let note_with_bend = self.note as f32 + bend_semitones;
        let frequency = 440.0 * 2.0_f32.powf((note_with_bend - 69.0) / 12.0);
        self.oscillator.set_frequency(frequency);
    }

    fn next_sample(&mut self, params: &SynthParameters) -> f32 {
        if !self.is_active() {
            self.active = false;
            return 0.0;
        }

        // Get oscillator value based on current parameters
        let osc_type = match params.get_parameter(Parameter::OscillatorType) {
            x if x < 0.17 => OscillatorType::Sine,
            x if x < 0.33 => OscillatorType::Square,
            x if x < 0.5 => OscillatorType::Saw,
            x if x < 0.67 => OscillatorType::Triangle,
            x if x < 0.83 => OscillatorType::Wavetable,
            _ => OscillatorType::FM,
        };
        self.oscillator.set_type(osc_type);
        
        // If using FM synthesis, update the parameters
        if osc_type == OscillatorType::FM {
            // Map normalized parameters to appropriate ranges for FM synthesis
            let carrier_ratio = 0.5 + params.get_parameter(Parameter::FMCarrierRatio) * 1.5; // 0.5 to 2.0
            let modulator_ratio = 0.5 * (2.0_f32.powf(4.0 * params.get_parameter(Parameter::FMModulatorRatio))); // 0.5 to 8.0
            let mod_index = params.get_parameter(Parameter::FMModIndex) * 10.0; // 0 to 10
            
            // Update FM parameters in the oscillator
            self.oscillator.set_fm_params(carrier_ratio, modulator_ratio, mod_index);
        }
        
        // If using wavetable, we might want to control the number of harmonics
        if osc_type == OscillatorType::Wavetable {
            // Additional parameters could be used to alter the wavetable in real-time
            // This could be implemented in the future for more dynamic control
            
            // For now, we'll just use the preset wavetable that was generated
            // when we switched to wavetable mode
        }

        // Get raw oscillator sample
        let mut sample = self.oscillator.next_sample();
        
        // Apply modulation (subtle vibrato effect when mod wheel is used)
        if self.modulation > 0.0 {
            // Create a low-frequency oscillation (5Hz) for vibrato
            let vibrato_amount = self.modulation * 0.05; // Max 5% frequency variation
            let lfo_value = (std::f32::consts::PI * 5.0 * params.time).sin() * vibrato_amount;
            
            // Temporarily adjust the frequency for vibrato
            let base_freq = self.oscillator.get_frequency();
            self.oscillator.set_frequency(base_freq * (1.0 + lfo_value));
            sample = self.oscillator.next_sample();
            // Restore the original frequency
            self.oscillator.set_frequency(base_freq);
        }

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
    editor: Option<ui::SynthEditor>,
}

impl BasicSynth {
    fn new(_host: HostCallback) -> Self {
        let params = Arc::new(SynthParameters::default());
        
        let mut voices = Vec::with_capacity(MAX_VOICES);
        for _ in 0..MAX_VOICES {
            voices.push(Voice::new());
        }

        BasicSynth {
            sample_rate: 44100.0,
            params: params.clone(),
            voices,
            filter: Filter::new(44100.0),
            note_to_voice: HashMap::new(),
            editor: None,
        }
    }

    fn process_midi_event(&mut self, event: &Event<'_>) {
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
                        let controller = message[1];
                        let value = message[2] as f32 / 127.0;
                        
                        // Handle different MIDI CC messages
                        match controller {
                            // CC 1 = Modulation Wheel
                            1 => {
                                // Apply modulation to all active voices
                                for voice in &mut self.voices {
                                    if voice.is_active() {
                                        voice.set_modulation(value);
                                    }
                                }
                            },
                            // Add more CC handlers here as needed
                            // CC 7 = Channel Volume
                            7 => {
                                // We could update a volume parameter here
                                // For now we'll just use a placeholder
                                println!("Volume CC received: {}", value);
                            },
                            // Other CC messages
                            _ => {
                                // Ignore unhandled CC messages
                            }
                        }
                    }
                    // Pitch Bend
                    0xE0 => {
                        let lsb = message[1] as u16;
                        let msb = message[2] as u16;
                        let normalized_bend = ((msb << 7) | lsb) as f32 / 8192.0 - 1.0;
                        
                        // Apply pitch bend to all active voices
                        for voice in &mut self.voices {
                            if voice.is_active() {
                                voice.set_pitch_bend(normalized_bend);
                            }
                        }
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
    fn new(host: HostCallback) -> Self {
        BasicSynth::new(host)
    }

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
        // Get the number of samples first - all channels have the same length in a VST buffer
        let samples = buffer.samples();
        
        let (_, mut outputs) = buffer.split();
        let output_count = outputs.len();
        
        // Make sure we have at least one output channel
        if output_count == 0 {
            return;
        }
        
        // Zero the outputs first
        for channel_idx in 0..output_count {
            for sample_idx in 0..samples {
                outputs[channel_idx][sample_idx] = 0.0;
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

        // Get a reference to the parameters for internal mutation of time
        let params_ptr = Arc::as_ptr(&self.params);
        let params = unsafe { &mut *(params_ptr as *mut SynthParameters) };
        
        // Calculate time increment for this buffer
        let time_increment = 1.0 / self.sample_rate;

        // Process each voice and accumulate
        for sample_idx in 0..samples {
            // Update the global time for modulation effects
            params.time += time_increment;
            // Wrap time at a reasonable interval to prevent floating-point precision issues
            if params.time > 1000.0 {
                params.time -= 1000.0;
            }
            
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
                outputs[channel_idx][sample_idx] = filtered;
            }
        }
    }

    fn process_events(&mut self, events: &Events) {
        for event in events.events() {
            self.process_midi_event(&event);
        }
    }

    fn can_do(&self, can_do: CanDo) -> Supported {
        match can_do {
            CanDo::ReceiveEvents | CanDo::ReceiveMidiEvent => Supported::Yes,
            CanDo::SendEvents | CanDo::SendMidiEvent => Supported::No,
            _ => Supported::Maybe,
        }
    }

    fn get_parameter_object(&mut self) -> Arc<dyn PluginParameters> {
        Arc::clone(&self.params) as Arc<dyn PluginParameters>
    }
    
    fn get_editor(&mut self) -> Option<Box<dyn vst::editor::Editor>> {
        if self.editor.is_none() {
            self.editor = Some(ui::SynthEditor::new(Arc::clone(&self.params)));
        }
        
        if let Some(editor) = &mut self.editor {
            Some(Box::new(editor.clone()))
        } else {
            None
        }
    }
}

plugin_main!(BasicSynth);
