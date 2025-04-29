use std::f32::consts::PI;

#[derive(Copy, Clone, PartialEq)]
pub enum OscillatorType {
    Sine,
    Square,
    Saw,
    Triangle,
    Wavetable,
    FM,
}

// Parameters specific to FM synthesis
pub struct FMParams {
    pub carrier_ratio: f32,    // Ratio of carrier to base frequency (typically 1.0)
    pub modulator_ratio: f32,  // Ratio of modulator to base frequency (e.g., 0.5, 2.0, etc.)
    pub mod_index: f32,        // Modulation index (intensity of modulation)
    modulator_phase: f32,      // Phase of the modulator oscillator
}

impl Default for FMParams {
    fn default() -> Self {
        FMParams {
            carrier_ratio: 1.0,
            modulator_ratio: 2.0,
            mod_index: 3.0,
            modulator_phase: 0.0,
        }
    }
}

// Wavetable structure with interpolation
pub struct Wavetable {
    pub table: Vec<f32>,
    pub size: usize,
}

impl Wavetable {
    // Create a new wavetable with sine wave
    pub fn new(size: usize) -> Self {
        let mut table = Vec::with_capacity(size);
        for i in 0..size {
            let phase = (i as f32) / (size as f32);
            table.push((phase * 2.0 * PI).sin());
        }
        
        Wavetable {
            table,
            size,
        }
    }
    
    // Get a sample from the wavetable with linear interpolation
    pub fn get_sample(&self, phase: f32) -> f32 {
        let phase_normalized = phase - phase.floor(); // Ensure phase is in [0, 1)
        let pos = phase_normalized * (self.size as f32);
        
        // Integer and fractional parts
        let pos_i = pos as usize % self.size;
        let pos_f = pos - pos_i as f32;
        
        // Get two adjacent samples for interpolation
        let sample1 = self.table[pos_i];
        let sample2 = self.table[(pos_i + 1) % self.size];
        
        // Linear interpolation
        sample1 + pos_f * (sample2 - sample1)
    }
    
    // Generate different wavetable shapes
    pub fn generate(&mut self, wave_type: OscillatorType) {
        for i in 0..self.size {
            let phase = (i as f32) / (self.size as f32);
            let sample = match wave_type {
                OscillatorType::Sine => (phase * 2.0 * PI).sin(),
                OscillatorType::Square => if phase < 0.5 { 1.0 } else { -1.0 },
                OscillatorType::Saw => 2.0 * phase - 1.0,
                OscillatorType::Triangle => {
                    if phase < 0.25 {
                        phase * 4.0
                    } else if phase < 0.75 {
                        2.0 - phase * 4.0
                    } else {
                        phase * 4.0 - 4.0
                    }
                },
                // For these types, just use sine as fallback
                OscillatorType::Wavetable | OscillatorType::FM => (phase * 2.0 * PI).sin(),
            };
            self.table[i] = sample;
        }
    }
}

pub struct Oscillator {
    sample_rate: f32,
    frequency: f32,
    phase: f32,
    osc_type: OscillatorType,
    
    // Additional fields for advanced oscillator types
    wavetable: Wavetable,           // Wavetable for wavetable synthesis
    fm_params: FMParams,            // Parameters for FM synthesis
}

impl Oscillator {
    pub fn new(sample_rate: f32) -> Self {
        // Create a default wavetable with 2048 samples (good size for quality vs. performance)
        let wavetable_size = 2048;
        let mut wavetable = Wavetable::new(wavetable_size);
        
        // Initialize with sine wave
        wavetable.generate(OscillatorType::Sine);
        
        Oscillator {
            sample_rate,
            frequency: 440.0, // Default to A4
            phase: 0.0,
            osc_type: OscillatorType::Sine,
            wavetable,
            fm_params: FMParams::default(),
        }
    }

    pub fn set_frequency(&mut self, frequency: f32) {
        self.frequency = frequency.max(0.1).min(20000.0); // Clamp to audible range
    }

    pub fn set_sample_rate(&mut self, sample_rate: f32) {
        self.sample_rate = sample_rate;
    }

    pub fn set_type(&mut self, osc_type: OscillatorType) {
        // If switching to wavetable mode, make sure the wavetable is generated with proper content
        if osc_type == OscillatorType::Wavetable && self.osc_type != OscillatorType::Wavetable {
            // Generate a complex wavetable with multiple harmonics (sawtooth-like)
            self.generate_complex_wavetable();
        }
        
        self.osc_type = osc_type;
    }

    pub fn get_frequency(&self) -> f32 {
        self.frequency
    }
    
    pub fn set_fm_params(&mut self, carrier_ratio: f32, modulator_ratio: f32, mod_index: f32) {
        self.fm_params.carrier_ratio = carrier_ratio;
        self.fm_params.modulator_ratio = modulator_ratio;
        self.fm_params.mod_index = mod_index;
    }
    
    // Generate a rich wavetable with harmonics for more interesting sounds
    fn generate_complex_wavetable(&mut self) {
        // Clear the wavetable first
        for i in 0..self.wavetable.size {
            self.wavetable.table[i] = 0.0;
        }
        
        // Add harmonics with decreasing amplitude (similar to sawtooth but smoother)
        let num_harmonics = 16;
        for harmonic in 1..=num_harmonics {
            let amplitude = 1.0 / (harmonic as f32);
            
            for i in 0..self.wavetable.size {
                let phase = (i as f32) / (self.wavetable.size as f32);
                let harmonic_phase = phase * (harmonic as f32);
                self.wavetable.table[i] += amplitude * (harmonic_phase * 2.0 * PI).sin();
            }
        }
        
        // Normalize to keep peak amplitude at 1.0
        let mut max_amplitude: f32 = 0.0;
        for sample in &self.wavetable.table {
            max_amplitude = max_amplitude.max(sample.abs());
        }
        
        if max_amplitude > 0.0 {
            for i in 0..self.wavetable.size {
                self.wavetable.table[i] /= max_amplitude;
            }
        }
    }
    
    // FM synthesis implementation
    fn process_fm(&mut self) -> f32 {
        // Calculate modulator frequency based on base frequency and ratio
        let mod_freq = self.frequency * self.fm_params.modulator_ratio;
        
        // Calculate carrier frequency
        let carrier_freq = self.frequency * self.fm_params.carrier_ratio;
        
        // Update modulator phase
        self.fm_params.modulator_phase += mod_freq / self.sample_rate;
        self.fm_params.modulator_phase -= self.fm_params.modulator_phase.floor();
        
        // Calculate modulation amount
        let mod_amount = self.fm_params.mod_index * (self.fm_params.modulator_phase * 2.0 * PI).sin();
        
        // Calculate carrier phase with modulation
        let carrier_phase = self.phase + mod_amount / (2.0 * PI);
        
        // Generate carrier signal (simple sine wave for now)
        let output = (carrier_phase * 2.0 * PI).sin();
        
        // Update carrier phase for next sample
        self.phase += carrier_freq / self.sample_rate;
        self.phase -= self.phase.floor();
        
        output
    }

    pub fn next_sample(&mut self) -> f32 {
        // Calculate the sample based on the oscillator type
        let sample = match self.osc_type {
            OscillatorType::Sine => (self.phase * 2.0 * PI).sin(),
            OscillatorType::Square => self.bl_square(),
            OscillatorType::Saw => self.bl_saw(),
            OscillatorType::Triangle => self.bl_triangle(),
            OscillatorType::Wavetable => self.wavetable.get_sample(self.phase),
            OscillatorType::FM => self.process_fm(),
        };

        // Update phase for next sample (only for non-FM oscillators - FM updates its own phase)
        if self.osc_type != OscillatorType::FM {
            self.phase += self.frequency / self.sample_rate;
            self.phase -= self.phase.floor(); // Keep phase in [0, 1) range
        }

        sample
    }
}

// Implementation for bandlimited oscillators to reduce aliasing
// This is a simple implementation for educational purposes
impl Oscillator {
    // Helper function for bandlimited saw wave using polyBLEP
    fn poly_blep(&self, t: f32) -> f32 {
        let dt = self.frequency / self.sample_rate;
        
        // t is the phase in [0, 1)
        // Check if we're at the discontinuity
        if t < dt {
            // Rising edge (start of waveform)
            let t = t / dt;
            return t * t * 2.0 - t * 4.0 + 2.0;
        } else if t > 1.0 - dt {
            // Falling edge (end of waveform)
            let t = (t - 1.0) / dt + 1.0;
            return t * t * 2.0 - t * 4.0 + 2.0;
        }
        
        0.0
    }

    // Bandlimited sawtooth wave
    pub fn bl_saw(&mut self) -> f32 {
        // Basic sawtooth
        let saw = 2.0 * self.phase - 1.0;
        
        // Add polyBLEP to reduce aliasing
        saw - self.poly_blep(self.phase)
    }

    // Bandlimited square wave
    pub fn bl_square(&mut self) -> f32 {
        // Basic square
        let square = if self.phase < 0.5 { 1.0 } else { -1.0 };
        
        // Add polyBLEP at both edges
        square - self.poly_blep(self.phase) + self.poly_blep((self.phase + 0.5) % 1.0)
    }

    // Bandlimited triangle wave
    pub fn bl_triangle(&mut self) -> f32 {
        // Triangle can be derived from integrating a square wave
        // For simplicity, we'll use the simple method here
        let p = self.phase;
        
        // Basic triangle wave
        if p < 0.25 {
            p * 4.0
        } else if p < 0.75 {
            2.0 - p * 4.0
        } else {
            p * 4.0 - 4.0
        }
    }
}
