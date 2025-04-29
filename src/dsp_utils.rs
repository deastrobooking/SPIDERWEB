// Digital Signal Processing utilities for the synthesizer
// This module contains utility functions for audio processing used throughout the synth

// Convert MIDI note number to frequency
// Note: A4 = 69 = 440Hz
pub fn midi_note_to_freq(note: u8) -> f32 {
    440.0 * 2.0_f32.powf((note as f32 - 69.0) / 12.0)
}

// Convert frequency to MIDI note number
pub fn freq_to_midi_note(freq: f32) -> f32 {
    69.0 + 12.0 * (freq / 440.0).log2()
}

// Linear interpolation between two values
pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

// Clamp a value between min and max
pub fn clamp(value: f32, min: f32, max: f32) -> f32 {
    if value < min {
        min
    } else if value > max {
        max
    } else {
        value
    }
}

// Convert decibels to linear amplitude
pub fn db_to_linear(db: f32) -> f32 {
    10.0_f32.powf(db / 20.0)
}

// Convert linear amplitude to decibels
pub fn linear_to_db(linear: f32) -> f32 {
    20.0 * linear.abs().max(1e-6).log10()
}

// Exponential smoothing filter for parameter changes
pub struct Smoother {
    value: f32,
    target: f32,
    coeff: f32,
}

impl Smoother {
    pub fn new(initial_value: f32, smoothing_time: f32, sample_rate: f32) -> Self {
        // Calculate smoothing coefficient
        // time is the time it takes to reach ~63% of the target value
        let coeff = 1.0 - (-1.0 / (smoothing_time * sample_rate)).exp();
        
        Smoother {
            value: initial_value,
            target: initial_value,
            coeff,
        }
    }
    
    pub fn set_target(&mut self, target: f32) {
        self.target = target;
    }
    
    pub fn set_smoothing_time(&mut self, smoothing_time: f32, sample_rate: f32) {
        self.coeff = 1.0 - (-1.0 / (smoothing_time * sample_rate)).exp();
    }
    
    pub fn process(&mut self) -> f32 {
        self.value += self.coeff * (self.target - self.value);
        self.value
    }
    
    pub fn reset(&mut self, value: f32) {
        self.value = value;
        self.target = value;
    }
    
    pub fn current(&self) -> f32 {
        self.value
    }
}

// Simple delay line implementation
pub struct DelayLine {
    buffer: Vec<f32>,
    write_pos: usize,
    delay_samples: usize,
}

impl DelayLine {
    pub fn new(max_delay_samples: usize) -> Self {
        DelayLine {
            buffer: vec![0.0; max_delay_samples],
            write_pos: 0,
            delay_samples: max_delay_samples / 2, // Default to half max delay
        }
    }
    
    pub fn set_delay(&mut self, delay_samples: usize) {
        self.delay_samples = delay_samples.min(self.buffer.len() - 1);
    }
    
    pub fn process(&mut self, input: f32) -> f32 {
        // Write input to buffer
        self.buffer[self.write_pos] = input;
        
        // Calculate read position
        let read_pos = (self.write_pos + self.buffer.len() - self.delay_samples) % self.buffer.len();
        
        // Update write position
        self.write_pos = (self.write_pos + 1) % self.buffer.len();
        
        // Return delayed sample
        self.buffer[read_pos]
    }
    
    // Get a delayed sample without updating state (for non-integer delays)
    pub fn tap(&self, delay_samples: f32) -> f32 {
        let delay_samples = delay_samples.max(0.0).min((self.buffer.len() - 1) as f32);
        
        // Integer and fractional parts of the delay
        let delay_int = delay_samples.floor() as usize;
        let delay_frac = delay_samples - delay_int as f32;
        
        // Calculate read positions for interpolation
        let pos1 = (self.write_pos + self.buffer.len() - delay_int) % self.buffer.len();
        let pos2 = (pos1 + self.buffer.len() - 1) % self.buffer.len();
        
        // Linear interpolation between samples
        self.buffer[pos1] * (1.0 - delay_frac) + self.buffer[pos2] * delay_frac
    }
    
    pub fn clear(&mut self) {
        for sample in &mut self.buffer {
            *sample = 0.0;
        }
    }
}
