use std::f32::consts::PI;

#[derive(Copy, Clone, PartialEq)]
pub enum OscillatorType {
    Sine,
    Square,
    Saw,
    Triangle,
}

pub struct Oscillator {
    sample_rate: f32,
    frequency: f32,
    phase: f32,
    osc_type: OscillatorType,
}

impl Oscillator {
    pub fn new(sample_rate: f32) -> Self {
        Oscillator {
            sample_rate,
            frequency: 440.0, // Default to A4
            phase: 0.0,
            osc_type: OscillatorType::Sine,
        }
    }

    pub fn set_frequency(&mut self, frequency: f32) {
        self.frequency = frequency.max(0.1).min(20000.0); // Clamp to audible range
    }

    pub fn set_sample_rate(&mut self, sample_rate: f32) {
        self.sample_rate = sample_rate;
    }

    pub fn set_type(&mut self, osc_type: OscillatorType) {
        self.osc_type = osc_type;
    }

    pub fn next_sample(&mut self) -> f32 {
        // Calculate the sample based on the oscillator type
        let sample = match self.osc_type {
            OscillatorType::Sine => (self.phase * 2.0 * PI).sin(),
            OscillatorType::Square => if (self.phase * 2.0 * PI).sin() >= 0.0 { 1.0 } else { -1.0 },
            OscillatorType::Saw => 2.0 * (self.phase - (self.phase + 0.5).floor()),
            OscillatorType::Triangle => {
                let p = self.phase;
                if p < 0.25 {
                    p * 4.0
                } else if p < 0.75 {
                    2.0 - p * 4.0
                } else {
                    p * 4.0 - 4.0
                }
            }
        };

        // Update phase for next sample
        self.phase += self.frequency / self.sample_rate;
        self.phase -= self.phase.floor(); // Keep phase in [0, 1) range

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
