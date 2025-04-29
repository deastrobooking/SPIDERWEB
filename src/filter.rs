#[derive(Copy, Clone, PartialEq)]
pub enum FilterType {
    LowPass,
    HighPass,
    BandPass,
}

pub struct Filter {
    sample_rate: f32,
    cutoff: f32,
    resonance: f32,
    filter_type: FilterType,
    
    // State variables
    x1: f32,
    x2: f32,
    y1: f32,
    y2: f32,
}

impl Filter {
    pub fn new(sample_rate: f32) -> Self {
        Filter {
            sample_rate,
            cutoff: 1000.0,
            resonance: 0.5,
            filter_type: FilterType::LowPass,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }
    
    pub fn set_sample_rate(&mut self, sample_rate: f32) {
        self.sample_rate = sample_rate;
    }
    
    pub fn set_cutoff(&mut self, cutoff: f32) {
        self.cutoff = cutoff.max(20.0).min(20000.0); // Clamp to audible range
    }
    
    pub fn set_resonance(&mut self, resonance: f32) {
        self.resonance = resonance.max(0.0).min(0.99); // Prevent instability
    }
    
    pub fn set_type(&mut self, filter_type: FilterType) {
        self.filter_type = filter_type;
    }
    
    // Implementation of a 12dB/octave state-variable filter
    pub fn process(&mut self, input: f32) -> f32 {
        // Calculate filter parameters
        let f = 2.0 * self.cutoff / self.sample_rate; // Normalized frequency
        let q = 1.0 - self.resonance; // Higher resonance = lower q
        
        // Main filter algorithm (state variable filter)
        let hp = q * input - q * self.x1 - self.y1;
        let bp = hp + self.y1;
        let lp = bp + self.y2;
        
        // Update state variables
        self.y1 = bp * f + self.y1;
        self.y2 = lp * f + self.y2;
        self.x1 = input;
        
        // Return output based on filter type
        match self.filter_type {
            FilterType::LowPass => lp,
            FilterType::HighPass => hp,
            FilterType::BandPass => bp,
        }
    }
    
    // Implementation of a biquad filter (more flexible but more computationally expensive)
    pub fn process_biquad(&mut self, input: f32) -> f32 {
        // Biquad filter coefficients
        let omega = 2.0 * std::f32::consts::PI * self.cutoff / self.sample_rate;
        let sin_omega = omega.sin();
        let cos_omega = omega.cos();
        let alpha = sin_omega / (2.0 * self.resonance);
        
        let (b0, b1, b2, a0, a1, a2) = match self.filter_type {
            FilterType::LowPass => {
                let b0 = (1.0 - cos_omega) / 2.0;
                let b1 = 1.0 - cos_omega;
                let b2 = (1.0 - cos_omega) / 2.0;
                let a0 = 1.0 + alpha;
                let a1 = -2.0 * cos_omega;
                let a2 = 1.0 - alpha;
                (b0, b1, b2, a0, a1, a2)
            },
            FilterType::HighPass => {
                let b0 = (1.0 + cos_omega) / 2.0;
                let b1 = -(1.0 + cos_omega);
                let b2 = (1.0 + cos_omega) / 2.0;
                let a0 = 1.0 + alpha;
                let a1 = -2.0 * cos_omega;
                let a2 = 1.0 - alpha;
                (b0, b1, b2, a0, a1, a2)
            },
            FilterType::BandPass => {
                let b0 = alpha;
                let b1 = 0.0;
                let b2 = -alpha;
                let a0 = 1.0 + alpha;
                let a1 = -2.0 * cos_omega;
                let a2 = 1.0 - alpha;
                (b0, b1, b2, a0, a1, a2)
            },
        };
        
        // Normalize coefficients
        let b0_norm = b0 / a0;
        let b1_norm = b1 / a0;
        let b2_norm = b2 / a0;
        let a1_norm = a1 / a0;
        let a2_norm = a2 / a0;
        
        // Apply difference equation
        let output = b0_norm * input + b1_norm * self.x1 + b2_norm * self.x2 
                   - a1_norm * self.y1 - a2_norm * self.y2;
        
        // Update state
        self.x2 = self.x1;
        self.x1 = input;
        self.y2 = self.y1;
        self.y1 = output;
        
        output
    }
}
