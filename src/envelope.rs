#[derive(PartialEq)]
pub enum EnvelopeState {
    Idle,
    Attack,
    Decay,
    Sustain,
    Release,
}

pub struct Envelope {
    sample_rate: f32,
    state: EnvelopeState,
    value: f32,
    
    // ADSR parameters (in seconds)
    attack: f32,
    decay: f32,
    sustain: f32,
    release: f32,
    
    // Internal state tracking
    attack_rate: f32,
    decay_rate: f32,
    release_rate: f32,
    release_level: f32,
}

impl Envelope {
    pub fn new(sample_rate: f32) -> Self {
        let mut env = Envelope {
            sample_rate,
            state: EnvelopeState::Idle,
            value: 0.0,
            
            attack: 0.01,  // 10ms
            decay: 0.1,    // 100ms
            sustain: 0.7,  // 70% level
            release: 0.3,  // 300ms
            
            attack_rate: 0.0,
            decay_rate: 0.0,
            release_rate: 0.0,
            release_level: 0.0,
        };
        
        env.recalculate_rates();
        env
    }
    
    pub fn set_sample_rate(&mut self, sample_rate: f32) {
        self.sample_rate = sample_rate;
        self.recalculate_rates();
    }
    
    pub fn set_attack(&mut self, seconds: f32) {
        self.attack = seconds.max(0.001); // Minimum 1ms
        self.recalculate_rates();
    }
    
    pub fn set_decay(&mut self, seconds: f32) {
        self.decay = seconds.max(0.001); // Minimum 1ms
        self.recalculate_rates();
    }
    
    pub fn set_sustain(&mut self, level: f32) {
        self.sustain = level.max(0.0).min(1.0); // 0-1 range
    }
    
    pub fn set_release(&mut self, seconds: f32) {
        self.release = seconds.max(0.001); // Minimum 1ms
        self.recalculate_rates();
    }
    
    fn recalculate_rates(&mut self) {
        // Calculate rates in value change per sample
        self.attack_rate = 1.0 / (self.attack * self.sample_rate);
        self.decay_rate = (1.0 - self.sustain) / (self.decay * self.sample_rate);
        self.release_rate = self.sustain / (self.release * self.sample_rate);
    }
    
    pub fn trigger(&mut self) {
        self.state = EnvelopeState::Attack;
    }
    
    pub fn release(&mut self) {
        if self.state != EnvelopeState::Idle {
            self.state = EnvelopeState::Release;
            self.release_level = self.value;
        }
    }
    
    pub fn is_idle(&self) -> bool {
        self.state == EnvelopeState::Idle
    }
    
    pub fn next_value(&mut self) -> f32 {
        match self.state {
            EnvelopeState::Idle => {
                // Keep value at 0
                self.value = 0.0;
            }
            
            EnvelopeState::Attack => {
                // Increase value linearly during attack
                self.value += self.attack_rate;
                
                // If we've reached peak, move to decay
                if self.value >= 1.0 {
                    self.value = 1.0;
                    self.state = EnvelopeState::Decay;
                }
            }
            
            EnvelopeState::Decay => {
                // Decrease value during decay
                self.value -= self.decay_rate;
                
                // If we've reached sustain level, move to sustain
                if self.value <= self.sustain {
                    self.value = self.sustain;
                    self.state = EnvelopeState::Sustain;
                }
            }
            
            EnvelopeState::Sustain => {
                // Hold the sustain value
                self.value = self.sustain;
                // Stay in sustain until release is triggered
            }
            
            EnvelopeState::Release => {
                // Decrease from current level to 0
                self.value -= self.release_rate;
                
                // If we've reached 0, move to idle
                if self.value <= 0.0 {
                    self.value = 0.0;
                    self.state = EnvelopeState::Idle;
                }
            }
        }
        
        self.value
    }
}
