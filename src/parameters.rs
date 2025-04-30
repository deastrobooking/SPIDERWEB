use vst::plugin::PluginParameters;
use vst::util::AtomicFloat;

#[derive(Clone, Copy)]
pub enum Parameter {
    // Oscillator parameters
    OscillatorType, // Basic oscillator types + wavetable and FM
    
    // ADSR envelope parameters
    Attack,
    Decay,
    Sustain,
    Release,
    
    // Filter parameters
    FilterType,     // 0: LP, 0.33: HP, 0.66: BP
    FilterCutoff,
    FilterResonance,
    
    // FM synthesis parameters
    FMCarrierRatio,  // Ratio between carrier frequency and base frequency
    FMModulatorRatio, // Ratio between modulator frequency and base frequency
    FMModIndex,      // Modulation index (intensity of modulation)
    
    // Wavetable parameters
    WavetableHarmonics, // Controls the number of harmonics in complex wavetable
    
    // Master parameters
    MasterGain,
    
    // Count (keep at end)
    COUNT,
}

impl Parameter {
    // Convert enum to usize index
    pub fn as_index(&self) -> usize {
        *self as usize
    }
    
    // Get parameter name
    pub fn get_name(&self) -> String {
        match self {
            Parameter::OscillatorType => "Oscillator Type".to_string(),
            Parameter::Attack => "Attack".to_string(),
            Parameter::Decay => "Decay".to_string(),
            Parameter::Sustain => "Sustain".to_string(),
            Parameter::Release => "Release".to_string(),
            Parameter::FilterType => "Filter Type".to_string(),
            Parameter::FilterCutoff => "Filter Cutoff".to_string(),
            Parameter::FilterResonance => "Filter Resonance".to_string(),
            Parameter::FMCarrierRatio => "FM Carrier Ratio".to_string(),
            Parameter::FMModulatorRatio => "FM Modulator Ratio".to_string(),
            Parameter::FMModIndex => "FM Mod Index".to_string(),
            Parameter::WavetableHarmonics => "Wavetable Harmonics".to_string(),
            Parameter::MasterGain => "Master Gain".to_string(),
            Parameter::COUNT => "".to_string(),
        }
    }
    
    // Get default parameter value
    pub fn get_default(&self) -> f32 {
        match self {
            Parameter::OscillatorType => 0.0,      // Sine wave
            Parameter::Attack => 0.01,             // 10ms
            Parameter::Decay => 0.1,               // 100ms
            Parameter::Sustain => 0.7,             // 70%
            Parameter::Release => 0.3,             // 300ms
            Parameter::FilterType => 0.0,          // Low pass
            Parameter::FilterCutoff => 1.0,        // 20kHz (fully open)
            Parameter::FilterResonance => 0.1,     // Low resonance
            Parameter::FMCarrierRatio => 0.5,      // 1.0 ratio (maps to 0.5-2.0)
            Parameter::FMModulatorRatio => 0.5,    // 1.0 ratio (maps to 0.5-8.0)
            Parameter::FMModIndex => 0.3,          // Moderate modulation index
            Parameter::WavetableHarmonics => 0.5,  // Medium number of harmonics
            Parameter::MasterGain => 0.5,          // 50% volume
            Parameter::COUNT => 0.0,
        }
    }
    
    // Get parameter unit label
    pub fn get_unit(&self) -> String {
        match self {
            Parameter::Attack | Parameter::Decay | Parameter::Release => "s".to_string(),
            Parameter::FilterCutoff => "Hz".to_string(),
            Parameter::FMCarrierRatio | Parameter::FMModulatorRatio => "x".to_string(),
            _ => "".to_string(),
        }
    }
    
    // Get parameter display text (convert 0-1 value to actual display)
    pub fn get_display_text(&self, value: f32) -> String {
        match self {
            Parameter::OscillatorType => {
                match value {
                    v if v < 0.17 => "Sine".to_string(),
                    v if v < 0.33 => "Square".to_string(),
                    v if v < 0.5 => "Saw".to_string(),
                    v if v < 0.67 => "Triangle".to_string(),
                    v if v < 0.83 => "Wavetable".to_string(),
                    _ => "FM".to_string(),
                }
            },
            Parameter::Attack | Parameter::Decay | Parameter::Release => {
                format!("{:.2} s", value * 5.0) // 0-5 seconds
            },
            Parameter::Sustain => {
                format!("{:.0}%", value * 100.0) // 0-100%
            },
            Parameter::FilterType => {
                match value {
                    v if v < 0.33 => "Low Pass".to_string(),
                    v if v < 0.66 => "High Pass".to_string(),
                    _ => "Band Pass".to_string(),
                }
            },
            Parameter::FilterCutoff => {
                // Exponential mapping for cutoff (20Hz - 20kHz)
                let cutoff = 20.0 * (1000.0_f32).powf(value);
                if cutoff < 1000.0 {
                    format!("{:.0} Hz", cutoff)
                } else {
                    format!("{:.1} kHz", cutoff / 1000.0)
                }
            },
            Parameter::FilterResonance => {
                format!("{:.1}", value)
            },
            Parameter::FMCarrierRatio => {
                // Map 0-1 to carrier ratio range (0.5 to 2.0)
                let ratio = 0.5 + value * 1.5;
                format!("{:.2}x", ratio)
            },
            Parameter::FMModulatorRatio => {
                // Map 0-1 to modulator ratio range (0.5 to 8.0, exponential)
                let ratio = 0.5 * (2.0_f32.powf(4.0 * value));
                format!("{:.2}x", ratio)
            },
            Parameter::FMModIndex => {
                // Map 0-1 to modulation index (0 to 10)
                let index = value * 10.0;
                format!("{:.1}", index)
            },
            Parameter::WavetableHarmonics => {
                // Map 0-1 to number of harmonics (1 to 32)
                let harmonics = (1.0 + value * 31.0).round();
                format!("{:.0}", harmonics)
            },
            Parameter::MasterGain => {
                format!("{:.0}%", value * 100.0) // 0-100%
            },
            Parameter::COUNT => "".to_string(),
        }
    }
}

pub struct SynthParameters {
    values: Vec<AtomicFloat>,
    pub time: f32,     // Global time for modulation effects
}

impl Default for SynthParameters {
    fn default() -> Self {
        let mut params = SynthParameters {
            values: Vec::with_capacity(Parameter::COUNT as usize),
            time: 0.0,
        };
        
        // Initialize all parameters with default values
        for i in 0..Parameter::COUNT as usize {
            let param = match i {
                0 => Parameter::OscillatorType,
                1 => Parameter::Attack,
                2 => Parameter::Decay,
                3 => Parameter::Sustain,
                4 => Parameter::Release,
                5 => Parameter::FilterType,
                6 => Parameter::FilterCutoff,
                7 => Parameter::FilterResonance,
                8 => Parameter::FMCarrierRatio,
                9 => Parameter::FMModulatorRatio,
                10 => Parameter::FMModIndex,
                11 => Parameter::WavetableHarmonics,
                12 => Parameter::MasterGain,
                _ => Parameter::COUNT,
            };
            
            params.values.push(AtomicFloat::new(param.get_default()));
        }
        
        params
    }
}

impl SynthParameters {
    pub fn get_parameter(&self, param: Parameter) -> f32 {
        self.values[param as usize].get()
    }
    
    pub fn set_parameter(&self, param: Parameter, value: f32) {
        self.values[param as usize].set(value);
    }
}

impl PluginParameters for SynthParameters {
    fn get_parameter(&self, index: i32) -> f32 {
        if index >= 0 && index < self.values.len() as i32 {
            self.values[index as usize].get()
        } else {
            0.0
        }
    }

    fn set_parameter(&self, index: i32, value: f32) {
        if index >= 0 && index < self.values.len() as i32 {
            self.values[index as usize].set(value);
        }
    }

    fn get_parameter_name(&self, index: i32) -> String {
        if index >= 0 && index < Parameter::COUNT as i32 {
            // Convert index to Parameter enum
            let param = match index {
                0 => Parameter::OscillatorType,
                1 => Parameter::Attack,
                2 => Parameter::Decay,
                3 => Parameter::Sustain,
                4 => Parameter::Release,
                5 => Parameter::FilterType,
                6 => Parameter::FilterCutoff,
                7 => Parameter::FilterResonance,
                8 => Parameter::FMCarrierRatio,
                9 => Parameter::FMModulatorRatio,
                10 => Parameter::FMModIndex,
                11 => Parameter::WavetableHarmonics,
                12 => Parameter::MasterGain,
                _ => Parameter::COUNT,
            };
            
            param.get_name()
        } else {
            "".to_string()
        }
    }

    fn get_parameter_text(&self, index: i32) -> String {
        if index >= 0 && index < Parameter::COUNT as i32 {
            let value = self.values[index as usize].get();
            
            // Convert index to Parameter enum
            let param = match index {
                0 => Parameter::OscillatorType,
                1 => Parameter::Attack,
                2 => Parameter::Decay,
                3 => Parameter::Sustain,
                4 => Parameter::Release,
                5 => Parameter::FilterType,
                6 => Parameter::FilterCutoff,
                7 => Parameter::FilterResonance,
                8 => Parameter::FMCarrierRatio,
                9 => Parameter::FMModulatorRatio,
                10 => Parameter::FMModIndex,
                11 => Parameter::WavetableHarmonics,
                12 => Parameter::MasterGain,
                _ => Parameter::COUNT,
            };
            
            param.get_display_text(value)
        } else {
            "".to_string()
        }
    }

    fn get_parameter_label(&self, index: i32) -> String {
        if index >= 0 && index < Parameter::COUNT as i32 {
            // Convert index to Parameter enum
            let param = match index {
                0 => Parameter::OscillatorType,
                1 => Parameter::Attack,
                2 => Parameter::Decay,
                3 => Parameter::Sustain,
                4 => Parameter::Release,
                5 => Parameter::FilterType,
                6 => Parameter::FilterCutoff,
                7 => Parameter::FilterResonance,
                8 => Parameter::FMCarrierRatio,
                9 => Parameter::FMModulatorRatio,
                10 => Parameter::FMModIndex,
                11 => Parameter::WavetableHarmonics,
                12 => Parameter::MasterGain,
                _ => Parameter::COUNT,
            };
            
            param.get_unit()
        } else {
            "".to_string()
        }
    }
}
