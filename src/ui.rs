// Advanced UI implementation for the VST synthesizer
// Using iced_baseview for a modern, cross-platform GUI

use crate::parameters::{SynthParameters, Parameter};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::fs;
use std::fmt;
use std::collections::HashMap;

use serde::{Serialize, Deserialize};
use directories::ProjectDirs;

use vst::editor::Editor;
use iced_baseview::{
    baseview, executor, Application, Command, Element, Length, Settings, Subscription,
    window, alignment, Color, Point, Rectangle, 
};
use iced_baseview::{
    widget::{
        button, checkbox, column, container, pick_list, row, scrollable, slider, text, text_input,
        Canvas, Container, Row, Column, Space, Button, Rule,
    },
    canvas::{self, Cache, Cursor, Fill, Frame, Geometry, Path, Stroke},
    Alignment, Background, Border,
};
use iced_graphics::canvas::event::{self, Event};
use iced_native::widget::canvas::Program;

// For cloning the editor
#[derive(Clone)]
pub struct SynthEditor {
    params: Arc<SynthParameters>,
    is_open: bool,
    window_handle: Option<baseview::WindowHandle>,
    window_dimensions: (i32, i32),
}

impl SynthEditor {
    pub fn new(params: Arc<SynthParameters>) -> Self {
        SynthEditor {
            params,
            is_open: false,
            window_handle: None,
            window_dimensions: (980, 680), // Larger window for our comprehensive UI
        }
    }
}

impl Editor for SynthEditor {
    fn size(&self) -> (i32, i32) {
        self.window_dimensions
    }

    fn position(&self) -> (i32, i32) {
        (0, 0) // Default position
    }

    fn open(&mut self, parent: *mut ::std::ffi::c_void) -> bool {
        if self.is_open {
            return false;
        }

        let settings = Settings {
            window: window::Settings {
                title: String::from("Basic Synth VST"),
                size: window::Size::new(self.window_dimensions.0 as f64, self.window_dimensions.1 as f64),
                min_size: Some(window::Size::new(800.0, 600.0)),
                max_size: None,
                scale: window::ScalePolicy::SystemScaleFactor,
                resizable: true,
                parent: parent,
            },
            flags: self.params.clone(),
        };

        let window_handle = SynthUI::open_with_settings(settings);
        self.window_handle = Some(window_handle);
        self.is_open = true;
        true
    }

    fn close(&mut self) {
        if let Some(window_handle) = self.window_handle.take() {
            // Clean up window
            drop(window_handle);
        }
        self.is_open = false;
    }

    fn is_open(&mut self) -> bool {
        self.is_open
    }
}

// Main UI State
struct SynthUI {
    params: Arc<SynthParameters>,
    
    // UI state
    current_tab: Tab,
    preset_name: String,
    preset_list: Vec<String>,
    
    // Sequencer state
    midi_sequencer: Sequencer,
    drum_sequencer: DrumSequencer,
    automation_sequencers: [AutomationSequencer; 4],
    
    // Modulation Matrix
    modulation_matrix: ModulationMatrix,
    
    // UI layout items
    knob_states: HashMap<Parameter, KnobState>,
    pad_states: HashMap<PadType, PadState>,

    // Canvas caches for custom drawing components
    envelope_display_cache: Cache,
    oscilloscope_cache: Cache,
    sequencer_cache: Cache,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Tab {
    Main,
    Sequencer,
    Modulation,
    Settings,
}

impl fmt::Display for Tab {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Tab::Main => write!(f, "Main"),
            Tab::Sequencer => write!(f, "Sequencer"),
            Tab::Modulation => write!(f, "Modulation"),
            Tab::Settings => write!(f, "Settings"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum SequencerType {
    Midi,
    Drum,
    AutomationA,
    AutomationB,
    AutomationC,
    AutomationD,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum PadType {
    XYPad1,
    XYPad2,
}

// Messages for UI state updates
#[derive(Debug, Clone)]
enum Message {
    TabSelected(Tab),
    ParameterChanged(Parameter, f32),
    SequencerStepChanged(SequencerType, usize, f32),
    SequencerLengthChanged(SequencerType, usize),
    PadPositionChanged(PadType, f32, f32),
    PresetNameChanged(String),
    SavePreset,
    LoadPreset(String),
    AutomationShapeSelected(usize, AutomationShape),
    ModulationSourceSelected(usize, ModulationSource),
    ModulationTargetSelected(usize, ModulationTarget),
    ModulationAmountChanged(usize, f32),
    DrumSequencerVelocityChanged(usize, usize, f32),
    DrumSequencerPitchChanged(usize, usize, f32),
    DrumSequencerNoteToggled(usize, usize),
    EnvelopePointAdded(usize, f32, f32),
    EnvelopePointRemoved(usize),
    EnvelopePointMoved(usize, f32, f32),
}

// Custom UI components
struct KnobState {
    value: f32,
    dragging: bool,
    start_pos: Point,
    start_value: f32,
}

struct PadState {
    x: f32,
    y: f32,
    dragging: bool,
}

// Sequencer data structures
struct Sequencer {
    steps: Vec<f32>,
    length: usize,
    active_step: usize,
}

struct DrumSequencer {
    drums: Vec<DrumTrack>,
    length: usize,
    active_step: usize,
}

struct DrumTrack {
    name: String,
    steps: Vec<bool>,
    velocities: Vec<f32>,
    pitches: Vec<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum AutomationShape {
    Linear,
    Exponential,
    Sine,
    Triangle,
    Square,
    Random,
    Custom,
}

impl fmt::Display for AutomationShape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AutomationShape::Linear => write!(f, "Linear"),
            AutomationShape::Exponential => write!(f, "Exponential"),
            AutomationShape::Sine => write!(f, "Sine"),
            AutomationShape::Triangle => write!(f, "Triangle"),
            AutomationShape::Square => write!(f, "Square"),
            AutomationShape::Random => write!(f, "Random"),
            AutomationShape::Custom => write!(f, "Custom"),
        }
    }
}

struct AutomationSequencer {
    target_parameter: Parameter,
    steps: Vec<f32>,
    length: usize,
    active_step: usize,
    shape: AutomationShape,
    custom_envelope: Vec<(f32, f32)>, // Custom curve points (x, y)
}

// Modulation Matrix
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum ModulationSource {
    LFO1,
    LFO2,
    Envelope1,
    Envelope2,
    Velocity,
    ModWheel,
    KeyFollow,
    Random,
}

impl fmt::Display for ModulationSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModulationSource::LFO1 => write!(f, "LFO 1"),
            ModulationSource::LFO2 => write!(f, "LFO 2"),
            ModulationSource::Envelope1 => write!(f, "Envelope 1"),
            ModulationSource::Envelope2 => write!(f, "Envelope 2"),
            ModulationSource::Velocity => write!(f, "Velocity"),
            ModulationSource::ModWheel => write!(f, "Mod Wheel"),
            ModulationSource::KeyFollow => write!(f, "Key Follow"),
            ModulationSource::Random => write!(f, "Random"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum ModulationTarget {
    Cutoff,
    Resonance,
    OscPitch,
    OscMix,
    LFORate,
    LFOAmount,
    EnvelopeAttack,
    EnvelopeDecay,
    EnvelopeRelease,
    PanPosition,
}

impl fmt::Display for ModulationTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModulationTarget::Cutoff => write!(f, "Filter Cutoff"),
            ModulationTarget::Resonance => write!(f, "Filter Resonance"),
            ModulationTarget::OscPitch => write!(f, "Oscillator Pitch"),
            ModulationTarget::OscMix => write!(f, "Oscillator Mix"),
            ModulationTarget::LFORate => write!(f, "LFO Rate"),
            ModulationTarget::LFOAmount => write!(f, "LFO Amount"),
            ModulationTarget::EnvelopeAttack => write!(f, "Envelope Attack"),
            ModulationTarget::EnvelopeDecay => write!(f, "Envelope Decay"),
            ModulationTarget::EnvelopeRelease => write!(f, "Envelope Release"),
            ModulationTarget::PanPosition => write!(f, "Pan Position"),
        }
    }
}

struct ModulationRoute {
    source: ModulationSource,
    target: ModulationTarget,
    amount: f32,
    active: bool,
}

struct ModulationMatrix {
    routes: Vec<ModulationRoute>,
}

// Preset data structure
#[derive(Serialize, Deserialize)]
struct Preset {
    name: String,
    parameters: HashMap<String, f32>,
    sequencer_data: Vec<f32>,
    drum_sequencer_data: Vec<DrumTrackData>,
    automation_data: Vec<AutomationData>,
    modulation_routes: Vec<ModulationRouteData>,
}

#[derive(Serialize, Deserialize)]
struct DrumTrackData {
    name: String,
    steps: Vec<bool>,
    velocities: Vec<f32>,
    pitches: Vec<f32>,
}

#[derive(Serialize, Deserialize)]
struct AutomationData {
    target: String,
    steps: Vec<f32>,
    shape: String,
    custom_points: Vec<(f32, f32)>,
}

#[derive(Serialize, Deserialize)]
struct ModulationRouteData {
    source: String,
    target: String,
    amount: f32,
    active: bool,
}

// Custom UI Component implementations
struct Knob {
    param: Parameter,
    label: String,
    state: &'static mut KnobState,
}

impl Knob {
    fn new(param: Parameter, label: &str, state: &'static mut KnobState) -> Self {
        Knob {
            param,
            label: label.to_string(),
            state,
        }
    }
}

impl Program<Message> for Knob {
    type State = ();

    fn draw(
        &self,
        _state: &Self::State,
        renderer: &mut canvas::Renderer,
        _theme: &iced_native::Theme,
        bounds: Rectangle,
        _cursor: Cursor,
    ) -> Vec<Geometry> {
        let background = renderer.fill(
            Path::circle(bounds.center(), bounds.width.min(bounds.height) / 2.0),
            Color::from_rgb(0.2, 0.2, 0.2),
        );

        let angle = 2.0 * std::f32::consts::PI * self.state.value;
        let indicator_path = Path::line(
            bounds.center(),
            Point::new(
                bounds.center().x + bounds.width / 3.0 * angle.cos(),
                bounds.center().y + bounds.height / 3.0 * angle.sin(),
            ),
        );

        let indicator = renderer.stroke(
            indicator_path,
            Stroke {
                width: 2.0,
                color: Color::from_rgb(0.8, 0.2, 0.2),
                line_cap: canvas::LineCap::Round,
                ..Stroke::default()
            },
        );

        vec![background, indicator]
    }

    fn update(
        &mut self,
        _state: &mut Self::State,
        event: Event,
        bounds: Rectangle,
        cursor: Cursor,
    ) -> (event::Status, Option<Message>) {
        match event {
            Event::Mouse(iced_native::mouse::Event::ButtonPressed(button)) if button == iced_native::mouse::Button::Left => {
                if cursor.is_over(bounds) {
                    self.state.dragging = true;
                    self.state.start_pos = cursor.position().unwrap();
                    self.state.start_value = self.state.value;
                    (event::Status::Captured, None)
                } else {
                    (event::Status::Ignored, None)
                }
            }
            Event::Mouse(iced_native::mouse::Event::ButtonReleased(button)) if button == iced_native::mouse::Button::Left => {
                if self.state.dragging {
                    self.state.dragging = false;
                    (event::Status::Captured, None)
                } else {
                    (event::Status::Ignored, None)
                }
            }
            Event::Mouse(iced_native::mouse::Event::CursorMoved { position }) => {
                if self.state.dragging {
                    let dy = (self.state.start_pos.y - position.y) / 100.0;
                    let new_value = (self.state.start_value + dy).max(0.0).min(1.0);
                    
                    if new_value != self.state.value {
                        self.state.value = new_value;
                        (
                            event::Status::Captured,
                            Some(Message::ParameterChanged(self.param, new_value)),
                        )
                    } else {
                        (event::Status::Captured, None)
                    }
                } else {
                    (event::Status::Ignored, None)
                }
            }
            _ => (event::Status::Ignored, None),
        }
    }
}

struct XYPad {
    pad_type: PadType,
    state: &'static mut PadState,
    x_label: String,
    y_label: String,
}

impl XYPad {
    fn new(pad_type: PadType, x_label: &str, y_label: &str, state: &'static mut PadState) -> Self {
        XYPad {
            pad_type,
            state,
            x_label: x_label.to_string(),
            y_label: y_label.to_string(),
        }
    }
}

impl Program<Message> for XYPad {
    type State = ();

    fn draw(
        &self,
        _state: &Self::State,
        renderer: &mut canvas::Renderer,
        _theme: &iced_native::Theme,
        bounds: Rectangle,
        _cursor: Cursor,
    ) -> Vec<Geometry> {
        let background = renderer.fill(
            Path::rectangle(bounds.position(), bounds.size()),
            Color::from_rgb(0.15, 0.15, 0.15),
        );

        let grid_color = Color::from_rgb(0.3, 0.3, 0.3);
        let mut grid_geometries = Vec::new();

        // Draw grid lines
        for i in 1..10 {
            let x = bounds.x + (bounds.width * i as f32 / 10.0);
            let y = bounds.y + (bounds.height * i as f32 / 10.0);

            let h_line = Path::line(
                Point::new(bounds.x, y),
                Point::new(bounds.x + bounds.width, y),
            );
            
            let v_line = Path::line(
                Point::new(x, bounds.y),
                Point::new(x, bounds.y + bounds.height),
            );

            grid_geometries.push(renderer.stroke(
                h_line,
                Stroke {
                    width: 1.0,
                    color: grid_color,
                    ..Stroke::default()
                },
            ));

            grid_geometries.push(renderer.stroke(
                v_line,
                Stroke {
                    width: 1.0,
                    color: grid_color,
                    ..Stroke::default()
                },
            ));
        }

        // Draw the marker at current position
        let marker_x = bounds.x + self.state.x * bounds.width;
        let marker_y = bounds.y + (1.0 - self.state.y) * bounds.height;
        
        let marker = renderer.fill(
            Path::circle(Point::new(marker_x, marker_y), 10.0),
            Color::from_rgb(0.8, 0.2, 0.2),
        );

        let mut geometries = vec![background];
        geometries.extend(grid_geometries);
        geometries.push(marker);
        
        geometries
    }

    fn update(
        &mut self,
        _state: &mut Self::State,
        event: Event,
        bounds: Rectangle,
        cursor: Cursor,
    ) -> (event::Status, Option<Message>) {
        match event {
            Event::Mouse(iced_native::mouse::Event::ButtonPressed(button)) if button == iced_native::mouse::Button::Left => {
                if cursor.is_over(bounds) {
                    self.state.dragging = true;
                    if let Some(position) = cursor.position() {
                        let normalized_x = ((position.x - bounds.x) / bounds.width).max(0.0).min(1.0);
                        let normalized_y = 1.0 - ((position.y - bounds.y) / bounds.height).max(0.0).min(1.0);
                        
                        self.state.x = normalized_x;
                        self.state.y = normalized_y;
                        
                        (
                            event::Status::Captured,
                            Some(Message::PadPositionChanged(self.pad_type, normalized_x, normalized_y)),
                        )
                    } else {
                        (event::Status::Captured, None)
                    }
                } else {
                    (event::Status::Ignored, None)
                }
            }
            Event::Mouse(iced_native::mouse::Event::ButtonReleased(button)) if button == iced_native::mouse::Button::Left => {
                if self.state.dragging {
                    self.state.dragging = false;
                    (event::Status::Captured, None)
                } else {
                    (event::Status::Ignored, None)
                }
            }
            Event::Mouse(iced_native::mouse::Event::CursorMoved { position }) => {
                if self.state.dragging {
                    let normalized_x = ((position.x - bounds.x) / bounds.width).max(0.0).min(1.0);
                    let normalized_y = 1.0 - ((position.y - bounds.y) / bounds.height).max(0.0).min(1.0);
                    
                    if normalized_x != self.state.x || normalized_y != self.state.y {
                        self.state.x = normalized_x;
                        self.state.y = normalized_y;
                        
                        (
                            event::Status::Captured,
                            Some(Message::PadPositionChanged(self.pad_type, normalized_x, normalized_y)),
                        )
                    } else {
                        (event::Status::Captured, None)
                    }
                } else {
                    (event::Status::Ignored, None)
                }
            }
            _ => (event::Status::Ignored, None),
        }
    }
}

// Implementation of the main application
impl Application for SynthUI {
    type Executor = executor::Default;
    type Message = Message;
    type Flags = Arc<SynthParameters>;

    fn new(params: Arc<SynthParameters>) -> (Self, Command<Message>) {
        let mut knob_states = HashMap::new();
        for param in [
            Parameter::OscillatorType,
            Parameter::Attack,
            Parameter::Decay,
            Parameter::Sustain,
            Parameter::Release,
            Parameter::FilterType,
            Parameter::FilterCutoff,
            Parameter::FilterResonance,
            Parameter::FMCarrierRatio,
            Parameter::FMModulatorRatio,
            Parameter::FMModIndex,
            Parameter::WavetableHarmonics,
            Parameter::MasterGain,
        ].iter() {
            knob_states.insert(*param, KnobState {
                value: params.get_parameter(*param),
                dragging: false,
                start_pos: Point::new(0.0, 0.0),
                start_value: 0.0,
            });
        }

        let pad_states = HashMap::from([
            (PadType::XYPad1, PadState { x: 0.5, y: 0.5, dragging: false }),
            (PadType::XYPad2, PadState { x: 0.5, y: 0.5, dragging: false }),
        ]);

        // Create drum tracks
        let mut drum_tracks = Vec::new();
        for name in ["Kick", "Snare", "Hi-Hat", "Tom", "Clap"].iter() {
            drum_tracks.push(DrumTrack {
                name: name.to_string(),
                steps: vec![false; 16],
                velocities: vec![0.7; 16],
                pitches: vec![0.5; 16],
            });
        }

        // Create automation sequencers
        let automation_sequencers = [
            AutomationSequencer {
                target_parameter: Parameter::FilterCutoff,
                steps: vec![0.5; 16],
                length: 16,
                active_step: 0,
                shape: AutomationShape::Linear,
                custom_envelope: vec![(0.0, 0.0), (0.5, 1.0), (1.0, 0.0)],
            },
            AutomationSequencer {
                target_parameter: Parameter::FilterResonance,
                steps: vec![0.5; 16],
                length: 16,
                active_step: 0,
                shape: AutomationShape::Sine,
                custom_envelope: vec![(0.0, 0.0), (0.5, 1.0), (1.0, 0.0)],
            },
            AutomationSequencer {
                target_parameter: Parameter::OscillatorType,
                steps: vec![0.0; 16],
                length: 16,
                active_step: 0,
                shape: AutomationShape::Triangle,
                custom_envelope: vec![(0.0, 0.0), (0.5, 1.0), (1.0, 0.0)],
            },
            AutomationSequencer {
                target_parameter: Parameter::FMModIndex,
                steps: vec![0.3; 16],
                length: 16,
                active_step: 0,
                shape: AutomationShape::Exponential,
                custom_envelope: vec![(0.0, 0.0), (0.5, 1.0), (1.0, 0.0)],
            },
        ];

        // Create modulation matrix
        let modulation_routes = vec![
            ModulationRoute {
                source: ModulationSource::LFO1,
                target: ModulationTarget::Cutoff,
                amount: 0.5,
                active: true,
            },
            ModulationRoute {
                source: ModulationSource::Envelope1,
                target: ModulationTarget::OscPitch,
                amount: 0.3,
                active: true,
            },
            ModulationRoute {
                source: ModulationSource::Velocity,
                target: ModulationTarget::EnvelopeAttack,
                amount: -0.2,
                active: true,
            },
            ModulationRoute {
                source: ModulationSource::ModWheel,
                target: ModulationTarget::LFOAmount,
                amount: 0.8,
                active: true,
            },
        ];

        // Load preset list
        let preset_list = load_preset_list();

        (
            SynthUI {
                params,
                current_tab: Tab::Main,
                preset_name: "New Preset".to_string(),
                preset_list,
                midi_sequencer: Sequencer {
                    steps: vec![0.0; 16],
                    length: 16,
                    active_step: 0,
                },
                drum_sequencer: DrumSequencer {
                    drums: drum_tracks,
                    length: 16,
                    active_step: 0,
                },
                automation_sequencers,
                modulation_matrix: ModulationMatrix {
                    routes: modulation_routes,
                },
                knob_states,
                pad_states,
                envelope_display_cache: Cache::new(),
                oscilloscope_cache: Cache::new(),
                sequencer_cache: Cache::new(),
            },
            Command::none(),
        )
    }

    fn title(&self) -> String {
        String::from("Basic Synth VST")
    }

    fn update(&mut self, message: Message) -> Command<Message> {
        match message {
            Message::TabSelected(tab) => {
                self.current_tab = tab;
            }
            Message::ParameterChanged(param, value) => {
                if let Some(state) = self.knob_states.get_mut(&param) {
                    state.value = value;
                }
                
                // This is where we would call params.set_parameter normally,
                // but we're keeping this simple for the example
                let params_ptr = Arc::as_ptr(&self.params);
                let params = unsafe { &mut *(params_ptr as *mut SynthParameters) };
                params.set_parameter(param, value);
            }
            Message::SequencerStepChanged(seq_type, step, value) => {
                match seq_type {
                    SequencerType::Midi => {
                        if step < self.midi_sequencer.steps.len() {
                            self.midi_sequencer.steps[step] = value;
                        }
                    }
                    SequencerType::AutomationA => {
                        if step < self.automation_sequencers[0].steps.len() {
                            self.automation_sequencers[0].steps[step] = value;
                        }
                    }
                    SequencerType::AutomationB => {
                        if step < self.automation_sequencers[1].steps.len() {
                            self.automation_sequencers[1].steps[step] = value;
                        }
                    }
                    SequencerType::AutomationC => {
                        if step < self.automation_sequencers[2].steps.len() {
                            self.automation_sequencers[2].steps[step] = value;
                        }
                    }
                    SequencerType::AutomationD => {
                        if step < self.automation_sequencers[3].steps.len() {
                            self.automation_sequencers[3].steps[step] = value;
                        }
                    }
                    SequencerType::Drum => {
                        // Drum sequencer is handled differently
                    }
                }
                
                // Invalidate the sequencer cache to force a redraw
                self.sequencer_cache.clear();
            }
            Message::SequencerLengthChanged(seq_type, length) => {
                let length = length.max(1).min(32);
                
                match seq_type {
                    SequencerType::Midi => {
                        if self.midi_sequencer.steps.len() < length {
                            self.midi_sequencer.steps.resize(length, 0.0);
                        }
                        self.midi_sequencer.length = length;
                    }
                    SequencerType::Drum => {
                        if self.drum_sequencer.length != length {
                            for drum in &mut self.drum_sequencer.drums {
                                if drum.steps.len() < length {
                                    drum.steps.resize(length, false);
                                    drum.velocities.resize(length, 0.7);
                                    drum.pitches.resize(length, 0.5);
                                }
                            }
                            self.drum_sequencer.length = length;
                        }
                    }
                    SequencerType::AutomationA => {
                        if self.automation_sequencers[0].steps.len() < length {
                            self.automation_sequencers[0].steps.resize(length, 0.5);
                        }
                        self.automation_sequencers[0].length = length;
                    }
                    SequencerType::AutomationB => {
                        if self.automation_sequencers[1].steps.len() < length {
                            self.automation_sequencers[1].steps.resize(length, 0.5);
                        }
                        self.automation_sequencers[1].length = length;
                    }
                    SequencerType::AutomationC => {
                        if self.automation_sequencers[2].steps.len() < length {
                            self.automation_sequencers[2].steps.resize(length, 0.5);
                        }
                        self.automation_sequencers[2].length = length;
                    }
                    SequencerType::AutomationD => {
                        if self.automation_sequencers[3].steps.len() < length {
                            self.automation_sequencers[3].steps.resize(length, 0.5);
                        }
                        self.automation_sequencers[3].length = length;
                    }
                }
                
                // Invalidate the sequencer cache to force a redraw
                self.sequencer_cache.clear();
            }
            Message::PadPositionChanged(pad_type, x, y) => {
                if let Some(state) = self.pad_states.get_mut(&pad_type) {
                    state.x = x;
                    state.y = y;
                }
                
                // Here we would map the XY pad values to synth parameters
                // For example, XYPad1 might control filter cutoff (x) and resonance (y)
                match pad_type {
                    PadType::XYPad1 => {
                        let params_ptr = Arc::as_ptr(&self.params);
                        let params = unsafe { &mut *(params_ptr as *mut SynthParameters) };
                        params.set_parameter(Parameter::FilterCutoff, x);
                        params.set_parameter(Parameter::FilterResonance, y);
                        
                        // Update the knob states too
                        if let Some(state) = self.knob_states.get_mut(&Parameter::FilterCutoff) {
                            state.value = x;
                        }
                        if let Some(state) = self.knob_states.get_mut(&Parameter::FilterResonance) {
                            state.value = y;
                        }
                    }
                    PadType::XYPad2 => {
                        let params_ptr = Arc::as_ptr(&self.params);
                        let params = unsafe { &mut *(params_ptr as *mut SynthParameters) };
                        params.set_parameter(Parameter::FMCarrierRatio, x);
                        params.set_parameter(Parameter::FMModulatorRatio, y);
                        
                        // Update the knob states too
                        if let Some(state) = self.knob_states.get_mut(&Parameter::FMCarrierRatio) {
                            state.value = x;
                        }
                        if let Some(state) = self.knob_states.get_mut(&Parameter::FMModulatorRatio) {
                            state.value = y;
                        }
                    }
                }
            }
            Message::PresetNameChanged(name) => {
                self.preset_name = name;
            }
            Message::SavePreset => {
                save_preset(self);
                // Refresh preset list
                self.preset_list = load_preset_list();
            }
            Message::LoadPreset(name) => {
                load_preset(self, &name);
            }
            Message::AutomationShapeSelected(index, shape) => {
                if index < self.automation_sequencers.len() {
                    self.automation_sequencers[index].shape = shape;
                    // Invalidate the envelope display cache
                    self.envelope_display_cache.clear();
                }
            }
            Message::ModulationSourceSelected(index, source) => {
                if index < self.modulation_matrix.routes.len() {
                    self.modulation_matrix.routes[index].source = source;
                }
            }
            Message::ModulationTargetSelected(index, target) => {
                if index < self.modulation_matrix.routes.len() {
                    self.modulation_matrix.routes[index].target = target;
                }
            }
            Message::ModulationAmountChanged(index, amount) => {
                if index < self.modulation_matrix.routes.len() {
                    self.modulation_matrix.routes[index].amount = amount;
                }
            }
            Message::DrumSequencerNoteToggled(track, step) => {
                if track < self.drum_sequencer.drums.len() && step < self.drum_sequencer.drums[track].steps.len() {
                    let current = self.drum_sequencer.drums[track].steps[step];
                    self.drum_sequencer.drums[track].steps[step] = !current;
                    // Invalidate the sequencer cache
                    self.sequencer_cache.clear();
                }
            }
            Message::DrumSequencerVelocityChanged(track, step, velocity) => {
                if track < self.drum_sequencer.drums.len() && step < self.drum_sequencer.drums[track].velocities.len() {
                    self.drum_sequencer.drums[track].velocities[step] = velocity;
                }
            }
            Message::DrumSequencerPitchChanged(track, step, pitch) => {
                if track < self.drum_sequencer.drums.len() && step < self.drum_sequencer.drums[track].pitches.len() {
                    self.drum_sequencer.drums[track].pitches[step] = pitch;
                }
            }
            Message::EnvelopePointAdded(seq_index, x, y) => {
                if seq_index < self.automation_sequencers.len() {
                    let mut points = &mut self.automation_sequencers[seq_index].custom_envelope;
                    points.push((x, y));
                    points.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                    // Invalidate the envelope display cache
                    self.envelope_display_cache.clear();
                }
            }
            Message::EnvelopePointRemoved(seq_index) => {
                if seq_index < self.automation_sequencers.len() {
                    let points = &mut self.automation_sequencers[seq_index].custom_envelope;
                    if points.len() > 2 {
                        points.pop();
                        // Invalidate the envelope display cache
                        self.envelope_display_cache.clear();
                    }
                }
            }
            Message::EnvelopePointMoved(seq_index, x, y) => {
                if seq_index < self.automation_sequencers.len() {
                    let points = &mut self.automation_sequencers[seq_index].custom_envelope;
                    // Find the closest point and move it
                    if !points.is_empty() {
                        let mut closest_idx = 0;
                        let mut closest_dist = f32::MAX;
                        
                        for (i, &(px, py)) in points.iter().enumerate() {
                            let dist = ((px - x).powi(2) + (py - y).powi(2)).sqrt();
                            if dist < closest_dist {
                                closest_dist = dist;
                                closest_idx = i;
                            }
                        }
                        
                        // Don't move the first or last point horizontally (keep them at 0.0 and 1.0)
                        if closest_idx == 0 {
                            points[closest_idx] = (0.0, y.max(0.0).min(1.0));
                        } else if closest_idx == points.len() - 1 {
                            points[closest_idx] = (1.0, y.max(0.0).min(1.0));
                        } else {
                            // Calculate min/max x values to maintain order
                            let min_x = points[closest_idx - 1].0 + 0.01;
                            let max_x = if closest_idx < points.len() - 1 {
                                points[closest_idx + 1].0 - 0.01
                            } else {
                                1.0
                            };
                            
                            points[closest_idx] = (x.max(min_x).min(max_x), y.max(0.0).min(1.0));
                        }
                        
                        // Invalidate the envelope display cache
                        self.envelope_display_cache.clear();
                    }
                }
            }
        }
        
        Command::none()
    }

    fn view(&mut self) -> Element<Message> {
        let content: Element<_> = match self.current_tab {
            Tab::Main => self.view_main_tab(),
            Tab::Sequencer => self.view_sequencer_tab(),
            Tab::Modulation => self.view_modulation_tab(),
            Tab::Settings => self.view_settings_tab(),
        };

        let tabs = row![
            button("Main").on_press(Message::TabSelected(Tab::Main))
                .style(style_for_tab(self.current_tab == Tab::Main)),
            button("Sequencer").on_press(Message::TabSelected(Tab::Sequencer))
                .style(style_for_tab(self.current_tab == Tab::Sequencer)),
            button("Modulation").on_press(Message::TabSelected(Tab::Modulation))
                .style(style_for_tab(self.current_tab == Tab::Modulation)),
            button("Settings").on_press(Message::TabSelected(Tab::Settings))
                .style(style_for_tab(self.current_tab == Tab::Settings)),
        ]
        .spacing(10)
        .padding(10);

        // Main column contains the tabs and the content
        column![tabs, content]
            .spacing(20)
            .padding(20)
            .into()
    }

    fn subscription(&self) -> Subscription<Message> {
        Subscription::none()
    }
}

// Helper functions for the UI implementation
impl SynthUI {
    fn view_main_tab(&self) -> Element<Message> {
        let title = text("Synth Parameters")
            .size(24)
            .horizontal_alignment(alignment::Horizontal::Center);

        // TODO: Implement the actual knobs and XY pads here
        // This is a placeholder implementation
        
        let osc_section = column![
            text("Oscillator").size(18),
            // Placeholder for oscillator controls
            text("Type: Sine/Square/Saw/etc"),
            row![
                text("Wavetable Harmonics: "),
                slider(0.0..=1.0, self.params.get_parameter(Parameter::WavetableHarmonics), |v| {
                    Message::ParameterChanged(Parameter::WavetableHarmonics, v)
                })
                .step(0.01)
                .width(Length::Fill),
            ]
        ]
        .spacing(10)
        .padding(10);

        let envelope_section = column![
            text("Envelope").size(18),
            row![
                text("Attack: "),
                slider(0.0..=1.0, self.params.get_parameter(Parameter::Attack), |v| {
                    Message::ParameterChanged(Parameter::Attack, v)
                })
                .step(0.01)
                .width(Length::Fill),
            ],
            row![
                text("Decay: "),
                slider(0.0..=1.0, self.params.get_parameter(Parameter::Decay), |v| {
                    Message::ParameterChanged(Parameter::Decay, v)
                })
                .step(0.01)
                .width(Length::Fill),
            ],
            row![
                text("Sustain: "),
                slider(0.0..=1.0, self.params.get_parameter(Parameter::Sustain), |v| {
                    Message::ParameterChanged(Parameter::Sustain, v)
                })
                .step(0.01)
                .width(Length::Fill),
            ],
            row![
                text("Release: "),
                slider(0.0..=1.0, self.params.get_parameter(Parameter::Release), |v| {
                    Message::ParameterChanged(Parameter::Release, v)
                })
                .step(0.01)
                .width(Length::Fill),
            ],
        ]
        .spacing(10)
        .padding(10);

        let filter_section = column![
            text("Filter").size(18),
            row![
                text("Cutoff: "),
                slider(0.0..=1.0, self.params.get_parameter(Parameter::FilterCutoff), |v| {
                    Message::ParameterChanged(Parameter::FilterCutoff, v)
                })
                .step(0.01)
                .width(Length::Fill),
            ],
            row![
                text("Resonance: "),
                slider(0.0..=1.0, self.params.get_parameter(Parameter::FilterResonance), |v| {
                    Message::ParameterChanged(Parameter::FilterResonance, v)
                })
                .step(0.01)
                .width(Length::Fill),
            ],
        ]
        .spacing(10)
        .padding(10);

        let fm_section = column![
            text("FM Synthesis").size(18),
            row![
                text("Carrier Ratio: "),
                slider(0.0..=1.0, self.params.get_parameter(Parameter::FMCarrierRatio), |v| {
                    Message::ParameterChanged(Parameter::FMCarrierRatio, v)
                })
                .step(0.01)
                .width(Length::Fill),
            ],
            row![
                text("Modulator Ratio: "),
                slider(0.0..=1.0, self.params.get_parameter(Parameter::FMModulatorRatio), |v| {
                    Message::ParameterChanged(Parameter::FMModulatorRatio, v)
                })
                .step(0.01)
                .width(Length::Fill),
            ],
            row![
                text("Mod Index: "),
                slider(0.0..=1.0, self.params.get_parameter(Parameter::FMModIndex), |v| {
                    Message::ParameterChanged(Parameter::FMModIndex, v)
                })
                .step(0.01)
                .width(Length::Fill),
            ],
        ]
        .spacing(10)
        .padding(10);

        let master_section = column![
            text("Master").size(18),
            row![
                text("Gain: "),
                slider(0.0..=1.0, self.params.get_parameter(Parameter::MasterGain), |v| {
                    Message::ParameterChanged(Parameter::MasterGain, v)
                })
                .step(0.01)
                .width(Length::Fill),
            ],
        ]
        .spacing(10)
        .padding(10);

        let preset_section = row![
            text_input("Preset Name", &self.preset_name)
                .on_input(Message::PresetNameChanged)
                .width(Length::FillPortion(3)),
            button("Save")
                .on_press(Message::SavePreset)
                .width(Length::FillPortion(1)),
            pick_list(
                &self.preset_list,
                None,
                Message::LoadPreset,
            )
            .width(Length::FillPortion(2)),
        ]
        .spacing(10)
        .padding(10);

        // Main layout
        let left_column = column![
            osc_section,
            envelope_section,
        ]
        .width(Length::FillPortion(1))
        .spacing(20);

        let right_column = column![
            filter_section,
            fm_section,
            master_section,
        ]
        .width(Length::FillPortion(1))
        .spacing(20);

        column![
            title,
            preset_section,
            row![left_column, right_column].spacing(20),
        ]
        .spacing(20)
        .into()
    }

    fn view_sequencer_tab(&self) -> Element<Message> {
        let title = text("Sequencers")
            .size(24)
            .horizontal_alignment(alignment::Horizontal::Center);

        // TODO: Implement actual sequencer UI components
        // This is a placeholder implementation
        
        let midi_sequencer = column![
            text("MIDI Sequencer").size(18),
            // Placeholder for MIDI sequencer
            text("16 steps with note/velocity control")
        ]
        .spacing(10)
        .padding(10);

        let drum_sequencer = column![
            text("Drum Sequencer").size(18),
            // Placeholder for drum sequencer
            text("5 tracks with velocity and pitch control")
        ]
        .spacing(10)
        .padding(10);

        let automation_sequencer = column![
            text("Automation Sequencers").size(18),
            // Placeholder for automation sequencers
            text("4 lanes of parameter automation with shape selection")
        ]
        .spacing(10)
        .padding(10);

        column![
            title,
            midi_sequencer,
            drum_sequencer,
            automation_sequencer,
        ]
        .spacing(20)
        .into()
    }

    fn view_modulation_tab(&self) -> Element<Message> {
        let title = text("Modulation Matrix")
            .size(24)
            .horizontal_alignment(alignment::Horizontal::Center);

        // TODO: Implement actual modulation matrix UI
        // This is a placeholder implementation
        
        let matrix_rows = column![
            row![
                text("Source"),
                text("Target"),
                text("Amount"),
                text("Enable"),
            ]
            .spacing(20)
            .padding(5),
            // Placeholder for modulation matrix rows
            row![
                text("LFO 1"),
                text("Filter Cutoff"),
                text("50%"),
                checkbox(true, ""),
            ]
            .spacing(20)
            .padding(5),
            row![
                text("Envelope 1"),
                text("Oscillator Pitch"),
                text("30%"),
                checkbox(true, ""),
            ]
            .spacing(20)
            .padding(5),
            row![
                text("Velocity"),
                text("Envelope Attack"),
                text("-20%"),
                checkbox(true, ""),
            ]
            .spacing(20)
            .padding(5),
            row![
                text("Mod Wheel"),
                text("LFO Amount"),
                text("80%"),
                checkbox(true, ""),
            ]
            .spacing(20)
            .padding(5),
        ]
        .spacing(10)
        .padding(10);

        column![
            title,
            matrix_rows,
            button("Add Modulation Route").padding(10),
        ]
        .spacing(20)
        .into()
    }

    fn view_settings_tab(&self) -> Element<Message> {
        let title = text("Settings")
            .size(24)
            .horizontal_alignment(alignment::Horizontal::Center);

        // TODO: Implement actual settings UI
        // This is a placeholder implementation
        
        let general_settings = column![
            text("General Settings").size(18),
            row![
                text("Voice Count: "),
                text("16"),
            ],
            row![
                text("MPE Mode: "),
                checkbox(false, "Enable MPE support"),
            ],
        ]
        .spacing(10)
        .padding(10);

        let midi_settings = column![
            text("MIDI Settings").size(18),
            row![
                text("MIDI Channel: "),
                text("All"),
            ],
            row![
                text("MIDI Learn: "),
                button("Start MIDI Learn"),
            ],
        ]
        .spacing(10)
        .padding(10);

        column![
            title,
            general_settings,
            midi_settings,
        ]
        .spacing(20)
        .into()
    }
}

// Helper functions for styling and button state
fn style_for_tab(selected: bool) -> fn(&iced_native::widget::button::Appearance) -> iced_native::widget::button::Appearance {
    if selected {
        |appearance| iced_native::widget::button::Appearance {
            background: Some(Background::Color(Color::from_rgb(0.3, 0.3, 0.4))),
            border: Border {
                color: Color::WHITE,
                width: 1.0,
                radius: 2.0.into(),
            },
            ..appearance.clone()
        }
    } else {
        |appearance| iced_native::widget::button::Appearance {
            background: Some(Background::Color(Color::from_rgb(0.2, 0.2, 0.3))),
            ..appearance.clone()
        }
    }
}

// Functions for preset management
fn get_presets_dir() -> Option<PathBuf> {
    if let Some(proj_dirs) = ProjectDirs::from("com", "SynthVST", "BasicSynth") {
        let dir = proj_dirs.data_dir();
        if !dir.exists() {
            if let Err(_) = fs::create_dir_all(dir) {
                return None;
            }
        }
        Some(dir.to_path_buf())
    } else {
        None
    }
}

fn load_preset_list() -> Vec<String> {
    if let Some(dir) = get_presets_dir() {
        match fs::read_dir(dir) {
            Ok(entries) => {
                let mut presets = Vec::new();
                for entry in entries.filter_map(Result::ok) {
                    if entry.path().extension().map_or(false, |ext| ext == "json") {
                        if let Some(name) = entry.path().file_stem().and_then(|os_str| os_str.to_str()) {
                            presets.push(name.to_string());
                        }
                    }
                }
                return presets;
            }
            Err(_) => {}
        }
    }
    Vec::new()
}

fn save_preset(ui: &SynthUI) {
    if let Some(dir) = get_presets_dir() {
        let file_path = dir.join(format!("{}.json", ui.preset_name));
        
        // Create a serializable preset object
        let mut parameters = HashMap::new();
        for (param, _) in &ui.knob_states {
            let value = ui.params.get_parameter(*param);
            parameters.insert(format!("{:?}", param), value);
        }
        
        // Convert modulation routes to serializable data
        let modulation_routes: Vec<ModulationRouteData> = ui.modulation_matrix.routes
            .iter()
            .map(|route| ModulationRouteData {
                source: format!("{:?}", route.source),
                target: format!("{:?}", route.target),
                amount: route.amount,
                active: route.active,
            })
            .collect();
        
        // Convert automation sequencers to serializable data
        let automation_data: Vec<AutomationData> = ui.automation_sequencers
            .iter()
            .map(|seq| AutomationData {
                target: format!("{:?}", seq.target_parameter),
                steps: seq.steps.clone(),
                shape: format!("{:?}", seq.shape),
                custom_points: seq.custom_envelope.clone(),
            })
            .collect();
        
        // Convert drum tracks to serializable data
        let drum_data: Vec<DrumTrackData> = ui.drum_sequencer.drums
            .iter()
            .map(|track| DrumTrackData {
                name: track.name.clone(),
                steps: track.steps.clone(),
                velocities: track.velocities.clone(),
                pitches: track.pitches.clone(),
            })
            .collect();
        
        // Create the preset structure
        let preset = Preset {
            name: ui.preset_name.clone(),
            parameters,
            sequencer_data: ui.midi_sequencer.steps.clone(),
            drum_sequencer_data: drum_data,
            automation_data,
            modulation_routes,
        };
        
        // Serialize to JSON and save
        if let Ok(json) = serde_json::to_string_pretty(&preset) {
            let _ = fs::write(file_path, json); // Ignore errors for this example
        }
    }
}

fn load_preset(ui: &mut SynthUI, name: &str) {
    if let Some(dir) = get_presets_dir() {
        let file_path = dir.join(format!("{}.json", name));
        
        if let Ok(json) = fs::read_to_string(file_path) {
            if let Ok(preset) = serde_json::from_str::<Preset>(&json) {
                // Update preset name
                ui.preset_name = preset.name;
                
                // Update parameters
                let params_ptr = Arc::as_ptr(&ui.params);
                let params = unsafe { &mut *(params_ptr as *mut SynthParameters) };
                
                for (param_str, value) in preset.parameters {
                    // This is a simplification - real code would need to parse the enum properly
                    // For example: if param_str == "OscillatorType" { params.set_parameter(Parameter::OscillatorType, value); }
                    // Updating all knob states would also be needed
                }
                
                // Update sequencer data
                ui.midi_sequencer.steps = preset.sequencer_data;
                
                // Update drum sequencer data
                // This would need to map the saved data to the UI structures
                
                // Update automation sequencers
                // This would need to map the saved data to the UI structures
                
                // Update modulation matrix
                // This would need to map the saved data to the UI structures
                
                // Clear all caches to force redraw
                ui.envelope_display_cache.clear();
                ui.oscilloscope_cache.clear();
                ui.sequencer_cache.clear();
            }
        }
    }
}
