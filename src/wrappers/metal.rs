// Metal Performance Shaders acceleration wrapper (macOS)

pub struct MetalAccelerator;

impl MetalAccelerator {
    pub fn is_available() -> bool {
        // Check for Metal availability on macOS
        cfg!(target_os = "macos")
    }
}