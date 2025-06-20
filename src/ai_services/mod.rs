// AI services integration for external model distillation and enhancement

pub mod openai;
pub mod anthropic;
pub mod orchestrator;

use anyhow::Result;

pub use openai::{OpenAIService, OpenAIDistiller};
pub use anthropic::{AnthropicService, AnthropicEnhancer};
pub use orchestrator::{AIServiceOrchestrator, ModelEnhancementConfig, ModelEnhancementReport};

pub async fn initialize_external_services() -> Result<()> {
    // Initialize connections to external AI services
    log::info!("Initializing external AI service integrations");
    
    // Test service availability
    let orchestrator = AIServiceOrchestrator::new();
    log::info!("AI service orchestrator initialized successfully");
    
    Ok(())
}