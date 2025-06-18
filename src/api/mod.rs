// ML-as-a-Service API implementation

pub mod training;
pub mod inference;
pub mod auth;
pub mod models;

use actix_web::{web, App, HttpServer, Result as ActixResult, middleware::Logger};
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::wrappers::FrameworkRouter;

pub struct ApiState {
    pub framework_router: Arc<FrameworkRouter>,
    pub model_registry: Arc<RwLock<models::ModelRegistry>>,
    pub auth_service: Arc<auth::AuthService>,
    pub training_manager: Arc<RwLock<training::TrainingManager>>,
}

pub async fn create_api_server() -> ActixResult<()> {
    env_logger::init();

    let api_state = web::Data::new(ApiState {
        framework_router: Arc::new(FrameworkRouter::new()),
        model_registry: Arc::new(RwLock::new(models::ModelRegistry::new())),
        auth_service: Arc::new(auth::AuthService::new()),
        training_manager: Arc::new(RwLock::new(training::TrainingManager::new())),
    });

    HttpServer::new(move || {
        App::new()
            .app_data(api_state.clone())
            .wrap(Logger::default())
            .service(
                web::scope("/v1")
                    .service(training::start_training)
                    .service(training::get_training_status)
                    .service(training::stop_training)
                    .service(inference::predict)
                    .service(inference::batch_predict)
                    .service(models::list_models)
                    .service(models::get_model_info)
                    .service(models::upload_model)
                    .service(models::delete_model)
            )
            .service(
                web::scope("/health")
                    .route("", web::get().to(health_check))
            )
    })
    .bind("0.0.0.0:5000")?
    .run()
    .await
}

async fn health_check() -> actix_web::HttpResponse {
    actix_web::HttpResponse::Ok().json(serde_json::json!({
        "status": "healthy",
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "version": env!("CARGO_PKG_VERSION")
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use actix_web::{test, App};

    #[actix_web::test]
    async fn test_health_endpoint() {
        let app = test::init_service(
            App::new().service(
                web::scope("/health")
                    .route("", web::get().to(health_check))
            )
        ).await;

        let req = test::TestRequest::get()
            .uri("/health")
            .to_request();
        
        let resp = test::call_service(&app, req).await;
        assert!(resp.status().is_success());
    }
}