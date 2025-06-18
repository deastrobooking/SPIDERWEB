// Authentication and authorization for ML API

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;
use chrono::{DateTime, Utc};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthToken {
    pub user_id: String,
    pub tier: UserTier,
    pub permissions: Vec<Permission>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UserTier {
    Free,
    Pro,
    Enterprise,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Permission {
    Train,
    Inference,
    ModelUpload,
    ModelDownload,
    AdvancedFeatures,
}

#[derive(Debug, Clone)]
pub struct QuotaCheck {
    pub can_train: bool,
    pub current_usage: u32,
    pub limit: u32,
}

pub struct AuthService {
    users: HashMap<String, UserInfo>,
    active_sessions: HashMap<String, SessionInfo>,
}

#[derive(Debug, Clone)]
struct UserInfo {
    id: String,
    tier: UserTier,
    quota_usage: QuotaUsage,
    created_at: DateTime<Utc>,
}

#[derive(Debug, Clone)]
struct QuotaUsage {
    training_jobs_today: u32,
    inference_requests_today: u32,
    storage_gb_used: f64,
}

#[derive(Debug, Clone)]
struct SessionInfo {
    user_id: String,
    created_at: DateTime<Utc>,
    last_activity: DateTime<Utc>,
}

impl AuthService {
    pub fn new() -> Self {
        Self {
            users: HashMap::new(),
            active_sessions: HashMap::new(),
        }
    }

    pub async fn check_training_quota(&self, user_id: &str) -> anyhow::Result<QuotaCheck> {
        let user = self.users.get(user_id)
            .ok_or_else(|| anyhow::anyhow!("User not found"))?;

        let limit = match user.tier {
            UserTier::Free => 5,
            UserTier::Pro => 100,
            UserTier::Enterprise => 1000,
        };

        Ok(QuotaCheck {
            can_train: user.quota_usage.training_jobs_today < limit,
            current_usage: user.quota_usage.training_jobs_today,
            limit,
        })
    }

    pub async fn record_training_start(&self, user_id: &str, job_id: &str) -> anyhow::Result<()> {
        // In a real implementation, this would update the database
        Ok(())
    }
}

impl actix_web::FromRequest for AuthToken {
    type Error = actix_web::Error;
    type Future = std::pin::Pin<Box<dyn std::future::Future<Output = Result<Self, Self::Error>>>>;

    fn from_request(req: &actix_web::HttpRequest, _: &mut actix_web::dev::Payload) -> Self::Future {
        let auth_header = req.headers().get("Authorization").cloned();
        
        Box::pin(async move {
            let header = auth_header
                .and_then(|h| h.to_str().ok())
                .ok_or_else(|| actix_web::error::ErrorUnauthorized("Missing Authorization header"))?;

            if !header.starts_with("Bearer ") {
                return Err(actix_web::error::ErrorUnauthorized("Invalid Authorization format"));
            }

            // Mock authentication - in production, validate JWT token
            let token = &header[7..];
            if token == "demo_token" {
                Ok(AuthToken {
                    user_id: "demo_user".to_string(),
                    tier: UserTier::Pro,
                    permissions: vec![Permission::Train, Permission::Inference],
                })
            } else {
                Err(actix_web::error::ErrorUnauthorized("Invalid token"))
            }
        })
    }
}