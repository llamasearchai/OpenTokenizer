use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tauri::State;
use std::sync::Arc;

use crate::tokenization::{TokenizedOutput, TokenizerState};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenVisualization {
    token: String,
    token_id: i32,
    is_special: bool,
    frequency: Option<i32>,
    byte_length: usize,
    char_length: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenFrequency {
    token: String,
    count: i32,
    percentage: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenComparisonItem {
    token: String,
    token_id: i32,
    is_special: bool,
    tokenizer_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenComparison {
    text: String,
    comparisons: Vec<Vec<TokenComparisonItem>>,
}

#[tauri::command]
pub async fn visualize_tokens(
    tokenized: TokenizedOutput,
    state: State<'_, Arc<TokenizerState>>
) -> Result<Vec<TokenVisualization>, String> {
    // Ensure we have tokens to visualize
    let tokens = match &tokenized.tokens {
        Some(t) => t,
        None => return Err("No tokens available for visualization".into()),
    };
    
    // Count token frequencies
    let mut token_counts = HashMap::new();
    for token in tokens {
        *token_counts.entry(token).or_insert(0) += 1;
    }
    
    // Create visualization data
    let special_token_ids = vec![0, 1, 2, 101, 102, 103]; // Sample special token IDs