use std::sync::Arc;
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};
use tauri::State;

use crate::tokenization::{TokenizerState, TokenizedOutput};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EfficiencyAnalysis {
    pub tokenizer_id: String,
    pub compression_ratio: f64,
    pub tokens_per_word: f64,
    pub chars_per_token: f64,
    pub special_tokens_percentage: f64,
    pub subword_percentage: f64,
    pub singleton_percentage: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompatibilityAnalysis {
    pub text: String,
    pub tokenizer_a_id: String,
    pub tokenizer_b_id: String,
    pub compatibility_score: f64,
    pub common_tokens: Vec<String>,
    pub unique_to_a: Vec<String>,
    pub unique_to_b: Vec<String>,
    pub vocabulary_overlap: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub tokenizer_id: String,
    pub tokens_per_second: f64,
    pub tokenization_time_ms: f64,
    pub batch_tokenization_time_ms: f64,
    pub encoding_time_ms: f64,
    pub decoding_time_ms: f64,
    pub memory_usage_mb: f64,
}

#[tauri::command]
pub async fn analyze_tokenization_efficiency(
    text: String,
    tokenizer_id: String,
    state: State<'_, Arc<TokenizerState>>
) -> Result<EfficiencyAnalysis, String> {
    // Tokenize the text
    let request = serde_json::json!({
        "text": text.clone(),
        "tokenizer_id": tokenizer_id.clone(),
        "add_special_tokens": false,
        "return_tokens": true
    });
    
    let response = state.client.post(&format!("{}/tokenize", state.api_url))
        .json(&request)
        .send()
        .await
        .map_err(|e| e.to_string())?;
    
    if !response.status().is_success() {
        return Err(format!("API request failed: {}", response.status()));
    }
    
    let tokenized: TokenizedOutput = response.json()
        .await
        .map_err(|e| e.to_string())?;
    
    // Get tokens
    let tokens = match tokenized.tokens {
        Some(t) => t,
        None => return Err("No tokens available for analysis".into()),
    };
    
    // Get special tokens
    let special_tokens_response = state.client.get(&format!("{}/tokenizers", state.api_url))
        .send()
        .await
        .map_err(|e| e.to_string())?
        .json::<Vec<serde_json::Value>>()
        .await
        .map_err(|e| e.to_string())?;
    
    let special_tokens: Vec<String> = special_tokens_response.iter()
        .filter(|t| t["id"].as_str().unwrap_or("") == tokenizer_id)
        .flat_map(|t| {
            t["special_tokens"].as_object()
                .map(|st| st.values().filter_map(|v| v.as_str().map(String::from)).collect())
                .unwrap_or_default()
        })
        .collect();
    
    // Count the words in the original text
    let word_count = text.split_whitespace().count();
    
    // Calculate compression ratio (chars in text / token count)
    let compression_ratio = text.chars().count() as f64 / tokens.len() as f64;
    
    // Calculate tokens per word
    let tokens_per_word = if word_count > 0 {
        tokens.len() as f64 / word_count as f64
    } else {
        0.0
    };
    
    // Calculate chars per token
    let chars_per_token = if !tokens.is_empty() {
        text.chars().count() as f64 / tokens.len() as f64
    } else {
        0.0
    };
    
    // Calculate special tokens percentage
    let special_token_count = tokens.iter()
        .filter(|t| special_tokens.contains(t))
        .count();
    let special_tokens_percentage = if !tokens.is_empty() {
        (special_token_count as f64 / tokens.len() as f64) * 100.0
    } else {
        0.0
    };
    
    // Calculate subword percentage (tokens that start with ##, ▁, etc.)
    let subword_indicators = ["##", "▁", "Ġ", "_", "##"];
    let subword_count = tokens.iter()
        .filter(|t| subword_indicators.iter().any(|&prefix| t.starts_with(prefix)))
        .count();
    let subword_percentage = if !tokens.is_empty() {
        (subword_count as f64 / tokens.len() as f64) * 100.0
    } else {
        0.0
    };
    
    // Calculate singleton percentage (tokens that appear only once)
    let mut token_counts = std::collections::HashMap::new();
    for token in &tokens {
        *token_counts.entry(token).or_insert(0) += 1;
    }
    let singleton_count = token_counts.values().filter(|&&count| count == 1).count();
    let singleton_percentage = if !tokens.is_empty() {
        (singleton_count as f64 / tokens.len() as f64) * 100.0
    } else {
        0.0
    };
    
    Ok(EfficiencyAnalysis {
        tokenizer_id,
        compression_ratio,
        tokens_per_word,
        chars_per_token,
        special_tokens_percentage,
        subword_percentage,
        singleton_percentage,
    })
}

#[tauri::command]
pub async fn test_cross_model_compatibility(
    texts: Vec<String>,
    source_tokenizer_id: String,
    target_tokenizer_id: String,
    state: State<'_, Arc<TokenizerState>>
) -> Result<CrossModelCompatibility, String> {
    let mut examples = Vec::new();
    let mut total_matching = 0;
    let mut total_tokens = 0;
    
    for text in &texts {
        // Tokenize with source tokenizer
        let source_request = serde_json::json!({
            "text": text,
            "tokenizer_id": source_tokenizer_id,
            "add_special_tokens": false,
            "return_tokens": true
        });
        
        let source_response = state.client.post(&format!("{}/tokenize", state.api_url))
            .json(&source_request)
            .send()
            .await
            .map_err(|e| e.to_string())?;
        
        if !source_response.status().is_success() {
            return Err(format!("API request failed for source tokenizer: {}", source_response.status()));
        }
        
        let source_result: TokenizedOutput = source_response.json()
            .await
            .map_err(|e| e.to_string())?;
        
        // Tokenize with target tokenizer
        let target_request = serde_json::json!({
            "text": text,
            "tokenizer_id": target_tokenizer_id,
            "add_special_tokens": false,
            "return_tokens": true
        });
        
        let target_response = state.client.post(&format!("{}/tokenize", state.api_url))
            .json(&target_request)
            .send()
            .await
            .map_err(|e| e.to_string())?;
        
        if !target_response.status().is_success() {
            return Err(format!("API request failed for target tokenizer: {}", target_response.status()));
        }
        
        let target_result: TokenizedOutput = target_response.json()
            .await
            .map_err(|e| e.to_string())?;
        
        // Get tokens
        let source_tokens = match &source_result.tokens {
            Some(t) => t.clone(),
            None => return Err("No tokens available for source tokenizer".into()),
        };
        
        let target_tokens = match &target_result.tokens {
            Some(t) => t.clone(),
            None => return Err("No tokens available for target tokenizer".into()),
        };
        
        // Calculate compatibility metrics for this example
        let reconstructed_source = source_tokens.join("");
        let reconstructed_target = target_tokens.join("");
        
        // Simple token-level compatibility (exact matches aren't expected between different tokenizers)
        let mut matching_tokens = 0;
        
        // This is a simplified approach - in a real system, you'd use more sophisticated
        // metrics like BLEU score or semantic similarity
        if reconstructed_source == reconstructed_target {
            matching_tokens = source_tokens.len().min(target_tokens.len());
        }
        
        total_matching += matching_tokens;
        total_tokens += source_tokens.len();
        
        examples.push(CompatibilityExample {
            original_text: text.clone(),
            source_tokens,
            target_tokens,
            matching_tokens,
            total_tokens: source_tokens.len(),
        });
    }
    
    // Calculate overall compatibility score
    let token_preservation = if total_tokens > 0 {
        total_matching as f64 / total_tokens as f64
    } else {
        0.0
    };
    
    // In a real implementation, meaning preservation would use semantic similarity
    // For now, use a placeholder value based on token preservation
    let meaning_preservation = token_preservation * 0.9;
    
    // Overall compatibility score
    let compatibility_score = (token_preservation + meaning_preservation) / 2.0;
    
    Ok(CrossModelCompatibility {
        source_tokenizer: source_tokenizer_id,
        target_tokenizer: target_tokenizer_id,
        compatibility_score,
        token_preservation,
        meaning_preservation,
        examples,
    })
}

#[tauri::command]
pub async fn benchmark_tokenization_speed(
    texts: Vec<String>,
    tokenizer_id: String,
    iterations: usize,
    state: State<'_, Arc<TokenizerState>>
) -> Result<BenchmarkResult, String> {
    // Benchmark single text tokenization
    let sample_text = if !texts.is_empty() {
        texts[0].clone()
    } else {
        return Err("No text provided for benchmarking".into());
    };
    
    // Warm up
    let _ = state.client.post(&format!("{}/tokenize", state.api_url))
        .json(&serde_json::json!({
            "text": sample_text,
            "tokenizer_id": tokenizer_id,
            "add_special_tokens": true,
            "return_tokens": false
        }))
        .send()
        .await
        .map_err(|e| e.to_string())?;
    
    // Benchmark single tokenization
    let mut tokenization_times = Vec::new();
    for _ in 0..iterations {
        let start = Instant::now();
        let _ = state.client.post(&format!("{}/tokenize", state.api_url))
            .json(&serde_json::json!({
                "text": sample_text.clone(),
                "tokenizer_id": tokenizer_id.clone(),
                "add_special_tokens": true,
                "return_tokens": false
            }))
            .send()
            .await
            .map_err(|e| e.to_string())?;
        
        tokenization_times.push(start.elapsed());
    }
    
    // Benchmark batch tokenization
    let mut batch_times = Vec::new();
    for _ in 0..iterations {
        let start = Instant::now();
        let _ = state.client.post(&format!("{}/tokenize_batch", state.api_url))
            .json(&serde_json::json!({
                "texts": texts.clone(),
                "tokenizer_id": tokenizer_id.clone(),
                "add_special_tokens": true,
                "return_tokens": false
            }))
            .send()
            .await
            .map_err(|e| e.to_string())?;
        
        batch_times.push(start.elapsed());
    }
    
    // Benchmark encoding and decoding
    let encode_request = serde_json::json!({
        "text": sample_text,
        "tokenizer_id": tokenizer_id,
        "add_special_tokens": true,
        "return_tokens": false
    });
    
    let encode_response = state.client.post(&format!("{}/tokenize", state.api_url))
        .json(&encode_request)
        .send()
        .await
        .map_err(|e| e.to_string())?;
    
    let encoded: TokenizedOutput = encode_response.json()
        .await
        .map_err(|e| e.to_string())?;
    
    let mut encoding_times = Vec::new();
    let mut decoding_times = Vec::new();
    
    for _ in 0..iterations {
        // Encoding benchmark
        let start = Instant::now();
        let _ = state.client.post(&format!("{}/tokenize", state.api_url))
            .json(&encode_request)
            .send()
            .await
            .map_err(|e| e.to_string())?;
        encoding_times.push(start.elapsed());
        
        // Decoding benchmark
        let start = Instant::now();
        let _ = state.client.post(&format!("{}/decode", state.api_url))
            .json(&serde_json::json!({
                "token_ids": encoded.input_ids.clone(),
                "skip_special_tokens": true,
                "tokenizer_id": tokenizer_id.clone(),
            }))
            .send()
            .await
            .map_err(|e| e.to_string())?;
        decoding_times.push(start.elapsed());
    }
    
    // Calculate average times
    let avg_tokenization_time = tokenization_times.iter()
        .map(|d| d.as_secs_f64() * 1000.0)  // Convert to ms
        .sum::<f64>() / iterations as f64;
    
    let avg_batch_time = batch_times.iter()
        .map(|d| d.as_secs_f64() * 1000.0)  // Convert to ms
        .sum::<f64>() / iterations as f64;
    
    let avg_encoding_time = encoding_times.iter()
        .map(|d| d.as_secs_f64() * 1000.0)  // Convert to ms
        .sum::<f64>() / iterations as f64;
    
    let avg_decoding_time = decoding_times.iter()
        .map(|d| d.as_secs_f64() * 1000.0)  // Convert to ms
        .sum::<f64>() / iterations as f64;
    
    // Calculate tokens per second (using the average time)
    let tokens_per_second = if avg_tokenization_time > 0.0 {
        // Approximating tokens as words * 1.3 for simplicity
        (sample_text.split_whitespace().count() as f64 * 1.3) / (avg_tokenization_time / 1000.0)
    } else {
        0.0
    };
    
    // Get memory usage (this is a placeholder - real implementation would use process stats)
    let memory_usage_mb = 0.0;  // Placeholder
    
    Ok(BenchmarkResult {
        tokenizer_id,
        tokens_per_second,
        tokenization_time_ms: avg_tokenization_time,
        batch_tokenization_time_ms: avg_batch_time,
        encoding_time_ms: avg_encoding_time,
        decoding_time_ms: avg_decoding_time,
        memory_usage_mb,
    })
}