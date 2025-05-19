// Create visualization data
    let mut token_counts = HashMap::new();
    for token in &tokens {
        *token_counts.entry(token).or_insert(0) += 1;
    }
    
    // Create visualization data
    let mut visualizations = Vec::with_capacity(tokens.len());
    
    // Get special tokens to mark them in visualization
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
    
    for (i, token) in tokens.iter().enumerate() {
        let is_special = special_tokens.contains(token);
        let token_id = tokenized.input_ids[i];
        let frequency = *token_counts.get(token).unwrap_or(&0);
        
        visualizations.push(TokenVisualization {
            token: token.clone(),
            token_id,
            is_special,
            frequency: Some(frequency),
            byte_length: token.as_bytes().len(),
            char_length: token.chars().count(),
        });
    }
    
    Ok(visualizations)
}

#[tauri::command]
pub async fn generate_token_frequency_chart(
    text: String,
    tokenizer_id: String,
    state: State<'_, Arc<TokenizerState>>
) -> Result<Vec<TokenFrequency>, String> {
    // Tokenize the text
    let request = serde_json::json!({
        "text": text,
        "tokenizer_id": tokenizer_id,
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
        None => return Err("No tokens available for chart".into()),
    };
    
    // Count token frequencies
    let mut token_counts = HashMap::new();
    for token in &tokens {
        *token_counts.entry(token.clone()).or_insert(0) += 1;
    }
    
    // Calculate total tokens
    let total_tokens = tokens.len() as f64;
    
    // Create frequency data
    let mut frequencies: Vec<TokenFrequency> = token_counts.into_iter()
        .map(|(token, count)| TokenFrequency {
            token,
            count,
            percentage: (count as f64 / total_tokens) * 100.0,
        })
        .collect();
    
    // Sort by frequency
    frequencies.sort_by(|a, b| b.count.cmp(&a.count));
    
    // Limit to top 50
    if frequencies.len() > 50 {
        frequencies.truncate(50);
    }
    
    Ok(frequencies)
}

#[tauri::command]
pub async fn compare_tokenization(
    text: String,
    tokenizer_ids: Vec<String>,
    state: State<'_, Arc<TokenizerState>>
) -> Result<TokenComparison, String> {
    if tokenizer_ids.is_empty() {
        return Err("No tokenizers specified for comparison".into());
    }
    
    let mut comparisons = Vec::with_capacity(tokenizer_ids.len());
    
    for tokenizer_id in &tokenizer_ids {
        // Tokenize with this tokenizer
        let request = serde_json::json!({
            "text": text,
            "tokenizer_id": tokenizer_id,
            "add_special_tokens": false,
            "return_tokens": true
        });
        
        let response = state.client.post(&format!("{}/tokenize", state.api_url))
            .json(&request)
            .send()
            .await
            .map_err(|e| e.to_string())?;
        
        if !response.status().is_success() {
            return Err(format!("API request failed for tokenizer {}: {}", tokenizer_id, response.status()));
        }
        
        let tokenized: TokenizedOutput = response.json()
            .await
            .map_err(|e| e.to_string())?;
        
        // Get tokens
        let tokens = match tokenized.tokens {
            Some(t) => t,
            None => return Err(format!("No tokens available for tokenizer {}", tokenizer_id)),
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
            .filter(|t| t["id"].as_str().unwrap_or("") == *tokenizer_id)
            .flat_map(|t| {
                t["special_tokens"].as_object()
                    .map(|st| st.values().filter_map(|v| v.as_str().map(String::from)).collect())
                    .unwrap_or_default()
            })
            .collect();
        
        // Create comparison items
        let comparison_items: Vec<TokenComparisonItem> = tokens.iter().enumerate()
            .map(|(i, token)| TokenComparisonItem {
                token: token.clone(),
                token_id: tokenized.input_ids[i],
                is_special: special_tokens.contains(token),
                tokenizer_id: tokenizer_id.clone(),
            })
            .collect();
        
        comparisons.push(comparison_items);
    }
    
    Ok(TokenComparison {
        text,
        comparisons,
    })
}}