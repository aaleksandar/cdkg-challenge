// Learn more about clients at https://docs.boundaryml.com/docs/snippets/clients/overview

// The single client used by every BAML function in this project.
// Pinned to a specific model, not an alias like "gemini-flash-latest", so the
// model cannot shift underneath the prompts. gemini-2.0-flash was used here
// previously and was retired by Google — it now returns 404 on every call.
// evaluate.py's LLM judge is set to the same model; update both together.
client<llm> GeminiFlash {
  provider google-ai
  retry_policy Exponential
  options {
    model "gemini-3.6-flash"
    api_key env.GOOGLE_API_KEY
    generationConfig {
      temperature 0
    }
  }
}

// https://docs.boundaryml.com/docs/snippets/clients/retry
retry_policy Exponential {
  max_retries 2
  // Strategy is optional
  strategy {
    type exponential_backoff
    delay_ms 300
    multiplier 1.5
    max_delay_ms 10000
  }
}
