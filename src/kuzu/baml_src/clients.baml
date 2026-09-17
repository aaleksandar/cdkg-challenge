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
    model "gemini-3.7-flash"
    api_key env.GOOGLE_API_KEY
    generationConfig {
      temperature 0
    }
  }
}

// https://docs.boundaryml.com/docs/snippets/clients/retry
//
// Sized for Google's 503 "high demand" spikes, which last well beyond a
// second: the previous two retries at 300 ms burned all three attempts inside
// about a second and failed the run before the spike had a chance to pass.
// Five retries at 1 s, doubling and capped at 30 s, wait up to about a minute
// in total — long enough to ride out a short spike, short enough that a real
// outage still fails the run promptly. Nothing on disk is lost either way:
// ingesting again resumes at tag extraction.
retry_policy Exponential {
  max_retries 5
  strategy {
    type exponential_backoff
    delay_ms 1000
    multiplier 2
    max_delay_ms 30000
  }
}
