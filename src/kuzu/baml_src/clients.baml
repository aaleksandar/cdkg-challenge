// Learn more about clients at https://docs.boundaryml.com/docs/snippets/clients/overview

// The client used by ingestion: tag extraction and speaker recovery.
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

// The client the chat answers with: RAGText2Cypher and RAGAnswerQuestion.
// Same model as GeminiFlash, tuned for someone waiting on an answer rather
// than for a background ingestion run. Gemini 3.x thinks before it answers
// unless told otherwise, and both calls run back to back on every question, so
// the default thinking level cost the chat most of a minute per answer.
// Writing one Cypher statement and summarising a few rows need little of it.
// The retry policy is short for the same reason: a user watching a spinner is
// better served by an error to retry than by a minute of backoff.
// rag.py can route both calls back through GeminiFlash (RAG_CLIENT), which is
// how evaluate.py compares the two.
client<llm> GeminiFlashChat {
  provider google-ai
  retry_policy Brief
  options {
    model "gemini-3.7-flash"
    api_key env.GOOGLE_API_KEY
    generationConfig {
      temperature 0
      thinkingConfig {
        thinkingLevel "low"
      }
    }
  }
}

retry_policy Brief {
  max_retries 2
  strategy {
    type exponential_backoff
    delay_ms 500
    multiplier 2
    max_delay_ms 2000
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
