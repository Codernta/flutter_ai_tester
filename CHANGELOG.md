## 0.1.2

* Security hardening and attack prevention across all evaluation and layout pipelines:
  * **SSRF Prevention & URL Validation**: Enforced strict `http`/`https` scheme requirements and non-empty host checks in `HttpAiSemanticEvaluator`.
  * **HTTP Header Injection Defense**: Added validation preventing CRLF (`\r`, `\n`) injection in custom HTTP headers.
  * **Credential Redaction in Logs**: Automatically redacts sensitive query parameters (e.g. `key`, `api_key`, `token`, `auth`, `secret`) and user authentication info (`***:***`) from endpoint URIs in exception messages and metadata.
  * **Prompt Injection Defense**: Isolated model inputs with structural boundary XML tags and added explicit security instructions warning the LLM-as-a-judge to ignore adversarial override commands within untrusted actual output.
  * **Robust JSON Extraction & Error Truncation**: Added resilient JSON extraction for conversational LLM outputs with markdown fences, and truncated HTTP error bodies to prevent memory bloat and log exhaustion.
  * **Viewport & Layout Bounds Enforcement**: Added strict validation for viewport dimensions (requiring positive, finite width/height) and pump durations in `LayoutAsserter`.
  * **Diagnostic Sanitization**: Automatically redacts sensitive payload keys (e.g. passwords, tokens, API keys, credit cards) and truncates massive payloads in `LayoutFailureEntry` format output.
  * **Budget Bypass Prevention**: Validated non-negative finite pricing rates in `AiTokenPricing`, and guarded against infinite/NaN budget thresholds in `PromptBudget` and `AiTestMetrics`.
  * **Vector & Numerical Safety**: Enforced finite number constraints in `cosineSimilarity` to prevent NaN and Infinity pollution.

## 0.1.1

* Added runnable `example/main.dart` demonstrating Generative UI layout testing, semantic assertions, and AI budget enforcement.
* Updated package documentation with example execution instructions.

## 0.1.0

* Initial release of `flutter_ai_tester`:
  * Added `LayoutAsserter` and `testGenerativeLayout` for multi-payload and multi-viewport Generative UI overflow/exception detection.
  * Added `expectSemanticMatch`, `SemanticStringAssertions`, `matchesSemantically`, `SemanticIntent`, `DeterministicSemanticEvaluator`, `EmbeddingSemanticEvaluator`, and `HttpAiSemanticEvaluator`.
  * Added `PromptBudget`, `AiTestMetrics`, `AiTokenPricing`, `expectWithinBudget`, and `isWithinBudget` for latency, token, and USD cost budget enforcement.
