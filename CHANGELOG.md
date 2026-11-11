## 0.1.0

* Initial release of `flutter_ai_tester`:
  * Added `LayoutAsserter` and `testGenerativeLayout` for multi-payload and multi-viewport Generative UI overflow/exception detection.
  * Added `expectSemanticMatch`, `SemanticStringAssertions`, `matchesSemantically`, `SemanticIntent`, `DeterministicSemanticEvaluator`, `EmbeddingSemanticEvaluator`, and `HttpAiSemanticEvaluator`.
  * Added `PromptBudget`, `AiTestMetrics`, `AiTokenPricing`, `expectWithinBudget`, and `isWithinBudget` for latency, token, and USD cost budget enforcement.
