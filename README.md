# flutter_ai_tester

![flutter_ai_tester poster](https://raw.githubusercontent.com/Codernta/flutter_ai_tester/main/doc/flutter_ai_tester_poster.jpg)

A Flutter & Dart testing toolkit for regression, widget, and integration tests on Generative UI layouts, non-deterministic LLM outputs, and AI token/cost/latency budgets.

## Features

- **Generative UI Layout Testing (`LayoutAsserter`):** Stress-test widget builders across batches of synthetic or LLM-generated JSON payloads and multiple screen viewports to catch `RenderFlex` overflows, container overflows, and build/runtime crashes with pretty-printed failing payload reports.
- **Semantic Assertions (`expectSemanticMatch` / `matchesSemantically`):** Validate non-deterministic LLM text using either an offline deterministic fallback mode (token containment + TF cosine similarity, `RegExp`, keyword checklists, and `SemanticIntent` required/forbidden guardrails) or pluggable `AiSemanticEvaluator`s (`EmbeddingSemanticEvaluator`, `HttpAiSemanticEvaluator`, or custom mocks).
- **Performance & Budget Monitor (`PromptBudget` / `expectWithinBudget`):** Track wall-clock latency, prompt/completion/total tokens, and USD cost in real time with `AiTestMetrics` and fail CI pipelines when thresholds are breached.

## Getting started

Add `flutter_ai_tester` under `dev_dependencies` in your `pubspec.yaml`:

```yaml
dev_dependencies:
  flutter_ai_tester: ^0.1.0
```

Or run:

```sh
flutter pub add --dev flutter_ai_tester
```

Import the single barrel library in your test files:

```dart
import 'package:flutter_ai_tester/flutter_ai_tester.dart';
```

## Usage

### 1. Generative UI Layout Testing

```dart
import 'package:flutter/material.dart';
import 'package:flutter_ai_tester/flutter_ai_tester.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  testWidgets('generative card handles varied LLM payloads', (tester) async {
    final payloads = <Map<String, dynamic>>[
      {'title': 'Order #1024', 'body': 'Delivered today.'},
      {'title': 'Verbose Summary', 'body': 'Long AI response... ' * 50},
    ];

    await LayoutAsserter.testGenerativeLayout(
      tester,
      (json) => MaterialApp(
        home: Scaffold(
          body: SingleChildScrollView(
            child: Column(
              children: [
                Text(json['title'] as String),
                Text(json['body'] as String),
              ],
            ),
          ),
        ),
      ),
      payloads,
      viewports: const [Size(375, 812), Size(800, 600)],
    );
  });
}
```

### 2. Semantic Assertions

```dart
// Offline deterministic mode (async or synchronous Matcher)
await expectSemanticMatch(
  "France's capital city is Paris.",
  'The capital of France is Paris.',
  similarityThreshold: 0.75,
);

expect(
  'Flight confirmation for Tokyo completed.',
  matchesSemantically('flight confirmation Tokyo'),
);

// Structured intent guardrails
await expectSemanticMatch(
  'Your flight booking to Tokyo is confirmed!',
  const SemanticIntent(
    description: 'Flight booking confirmation',
    requiredPatterns: ['confirmed', 'Tokyo'],
    forbiddenPatterns: ['error', 'declined'],
  ),
);

// Pluggable LLM-as-a-judge HTTP evaluator
final evaluator = HttpAiSemanticEvaluator(
  endpoint: Uri.parse('https://api.example.com/v1/chat/completions'),
  model: 'gpt-4o-mini',
  headers: {'Authorization': 'Bearer $apiKey'},
);

await expectSemanticMatch(
  modelOutput,
  'User receives confirmation of subscription cancellation.',
  similarityThreshold: 0.8,
  evaluator: evaluator,
);
```

### 3. Performance & Cost Budget Enforcement

```dart
const pricing = AiTokenPricing(
  promptCostPerMillionTokensUsd: 0.15,
  completionCostPerMillionTokensUsd: 0.60,
);

const budget = PromptBudget(
  maxLatency: Duration(seconds: 2),
  maxPromptTokens: 1000,
  maxCompletionTokens: 500,
  maxCostUsd: 0.01,
);

final tracker = AiTestMetrics(modelName: 'my-llm')..start();
final response = await callMyLlm();
tracker.stop(
  promptTokens: response.promptTokens,
  completionTokens: response.completionTokens,
  pricing: pricing,
);

expectWithinBudget(tracker, budget);
// Or using the matcher syntax:
expect(tracker, isWithinBudget(budget));
```

## Running the Package Tests

```sh
flutter test
```
