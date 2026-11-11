import 'package:flutter/material.dart';
import 'package:flutter_ai_tester/flutter_ai_tester.dart';
import 'package:flutter_test/flutter_test.dart';

/// Runnable example demonstrating `flutter_ai_tester` across Generative UI
/// layout testing, semantic output matching, and AI budget enforcement.
///
/// Run with: `flutter test example/main.dart`
void main() {
  testWidgets(
    'stress-tests a Generative UI card across payloads and viewports',
    (WidgetTester tester) async {
      final payloads = <Map<String, dynamic>>[
        <String, dynamic>{
          'title': 'Flight Confirmed',
          'summary': 'Your itinerary to Tokyo departs Friday at 09:30.',
        },
        <String, dynamic>{
          'title': 'Detailed AI Travel Brief',
          'summary': List<String>.filled(
            40,
            'Terminal, baggage, and transit instructions wrapped safely.',
          ).join(' '),
        },
      ];

      await LayoutAsserter.testGenerativeLayout(
        tester,
        (Map<String, dynamic> json) => MaterialApp(
          home: Scaffold(
            body: SingleChildScrollView(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: <Widget>[
                  Text(
                    json['title'] as String,
                    style: const TextStyle(fontSize: 20),
                  ),
                  const SizedBox(height: 8),
                  Text(json['summary'] as String),
                ],
              ),
            ),
          ),
        ),
        payloads,
        viewports: const <Size>[Size(375, 812), Size(800, 600)],
      );
    },
  );

  test(
    'validates non-deterministic LLM text and enforces prompt budgets',
    () async {
      const llmResponse =
          'Your flight booking to Tokyo is confirmed for Friday morning.';

      await expectSemanticMatch(
        llmResponse,
        const SemanticIntent(
          description: 'Flight booking to Tokyo confirmed',
          requiredPatterns: <Pattern>['Tokyo', 'confirmed'],
          forbiddenPatterns: <Pattern>['error', 'cancelled'],
        ),
        similarityThreshold: 0.75,
      );

      expect(
        llmResponse,
        matchesSemantically('Flight booking to Tokyo confirmed'),
      );

      const pricing = AiTokenPricing(
        promptCostPerMillionTokensUsd: 0.15,
        completionCostPerMillionTokensUsd: 0.60,
      );
      const budget = PromptBudget(
        maxLatency: Duration(seconds: 2),
        maxPromptTokens: 1000,
        maxCompletionTokens: 400,
        maxCostUsd: 0.01,
      );

      final metrics = AiTestMetrics(modelName: 'example-model')..start();
      await Future<void>.delayed(const Duration(milliseconds: 5));
      metrics.stop(promptTokens: 320, completionTokens: 90, pricing: pricing);

      expectWithinBudget(metrics, budget);
      expect(metrics, isWithinBudget(budget));
    },
  );
}
