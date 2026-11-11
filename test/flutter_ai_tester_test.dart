import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:flutter_ai_tester/flutter_ai_tester.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';

void main() {
  group('LayoutAsserter', () {
    testWidgets('passes when all generative payloads render without overflow', (
      WidgetTester tester,
    ) async {
      final payloads = <Map<String, dynamic>>[
        <String, dynamic>{
          'title': 'Short Summary',
          'body': 'Order #1024 has shipped.',
          'badge': 'Delivered',
        },
        <String, dynamic>{
          'title': 'Detailed AI Breakdown',
          'body': List<String>.filled(
            120,
            'Responsive generative card content that wraps safely inside a scrollable container.',
          ).join(' '),
          'badge': 'In Transit',
        },
      ];

      var callbackCount = 0;
      await testGenerativeLayout(
        tester,
        (Map<String, dynamic> json) => MaterialApp(
          home: Scaffold(
            body: SingleChildScrollView(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: <Widget>[
                  Text(json['title'] as String),
                  const SizedBox(height: 8),
                  Text(json['body'] as String),
                  const SizedBox(height: 8),
                  Chip(label: Text(json['badge'] as String)),
                ],
              ),
            ),
          ),
        ),
        payloads,
        onPayloadRendered: (WidgetTester t, Map<String, dynamic> payload) {
          callbackCount++;
          expect(find.text(payload['title'] as String), findsOneWidget);
        },
      );

      expect(callbackCount, 2);
    });

    testWidgets(
      'catches vertical RenderFlex overflow from massive AI text in unconstrained Column',
      (WidgetTester tester) async {
        final String massiveAiText = List<String>.filled(
          250,
          'LLM hallucinated paragraph causing unbounded vertical expansion.',
        ).join(' ');

        final payloads = <Map<String, dynamic>>[
          <String, dynamic>{
            'id': 'safe-payload',
            'title': 'Brief Answer',
            'body': 'Fits cleanly within viewport.',
          },
          <String, dynamic>{
            'id': 'overflow-payload',
            'title': 'Verbose AI Response',
            'body': massiveAiText,
          },
        ];

        LayoutAssertionFailure? caughtFailure;
        try {
          await LayoutAsserter.testGenerativeLayout(
            tester,
            // Intentionally broken Generative UI widget: Column without
            // SingleChildScrollView or Expanded wrapping a massive Text payload.
            (Map<String, dynamic> json) => MaterialApp(
              home: Scaffold(
                body: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: <Widget>[
                    Text(
                      json['title'] as String,
                      style: const TextStyle(fontSize: 20),
                    ),
                    Text(json['body'] as String),
                  ],
                ),
              ),
            ),
            payloads,
          );
          fail('Expected LayoutAssertionFailure for overflowing Column.');
        } on LayoutAssertionFailure catch (failure) {
          caughtFailure = failure;
        }

        expect(caughtFailure, isA<TestFailure>());
        expect(caughtFailure.totalRuns, 2);
        expect(caughtFailure.failures, hasLength(1));

        final LayoutFailureEntry entry = caughtFailure.failures.single;
        expect(entry.payloadIndex, 1);
        expect(entry.payload['id'], 'overflow-payload');
        expect(
          entry.overflowDiagnostics.isNotEmpty ||
              entry.frameworkException != null,
          isTrue,
        );
        expect(caughtFailure.message, contains('Payload #1'));
        expect(caughtFailure.message, contains('"id": "overflow-payload"'));
        expect(caughtFailure.message, contains('RenderFlex'));
      },
    );

    testWidgets('catches horizontal Row overflow on long single-line label', (
      WidgetTester tester,
    ) async {
      final payloads = <Map<String, dynamic>>[
        <String, dynamic>{'tag': 'AI_GENERATED_UNBROKEN_CHIP_LABEL_' * 8},
      ];

      await expectLater(
        () => LayoutAsserter.testGenerativeLayout(
          tester,
          (Map<String, dynamic> json) => MaterialApp(
            home: Scaffold(
              body: Row(
                children: <Widget>[
                  Text(json['tag'] as String, softWrap: false),
                ],
              ),
            ),
          ),
          payloads,
        ),
        throwsA(
          isA<LayoutAssertionFailure>().having(
            (LayoutAssertionFailure f) => f.message,
            'message',
            allOf(contains('Payload #0'), contains('OVERFLOWING')),
          ),
        ),
      );
    });

    testWidgets('catches builder schema errors from malformed LLM JSON', (
      WidgetTester tester,
    ) async {
      final payloads = <Map<String, dynamic>>[
        <String, dynamic>{'title': 'Valid Title'},
        <String, dynamic>{'title': null, 'unexpectedField': 42},
        <String, dynamic>{'wrongKey': 'Missing title'},
      ];

      LayoutAssertionFailure? failure;
      try {
        await LayoutAsserter.testGenerativeLayout(
          tester,
          (Map<String, dynamic> json) =>
              MaterialApp(home: Scaffold(body: Text(json['title'] as String))),
          payloads,
        );
        fail('Expected LayoutAssertionFailure for malformed JSON.');
      } on LayoutAssertionFailure catch (e) {
        failure = e;
      }

      expect(failure.failures, hasLength(2));
      expect(failure.failures[0].payloadIndex, 1);
      expect(failure.failures[1].payloadIndex, 2);
      expect(failure.message, contains('"unexpectedField": 42'));
      expect(failure.message, contains('"wrongKey": "Missing title"'));
    });

    testWidgets('respects failFast when enabled', (WidgetTester tester) async {
      final payloads = <Map<String, dynamic>>[
        <String, dynamic>{'title': null},
        <String, dynamic>{'title': null},
      ];

      LayoutAssertionFailure? failure;
      try {
        await LayoutAsserter.testGenerativeLayout(
          tester,
          (Map<String, dynamic> json) =>
              MaterialApp(home: Scaffold(body: Text(json['title'] as String))),
          payloads,
          failFast: true,
        );
        fail('Expected LayoutAssertionFailure.');
      } on LayoutAssertionFailure catch (e) {
        failure = e;
      }

      expect(failure.totalRuns, 1);
      expect(failure.failures, hasLength(1));
    });

    testWidgets(
      'detects viewport-specific overflow and restores original view size',
      (WidgetTester tester) async {
        final Size initialPhysicalSize = tester.view.physicalSize;
        final double initialDpr = tester.view.devicePixelRatio;

        final payloads = <Map<String, dynamic>>[
          <String, dynamic>{'label': 'FixedWidthToolbarItemTitle'},
        ];

        LayoutAssertionFailure? failure;
        try {
          await const LayoutAsserter(
            viewports: <Size>[Size(800, 600), Size(180, 400)],
          ).run(
            tester,
            (Map<String, dynamic> json) => MaterialApp(
              home: Scaffold(
                body: Row(
                  children: <Widget>[
                    const SizedBox(width: 140),
                    Text(json['label'] as String, softWrap: false),
                  ],
                ),
              ),
            ),
            payloads,
          );
          fail('Expected LayoutAssertionFailure on narrow viewport.');
        } on LayoutAssertionFailure catch (e) {
          failure = e;
        }

        expect(failure.totalRuns, 2);
        expect(failure.failures, hasLength(1));
        expect(failure.failures.single.viewport, const Size(180, 400));
        expect(failure.message, contains('180x400'));
        expect(tester.view.physicalSize, initialPhysicalSize);
        expect(tester.view.devicePixelRatio, initialDpr);
      },
    );

    testWidgets('throws ArgumentError when payloads list is empty', (
      WidgetTester tester,
    ) async {
      await expectLater(
        () => LayoutAsserter.testGenerativeLayout(
          tester,
          (Map<String, dynamic> _) => const SizedBox.shrink(),
          const <Map<String, dynamic>>[],
        ),
        throwsArgumentError,
      );
    });
  });

  group('SemanticAssertions', () {
    group('Deterministic Fallback Mode', () {
      test('matches semantically equivalent paraphrased strings', () async {
        final SemanticEvaluationResult result = await expectSemanticMatch(
          "France's capital city is Paris.",
          'The capital of France is Paris.',
          similarityThreshold: 0.75,
        );

        expect(result.isMatch, isTrue);
        expect(result.mode, 'deterministic');
        expect(result.score, greaterThanOrEqualTo(0.75));
      });

      test('fails when meaning diverges below threshold', () async {
        await expectLater(
          () => expectSemanticMatch(
            'Berlin is the capital of Germany.',
            'The capital of France is Paris.',
            similarityThreshold: 0.75,
          ),
          throwsA(
            isA<SemanticMatchFailure>().having(
              (SemanticMatchFailure f) => f.message,
              'message',
              allOf(
                contains('deterministic mode'),
                contains('required >= 0.750'),
              ),
            ),
          ),
        );
      });

      test('supports RegExp targets and String extension syntax', () async {
        await 'Your ticket #AB-9482 is confirmed.'.expectSemanticMatch(
          RegExp(r'ticket #[A-Z]{2}-\d{4}'),
        );

        await expectLater(
          () => 'Confirmation pending.'.expectSemanticMatch(
            RegExp(r'ticket #[A-Z]{2}-\d{4}'),
          ),
          throwsA(isA<SemanticMatchFailure>()),
        );
      });

      test('supports keyword checklists and reports missing items', () async {
        await expectSemanticMatch(
          'Your refund for order 4412 will arrive within 5 business days.',
          <String>['refund', '4412', '5 business days'],
        );

        await expectLater(
          () => expectSemanticMatch('Your refund has been initiated.', <String>[
            'refund',
            'order',
            '5 business days',
          ]),
          throwsA(
            isA<SemanticMatchFailure>().having(
              (SemanticMatchFailure f) => f.message,
              'message',
              allOf(contains('"order"'), contains('"5 business days"')),
            ),
          ),
        );
      });

      test(
        'enforces SemanticIntent requiredPatterns and forbiddenPatterns',
        () async {
          const intent = SemanticIntent(
            description: 'Flight booking confirmation',
            requiredPatterns: <Pattern>['confirmed', 'Tokyo'],
            forbiddenPatterns: <Pattern>['error', 'declined'],
          );

          final SemanticEvaluationResult passResult = await expectSemanticMatch(
            'Your flight booking to Tokyo is confirmed!',
            intent,
            similarityThreshold: 0.75,
          );
          expect(passResult.isMatch, isTrue);

          await expectLater(
            () => expectSemanticMatch(
              'Error: Your flight booking to Tokyo was declined even if confirmed earlier.',
              intent,
            ),
            throwsA(
              isA<SemanticMatchFailure>().having(
                (SemanticMatchFailure f) => f.result.score,
                'score',
                0.0,
              ),
            ),
          );
        },
      );

      test('works with synchronous SemanticMatcher', () {
        expect(
          'Flight confirmation for Tokyo completed.',
          matchesSemantically('flight confirmation Tokyo'),
        );
        expect(
          'Unconfirmed flight.',
          isNot(matchesSemantically('Confirm flight')),
        );
        expect(
          'Payment declined.',
          isNot(matchesSemantically('flight confirmation Tokyo')),
        );
      });
    });

    group('Evaluator Mode', () {
      test('delegates to custom AiSemanticEvaluator', () async {
        final evaluator = _FakeSemanticEvaluator(
          score: 0.91,
          isMatch: true,
          rationale: 'Both responses confirm subscription cancellation.',
        );

        final SemanticEvaluationResult result = await expectSemanticMatch(
          'I have ended your monthly plan effective immediately.',
          'Confirm subscription cancellation',
          similarityThreshold: 0.8,
          evaluator: evaluator,
        );

        expect(result.mode, 'fake-evaluator');
        expect(result.score, 0.91);
        expect(evaluator.lastActual, contains('ended your monthly plan'));

        await expectLater(
          expectSemanticMatch(
            'Unable to cancel right now.',
            'Confirm subscription cancellation',
            similarityThreshold: 0.8,
            evaluator: _FakeSemanticEvaluator(
              score: 0.25,
              isMatch: false,
              rationale: 'Refusal does not confirm cancellation.',
            ),
          ),
          throwsA(isA<SemanticMatchFailure>()),
        );
      });

      test(
        'supports completion() with matchesSemantically for Futures',
        () async {
          await expectLater(
            Future<String>.value('Subscription cancelled successfully.'),
            completion(matchesSemantically('Cancel subscription')),
          );
        },
      );

      test(
        'supports EmbeddingSemanticEvaluator with cosine similarity',
        () async {
          final evaluator = EmbeddingSemanticEvaluator(
            embedder: (String text) async {
              if (text.contains('refund') || text.contains('money back')) {
                return <double>[0.9, 0.4, 0.1];
              }
              return <double>[0.0, 0.1, 0.99];
            },
          );

          final SemanticEvaluationResult matchResult =
              await expectSemanticMatch(
                'We sent your money back.',
                'Customer receives a refund.',
                similarityThreshold: 0.85,
                evaluator: evaluator,
              );
          expect(matchResult.score, greaterThan(0.99));

          await expectLater(
            () => expectSemanticMatch(
              'Your password has been reset.',
              'Customer receives a refund.',
              similarityThreshold: 0.85,
              evaluator: evaluator,
            ),
            throwsA(isA<SemanticMatchFailure>()),
          );
        },
      );

      test('supports HttpAiSemanticEvaluator using MockClient', () async {
        final mockClient = MockClient((http.Request request) async {
          final decoded = jsonDecode(request.body) as Map<String, dynamic>;
          expect(decoded['model'], 'judge-model-v1');
          expect(request.headers['Authorization'], 'Bearer test-key');

          return http.Response(
            jsonEncode(<String, Object?>{
              'choices': <Map<String, Object?>>[
                <String, Object?>{
                  'message': <String, Object?>{
                    'content': jsonEncode(<String, Object?>{
                      'score': 0.93,
                      'is_match': true,
                      'rationale': 'Output accurately summarizes the order.',
                    }),
                  },
                },
              ],
            }),
            200,
          );
        });

        final evaluator = HttpAiSemanticEvaluator(
          endpoint: Uri.parse('https://api.example.com/v1/chat/completions'),
          model: 'judge-model-v1',
          headers: const <String, String>{'Authorization': 'Bearer test-key'},
          client: mockClient,
        );

        final SemanticEvaluationResult result = await expectSemanticMatch(
          'Order #99 is arriving Friday.',
          'Inform user about order delivery day.',
          similarityThreshold: 0.8,
          evaluator: evaluator,
        );

        expect(result.score, 0.93);
        expect(result.isMatch, isTrue);
        expect(result.mode, 'http-llm(judge-model-v1)');
      });
    });
  });

  group('BudgetMonitor', () {
    test('passes when metrics are under or equal to budget thresholds', () {
      const budget = PromptBudget(
        maxLatency: Duration(milliseconds: 1200),
        maxPromptTokens: 500,
        maxCompletionTokens: 250,
        maxTotalTokens: 750,
        maxCostUsd: 0.015,
      );

      final underBudget = AiTestMetrics(
        latency: const Duration(milliseconds: 420),
        promptTokens: 300,
        completionTokens: 120,
        costUsd: 0.0042,
        modelName: 'gemini-2.5-flash',
      );

      final atBoundary = AiTestMetrics(
        latency: const Duration(milliseconds: 1200),
        promptTokens: 500,
        completionTokens: 250,
        costUsd: 0.015,
      );

      expectWithinBudget(underBudget, budget);
      expectWithinBudget(atBoundary, budget);
      expect(underBudget, isWithinBudget(budget));
    });

    test('handles IEEE-754 summation rounding cleanly at cost boundary', () {
      const budget = PromptBudget(maxCostUsd: 0.003);
      final metrics = AiTestMetrics()
        ..recordUsage(costUsd: 0.001)
        ..recordUsage(costUsd: 0.002);

      expectWithinBudget(metrics, budget);
    });

    test('throws BudgetExceededFailure detailing each breached threshold', () {
      const budget = PromptBudget(
        maxLatency: Duration(milliseconds: 800),
        maxPromptTokens: 400,
        maxCompletionTokens: 150,
        maxTotalTokens: 500,
        maxCostUsd: 0.005,
      );

      final overBudget = AiTestMetrics(
        latency: const Duration(milliseconds: 1450),
        promptTokens: 520,
        completionTokens: 210,
        costUsd: 0.0125,
        modelName: 'candidate-model-v2',
      );

      BudgetExceededFailure? failure;
      try {
        expectWithinBudget(
          overBudget,
          budget,
          reason: 'Prompt v2 regression gate',
        );
        fail('Expected BudgetExceededFailure.');
      } on BudgetExceededFailure catch (e) {
        failure = e;
      }

      expect(failure, isA<TestFailure>());
      expect(failure.violations, hasLength(5));
      expect(
        failure.violations.map((BudgetViolation v) => v.metric),
        containsAll(<String>[
          'maxLatency',
          'maxPromptTokens',
          'maxCompletionTokens',
          'maxTotalTokens',
          'maxCostUsd',
        ]),
      );
      expect(failure.message, contains('Prompt v2 regression gate'));
      expect(failure.message, contains('candidate-model-v2'));
      expect(failure.message, contains('maxLatency'));
      expect(failure.message, contains('maxPromptTokens'));
      expect(failure.message, contains('maxCompletionTokens'));
      expect(failure.message, contains('maxTotalTokens'));
      expect(failure.message, contains('maxCostUsd'));
    });

    test(
      'supports real-time AiTestMetrics tracking, measure(), and AiTokenPricing',
      () async {
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

        final tracker = AiTestMetrics(modelName: 'live-tracker')..start();
        await Future<void>.delayed(const Duration(milliseconds: 15));
        final Duration lap = tracker.stop(
          promptTokens: 400,
          completionTokens: 200,
          pricing: pricing,
        );

        expect(lap, greaterThanOrEqualTo(const Duration(milliseconds: 10)));
        expect(tracker.isRunning, isFalse);
        expect(tracker.totalTokens, 600);
        expect(tracker.costUsd, closeTo(0.00018, 1e-9));
        expectWithinBudget(tracker, budget);

        final AiTrackedExecution<String> tracked =
            await AiTestMetrics.measure<String>(
              () async {
                await Future<void>.delayed(const Duration(milliseconds: 10));
                return 'Generated UI response';
              },
              pricing: pricing,
              extractor: (String value, Duration elapsed) => AiMetricSample(
                latency: elapsed,
                promptTokens: 800,
                completionTokens: 300,
              ),
            );

        expect(tracked.result, 'Generated UI response');
        expect(tracked.metrics.promptTokens, 800);
        expect(tracked.metrics.completionTokens, 300);
        expect(tracked.metrics.costUsd, closeTo(0.00030, 1e-9));
        expectWithinBudget(tracked.metrics, budget);
      },
    );

    test('fails single-metric latency budget while passing token budgets', () {
      const budget = PromptBudget(
        maxLatency: Duration(milliseconds: 100),
        maxPromptTokens: 1000,
        maxCompletionTokens: 500,
        maxCostUsd: 0.05,
      );

      final slowMetrics = AiTestMetrics(
        latency: const Duration(milliseconds: 250),
        promptTokens: 100,
        completionTokens: 50,
        costUsd: 0.001,
      );

      expect(
        () => expectWithinBudget(slowMetrics, budget),
        throwsA(
          isA<BudgetExceededFailure>()
              .having(
                (BudgetExceededFailure f) => f.violations,
                'violations',
                hasLength(1),
              )
              .having(
                (BudgetExceededFailure f) => f.violations.single.metric,
                'metric',
                'maxLatency',
              ),
        ),
      );
    });
  });
}

class _FakeSemanticEvaluator extends AiSemanticEvaluator {
  _FakeSemanticEvaluator({
    required this.score,
    required this.isMatch,
    this.rationale,
  });

  final double score;
  final bool isMatch;
  final String? rationale;
  String? lastActual;

  @override
  String get name => 'fake-evaluator';

  @override
  Future<SemanticEvaluationResult> evaluate({
    required String actualOutput,
    required dynamic expectedTarget,
    double similarityThreshold = 0.75,
  }) async {
    lastActual = actualOutput;
    return SemanticEvaluationResult(
      score: score,
      isMatch: isMatch,
      mode: name,
      rationale: rationale,
    );
  }
}
