import 'dart:convert';
import 'dart:math' as math;

import 'package:collection/collection.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:meta/meta.dart';

/// Structured specification of an expected semantic intent for LLM output
/// assertions.
///
/// Combines a natural language [description] with optional deterministic
/// [requiredPatterns] and [forbiddenPatterns] constraints.
@immutable
class SemanticIntent {
  /// Creates a [SemanticIntent] describing expected meaning and guardrails.
  ///
  /// - [description]: Natural language description of the expected meaning.
  /// - [requiredPatterns]: Keywords (`String`) or `RegExp` patterns that must
  ///   appear in the model output.
  /// - [forbiddenPatterns]: Keywords (`String`) or `RegExp` patterns that must
  ///   not appear in the model output (for example, refusal or error phrases).
  /// - [caseSensitive]: Whether literal string pattern checks are
  ///   case-sensitive.
  const SemanticIntent({
    this.description = '',
    this.requiredPatterns = const <Pattern>[],
    this.forbiddenPatterns = const <Pattern>[],
    this.caseSensitive = false,
  });

  /// Natural language summary of the expected semantic intent.
  final String description;

  /// Literal keywords or [RegExp] patterns required in the actual output.
  final List<Pattern> requiredPatterns;

  /// Literal keywords or [RegExp] patterns prohibited in the actual output.
  final List<Pattern> forbiddenPatterns;

  /// Whether string pattern checks respect character casing.
  final bool caseSensitive;

  @override
  String toString() {
    final parts = <String>[
      if (description.isNotEmpty) 'description: "$description"',
      if (requiredPatterns.isNotEmpty) 'required: $requiredPatterns',
      if (forbiddenPatterns.isNotEmpty) 'forbidden: $forbiddenPatterns',
    ];
    return 'SemanticIntent(${parts.join(', ')})';
  }
}

/// Result returned by an [AiSemanticEvaluator] comparing an LLM response to an
/// expected intent or target.
@immutable
class SemanticEvaluationResult {
  /// Creates a [SemanticEvaluationResult].
  ///
  /// [score] must be between `0.0` and `1.0` inclusive.
  SemanticEvaluationResult({
    required this.score,
    required this.isMatch,
    required this.mode,
    this.rationale,
    Map<String, Object?> metadata = const <String, Object?>{},
  }) : metadata = UnmodifiableMapView<String, Object?>(
         Map<String, Object?>.of(metadata),
       ) {
    if (score.isNaN || score < 0.0 || score > 1.0) {
      throw RangeError.range(
        score,
        0,
        1,
        'score',
        'Semantic similarity score must be between 0.0 and 1.0.',
      );
    }
  }

  /// Normalized semantic similarity or satisfaction score in `[0.0, 1.0]`.
  final double score;

  /// Whether the evaluator considers the actual output a semantic match.
  final bool isMatch;

  /// Identifier for the evaluation strategy (for example, `deterministic`,
  /// `embedding`, or `http-llm`).
  final String mode;

  /// Human-readable explanation of how the score was determined.
  final String? rationale;

  /// Additional diagnostic details (such as matched or missing keywords).
  final Map<String, Object?> metadata;

  @override
  String toString() =>
      'SemanticEvaluationResult(score: ${score.toStringAsFixed(3)}, '
      'isMatch: $isMatch, mode: $mode, rationale: $rationale)';
}

/// Contract for pluggable local or remote semantic evaluators used by
/// [expectSemanticMatch] and [SemanticStringAssertions.expectSemanticMatch].
///
/// Users can extend or implement this class to integrate custom LLM-as-a-judge
/// prompts, local on-device models, or mock evaluators in tests.
abstract class AiSemanticEvaluator {
  /// Base constructor for [AiSemanticEvaluator] implementations.
  const AiSemanticEvaluator();

  /// Display name reported in failure diagnostics and [SemanticEvaluationResult].
  String get name => runtimeType.toString();

  /// Evaluates whether [actualOutput] matches the semantic intent or pattern
  /// specified by [expectedTarget] under [similarityThreshold].
  Future<SemanticEvaluationResult> evaluate({
    required String actualOutput,
    required dynamic expectedTarget,
    double similarityThreshold = 0.75,
  });
}

/// Offline deterministic semantic evaluator relying on normalized token
/// overlap, cosine frequency similarity, regular expressions, and keyword
/// checklists.
class DeterministicSemanticEvaluator extends AiSemanticEvaluator {
  /// Creates a [DeterministicSemanticEvaluator].
  ///
  /// - [stopWords]: Common filler words ignored during natural-language token
  ///   scoring (unless all tokens in the target are stop words).
  /// - [caseSensitive]: Whether literal keyword and token matching preserves
  ///   character casing.
  const DeterministicSemanticEvaluator({
    this.stopWords = defaultStopWords,
    this.caseSensitive = false,
  });

  /// Default English stop words filtered during token-based comparison.
  static const Set<String> defaultStopWords = <String>{
    'a',
    'an',
    'and',
    'are',
    'as',
    'at',
    'be',
    'by',
    'can',
    'could',
    'for',
    'from',
    'had',
    'has',
    'have',
    'in',
    'is',
    'it',
    'its',
    'of',
    'on',
    'or',
    'that',
    'the',
    'this',
    'to',
    'was',
    'were',
    'will',
    'with',
    'would',
  };

  /// Stop words excluded when comparing prose strings.
  final Set<String> stopWords;

  /// Whether string comparisons are case-sensitive.
  final bool caseSensitive;

  @override
  String get name => 'deterministic';

  /// Computes the cosine similarity between two equal-length numeric vectors
  /// [vectorA] and [vectorB].
  ///
  /// Returns `0.0` if either vector has zero magnitude. Throws an
  /// [ArgumentError] if either vector is empty or their lengths do not match.
  static double cosineSimilarity(List<double> vectorA, List<double> vectorB) {
    if (vectorA.isEmpty || vectorB.isEmpty) {
      throw ArgumentError('Embedding vectors must not be empty.');
    }
    if (vectorA.length != vectorB.length) {
      throw ArgumentError(
        'Embedding vectors must have identical dimensions '
        '(${vectorA.length} != ${vectorB.length}).',
      );
    }

    var dot = 0.0;
    var normA = 0.0;
    var normB = 0.0;
    for (var i = 0; i < vectorA.length; i++) {
      final double a = vectorA[i];
      final double b = vectorB[i];
      dot += a * b;
      normA += a * a;
      normB += b * b;
    }

    final double denominator = math.sqrt(normA) * math.sqrt(normB);
    if (denominator == 0.0) {
      return 0.0;
    }
    return (dot / denominator).clamp(-1.0, 1.0);
  }

  @override
  Future<SemanticEvaluationResult> evaluate({
    required String actualOutput,
    required dynamic expectedTarget,
    double similarityThreshold = 0.75,
  }) {
    return Future<SemanticEvaluationResult>.value(
      evaluateSync(
        actualOutput: actualOutput,
        expectedTarget: expectedTarget,
        similarityThreshold: similarityThreshold,
      ),
    );
  }

  /// Synchronously evaluates [actualOutput] against [expectedTarget].
  ///
  /// Supported [expectedTarget] types:
  /// - [RegExp]: Matches via [RegExp.hasMatch].
  /// - [Iterable<Pattern>] (including `List<String>`, `Set<String>`, and
  ///   `List<RegExp>`): Verifies checklist presence and scores the matched
  ///   ratio.
  /// - [SemanticIntent]: Enforces [SemanticIntent.forbiddenPatterns],
  ///   [SemanticIntent.requiredPatterns], and [SemanticIntent.description].
  /// - [String]: Computes normalized substring containment and stemmed token
  ///   overlap / TF cosine similarity.
  SemanticEvaluationResult evaluateSync({
    required String actualOutput,
    required dynamic expectedTarget,
    double similarityThreshold = 0.75,
  }) {
    _validateThreshold(similarityThreshold);

    if (expectedTarget == null) {
      throw ArgumentError.notNull('expectedTarget');
    }

    if (expectedTarget is RegExp) {
      final bool matched = expectedTarget.hasMatch(actualOutput);
      final double score = matched ? 1.0 : 0.0;
      return SemanticEvaluationResult(
        score: score,
        isMatch: matched && score >= similarityThreshold,
        mode: name,
        rationale: matched
            ? 'Output matched RegExp pattern /${expectedTarget.pattern}/.'
            : 'Output did not match RegExp pattern /${expectedTarget.pattern}/.',
        metadata: <String, Object?>{
          'pattern': expectedTarget.pattern,
          'matched': matched,
        },
      );
    }

    if (expectedTarget is SemanticIntent) {
      return _evaluateIntent(
        actualOutput: actualOutput,
        intent: expectedTarget,
        similarityThreshold: similarityThreshold,
      );
    }

    if (expectedTarget is Iterable<Pattern>) {
      return _evaluatePatternChecklist(
        actualOutput: actualOutput,
        patterns: expectedTarget.toList(growable: false),
        similarityThreshold: similarityThreshold,
        preserveCase: caseSensitive,
      );
    }

    if (expectedTarget is Iterable) {
      final patterns = <Pattern>[];
      for (final Object? item in expectedTarget) {
        if (item is Pattern) {
          patterns.add(item);
        } else {
          throw ArgumentError.value(
            expectedTarget,
            'expectedTarget',
            'Iterable targets may only contain String or RegExp Pattern items; '
                'found ${item.runtimeType}.',
          );
        }
      }
      return _evaluatePatternChecklist(
        actualOutput: actualOutput,
        patterns: patterns,
        similarityThreshold: similarityThreshold,
        preserveCase: caseSensitive,
      );
    }

    if (expectedTarget is String) {
      return _evaluateStringSimilarity(
        actualOutput: actualOutput,
        expectedText: expectedTarget,
        similarityThreshold: similarityThreshold,
        preserveCase: caseSensitive,
      );
    }

    throw ArgumentError.value(
      expectedTarget,
      'expectedTarget',
      'Unsupported expectedIntentOrTarget type ${expectedTarget.runtimeType}. '
          'Expected String, RegExp, Iterable<Pattern>, or SemanticIntent.',
    );
  }

  SemanticEvaluationResult _evaluateIntent({
    required String actualOutput,
    required SemanticIntent intent,
    required double similarityThreshold,
  }) {
    final bool preserveCase = intent.caseSensitive || caseSensitive;

    final forbiddenHits = <String>[];
    for (final Pattern pattern in intent.forbiddenPatterns) {
      if (_matchesPattern(actualOutput, pattern, preserveCase: preserveCase)) {
        forbiddenHits.add(_describePattern(pattern));
      }
    }

    if (forbiddenHits.isNotEmpty) {
      return SemanticEvaluationResult(
        score: 0.0,
        isMatch: false,
        mode: name,
        rationale:
            'Output contained forbidden pattern(s): '
            '${forbiddenHits.join(', ')}.',
        metadata: <String, Object?>{'forbiddenMatches': forbiddenHits},
      );
    }

    final bool hasRequired = intent.requiredPatterns.isNotEmpty;
    final bool hasDescription = intent.description.trim().isNotEmpty;

    if (!hasRequired && !hasDescription) {
      throw ArgumentError.value(
        intent,
        'expectedTarget',
        'SemanticIntent must specify a non-empty description or at least one '
            'requiredPattern.',
      );
    }

    SemanticEvaluationResult? checklistResult;
    if (hasRequired) {
      checklistResult = _evaluatePatternChecklist(
        actualOutput: actualOutput,
        patterns: intent.requiredPatterns,
        similarityThreshold: similarityThreshold,
        preserveCase: preserveCase,
      );
    }

    SemanticEvaluationResult? descriptionResult;
    if (hasDescription) {
      descriptionResult = _evaluateStringSimilarity(
        actualOutput: actualOutput,
        expectedText: intent.description,
        similarityThreshold: similarityThreshold,
        preserveCase: preserveCase,
      );
    }

    if (checklistResult != null && descriptionResult != null) {
      final double combinedScore =
          (0.6 * checklistResult.score) + (0.4 * descriptionResult.score);
      final missing =
          (checklistResult.metadata['missing'] as List<String>?) ??
          const <String>[];
      final bool matchedAllRequired = missing.isEmpty;
      final bool isMatch =
          matchedAllRequired && combinedScore >= similarityThreshold;

      return SemanticEvaluationResult(
        score: combinedScore.clamp(0.0, 1.0),
        isMatch: isMatch,
        mode: name,
        rationale: [
          if (!matchedAllRequired)
            'Missing required pattern(s): ${missing.join(', ')}.',
          if (checklistResult.rationale != null) checklistResult.rationale!,
          if (descriptionResult.rationale != null) descriptionResult.rationale!,
        ].join(' '),
        metadata: <String, Object?>{
          ...checklistResult.metadata,
          'descriptionScore': descriptionResult.score,
        },
      );
    }

    return checklistResult ?? descriptionResult!;
  }

  SemanticEvaluationResult _evaluatePatternChecklist({
    required String actualOutput,
    required List<Pattern> patterns,
    required double similarityThreshold,
    required bool preserveCase,
  }) {
    if (patterns.isEmpty) {
      throw ArgumentError.value(
        patterns,
        'expectedTarget',
        'Pattern checklist must contain at least one Pattern.',
      );
    }

    final matched = <String>[];
    final missing = <String>[];
    for (final Pattern pattern in patterns) {
      final String label = _describePattern(pattern);
      if (_matchesPattern(actualOutput, pattern, preserveCase: preserveCase)) {
        matched.add(label);
      } else {
        missing.add(label);
      }
    }

    final double score = matched.length / patterns.length;
    final bool isMatch = score >= similarityThreshold;
    final String rationale = missing.isEmpty
        ? 'Matched all ${patterns.length} pattern(s).'
        : 'Matched ${matched.length}/${patterns.length} pattern(s); '
              'missing: ${missing.join(', ')}.';

    return SemanticEvaluationResult(
      score: score,
      isMatch: isMatch,
      mode: name,
      rationale: rationale,
      metadata: <String, Object?>{
        'matched': matched,
        'missing': missing,
        'matchedCount': matched.length,
        'totalCount': patterns.length,
      },
    );
  }

  SemanticEvaluationResult _evaluateStringSimilarity({
    required String actualOutput,
    required String expectedText,
    required double similarityThreshold,
    required bool preserveCase,
  }) {
    if (expectedText.trim().isEmpty) {
      throw ArgumentError.value(
        expectedText,
        'expectedTarget',
        'Expected target string must not be empty.',
      );
    }

    if (actualOutput.trim().isEmpty) {
      return SemanticEvaluationResult(
        score: 0.0,
        isMatch: false,
        mode: name,
        rationale: 'Actual output is empty.',
      );
    }

    final String normActual = _normalizeText(
      actualOutput,
      preserveCase: preserveCase,
    );
    final String normExpected = _normalizeText(
      expectedText,
      preserveCase: preserveCase,
    );

    if (normActual == normExpected) {
      return SemanticEvaluationResult(
        score: 1.0,
        isMatch: 1.0 >= similarityThreshold,
        mode: name,
        rationale: 'Exact normalized text match.',
      );
    }

    final List<String> expectedTokens = _extractMeaningfulTokens(
      normExpected,
      preserveCase: preserveCase,
    );
    final List<String> actualTokens = _extractMeaningfulTokens(
      normActual,
      preserveCase: preserveCase,
    );

    if (expectedTokens.isEmpty || actualTokens.isEmpty) {
      return SemanticEvaluationResult(
        score: 0.0,
        isMatch: false,
        mode: name,
        rationale: 'No comparable tokens found between actual and expected.',
      );
    }

    final Set<String> expectedSet = expectedTokens.toSet();
    final Set<String> actualSet = actualTokens.toSet();
    final matchedExpected = <String>{};
    final canonicalMap = <String, String>{};

    for (final String expToken in expectedSet) {
      if (actualSet.contains(expToken)) {
        matchedExpected.add(expToken);
        canonicalMap[expToken] = expToken;
        continue;
      }
      for (final String actToken in actualSet) {
        if (_areTokensMorphologicallyEquivalent(expToken, actToken)) {
          matchedExpected.add(expToken);
          canonicalMap.putIfAbsent(actToken, () => expToken);
          break;
        }
      }
    }

    final List<String> canonicalActualTokens = actualTokens
        .map((String token) => canonicalMap[token] ?? token)
        .toList(growable: false);

    final double containment = matchedExpected.length / expectedSet.length;
    final double cosine = _tokenVectorCosine(
      canonicalActualTokens,
      expectedTokens,
    );

    var score = (0.7 * containment) + (0.3 * cosine);
    if (normExpected.isNotEmpty && _containsPhrase(normActual, normExpected)) {
      score = math.max(score, 0.92);
    }
    score = score.clamp(0.0, 1.0);

    final List<String> missingTokens =
        expectedSet.difference(matchedExpected).toList()..sort();

    return SemanticEvaluationResult(
      score: score,
      isMatch: score >= similarityThreshold,
      mode: name,
      rationale:
          'Token containment: ${(containment * 100).toStringAsFixed(1)}%, '
          'TF cosine: ${(cosine * 100).toStringAsFixed(1)}%'
          '${missingTokens.isEmpty ? '' : ', unmatched expected tokens: ${missingTokens.join(', ')}'}.',
      metadata: <String, Object?>{
        'containment': containment,
        'cosineSimilarity': cosine,
        'matchedTokens': matchedExpected.toList()..sort(),
        'unmatchedExpectedTokens': missingTokens,
      },
    );
  }

  bool _matchesPattern(
    String input,
    Pattern pattern, {
    required bool preserveCase,
  }) {
    if (pattern is RegExp) {
      return pattern.hasMatch(input);
    }
    final String needle = pattern.toString();
    if (needle.isEmpty) {
      return false;
    }
    if (preserveCase) {
      return input.contains(needle);
    }
    return input.toLowerCase().contains(needle.toLowerCase());
  }

  String _describePattern(Pattern pattern) {
    if (pattern is RegExp) {
      return '/${pattern.pattern}/';
    }
    return '"$pattern"';
  }

  String _normalizeText(String input, {required bool preserveCase}) {
    final String cased = preserveCase ? input : input.toLowerCase();
    return cased
        .replaceAll(RegExp(r"['’][sS]\b"), '')
        .replaceAll(RegExp(r'[^\p{L}\p{N}\s]+', unicode: true), ' ')
        .replaceAll(RegExp(r'\s+'), ' ')
        .trim();
  }

  bool _containsPhrase(String normalizedHaystack, String normalizedNeedle) {
    if (normalizedHaystack == normalizedNeedle) {
      return true;
    }
    final RegExp boundaryPattern = RegExp(
      '(?:^|\\s)${RegExp.escape(normalizedNeedle)}(?:\$|\\s)',
    );
    return boundaryPattern.hasMatch(normalizedHaystack);
  }

  List<String> _extractMeaningfulTokens(
    String normalizedText, {
    required bool preserveCase,
  }) {
    final List<String> rawTokens = normalizedText
        .split(' ')
        .where((String token) => token.isNotEmpty)
        .toList();

    final filtered = rawTokens
        .where(
          (String token) =>
              !stopWords.contains(preserveCase ? token.toLowerCase() : token),
        )
        .toList();

    final List<String> effective = filtered.isNotEmpty ? filtered : rawTokens;
    return effective.map(_stemToken).toList(growable: false);
  }

  String _stemToken(String token) {
    if (token.length <= 3) {
      return token;
    }
    if (token.endsWith('ies') && token.length > 4) {
      return '${token.substring(0, token.length - 3)}y';
    }
    if (token.endsWith('ing') && token.length > 5) {
      return token.substring(0, token.length - 3);
    }
    if (token.endsWith('ed') && token.length > 4) {
      return token.substring(0, token.length - 2);
    }
    if (token.endsWith('s') &&
        !token.endsWith('ss') &&
        !token.endsWith('is') &&
        !token.endsWith('us')) {
      return token.substring(0, token.length - 1);
    }
    return token;
  }

  bool _areTokensMorphologicallyEquivalent(String a, String b) {
    if (a == b) {
      return true;
    }
    final int minLength = math.min(a.length, b.length);
    final int maxLength = math.max(a.length, b.length);
    if (minLength < 5 || (minLength / maxLength) < 0.5) {
      return false;
    }
    return a.startsWith(b) || b.startsWith(a);
  }

  double _tokenVectorCosine(List<String> tokensA, List<String> tokensB) {
    final vocabulary = <String>{...tokensA, ...tokensB}.toList();
    final countsA = <String, double>{};
    final countsB = <String, double>{};

    for (final String token in tokensA) {
      countsA[token] = (countsA[token] ?? 0.0) + 1.0;
    }
    for (final String token in tokensB) {
      countsB[token] = (countsB[token] ?? 0.0) + 1.0;
    }

    final vectorA = List<double>.generate(
      vocabulary.length,
      (int i) => countsA[vocabulary[i]] ?? 0.0,
    );
    final vectorB = List<double>.generate(
      vocabulary.length,
      (int i) => countsB[vocabulary[i]] ?? 0.0,
    );

    return cosineSimilarity(vectorA, vectorB);
  }
}

/// Semantic evaluator that generates dense embedding vectors via a user-supplied
/// [embedder] callback and scores semantic equivalence using cosine similarity.
class EmbeddingSemanticEvaluator extends AiSemanticEvaluator {
  /// Creates an [EmbeddingSemanticEvaluator].
  ///
  /// - [embedder]: Async callback returning an embedding vector for input text.
  /// - [fallbackEvaluator]: Used when [expectedTarget] is a [RegExp] or
  ///   `Iterable<Pattern>`, and for enforcing [SemanticIntent] keyword
  ///   constraints.
  const EmbeddingSemanticEvaluator({
    required this.embedder,
    this.fallbackEvaluator = const DeterministicSemanticEvaluator(),
  });

  /// Embedding generator callback (e.g., Gemini, OpenAI, or local model).
  final Future<List<double>> Function(String text) embedder;

  /// Deterministic evaluator used for regex/checklist constraints.
  final DeterministicSemanticEvaluator fallbackEvaluator;

  @override
  String get name => 'embedding';

  @override
  Future<SemanticEvaluationResult> evaluate({
    required String actualOutput,
    required dynamic expectedTarget,
    double similarityThreshold = 0.75,
  }) async {
    _validateThreshold(similarityThreshold);

    if (expectedTarget is RegExp || expectedTarget is Iterable) {
      return fallbackEvaluator.evaluate(
        actualOutput: actualOutput,
        expectedTarget: expectedTarget,
        similarityThreshold: similarityThreshold,
      );
    }

    String targetText;
    if (expectedTarget is SemanticIntent) {
      if (expectedTarget.forbiddenPatterns.isNotEmpty ||
          expectedTarget.requiredPatterns.isNotEmpty) {
        final SemanticEvaluationResult guardrailResult = fallbackEvaluator
            .evaluateSync(
              actualOutput: actualOutput,
              expectedTarget: SemanticIntent(
                description: expectedTarget.description,
                requiredPatterns: expectedTarget.requiredPatterns,
                forbiddenPatterns: expectedTarget.forbiddenPatterns,
                caseSensitive: expectedTarget.caseSensitive,
              ),
              similarityThreshold: 0.0,
            );
        final forbiddenHits =
            guardrailResult.metadata['forbiddenMatches'] as List<String>?;
        final missing = guardrailResult.metadata['missing'] as List<String>?;
        if (forbiddenHits != null && forbiddenHits.isNotEmpty) {
          return SemanticEvaluationResult(
            score: 0.0,
            isMatch: false,
            mode: name,
            rationale: guardrailResult.rationale,
            metadata: guardrailResult.metadata,
          );
        }
        if (missing != null && missing.isNotEmpty) {
          return SemanticEvaluationResult(
            score: guardrailResult.score,
            isMatch: false,
            mode: name,
            rationale: guardrailResult.rationale,
            metadata: guardrailResult.metadata,
          );
        }
      }
      targetText = expectedTarget.description;
    } else if (expectedTarget is String) {
      targetText = expectedTarget;
    } else {
      throw ArgumentError.value(
        expectedTarget,
        'expectedTarget',
        'Unsupported expectedTarget type ${expectedTarget.runtimeType}.',
      );
    }

    if (targetText.trim().isEmpty) {
      return fallbackEvaluator.evaluate(
        actualOutput: actualOutput,
        expectedTarget: expectedTarget,
        similarityThreshold: similarityThreshold,
      );
    }

    final List<List<double>> embeddings = await Future.wait(
      <Future<List<double>>>[embedder(actualOutput), embedder(targetText)],
    );

    final double rawCosine = DeterministicSemanticEvaluator.cosineSimilarity(
      embeddings[0],
      embeddings[1],
    );
    final double normalizedScore = rawCosine.clamp(0.0, 1.0);

    return SemanticEvaluationResult(
      score: normalizedScore,
      isMatch: normalizedScore >= similarityThreshold,
      mode: name,
      rationale:
          'Embedding cosine similarity: ${rawCosine.toStringAsFixed(4)}.',
      metadata: <String, Object?>{
        'rawCosineSimilarity': rawCosine,
        'dimensions': embeddings[0].length,
      },
    );
  }
}

/// HTTP-backed LLM-as-a-judge semantic evaluator implementing
/// [AiSemanticEvaluator].
///
/// Works out of the box with OpenAI-compatible chat completion APIs or custom
/// evaluation endpoints via [requestBodyBuilder] and [responseParser].
class HttpAiSemanticEvaluator extends AiSemanticEvaluator {
  /// Creates an [HttpAiSemanticEvaluator].
  ///
  /// - [endpoint]: Target HTTP endpoint URL for the evaluating LLM service.
  /// - [model]: Model identifier passed in the default JSON request body.
  /// - [headers]: HTTP headers (such as `Authorization: Bearer ...`).
  /// - [timeout]: Maximum duration to wait for an HTTP response.
  /// - [client]: Optional [http.Client] for connection reuse or test mocking.
  /// - [requestBodyBuilder]: Optional custom serializer returning the JSON
  ///   payload sent to [endpoint].
  /// - [responseParser]: Optional custom response parser converting the
  ///   [http.Response] into a [SemanticEvaluationResult].
  const HttpAiSemanticEvaluator({
    required this.endpoint,
    required this.model,
    this.headers = const <String, String>{},
    this.timeout = const Duration(seconds: 30),
    this.client,
    this.requestBodyBuilder,
    this.responseParser,
  });

  /// Remote or local evaluation endpoint URI.
  final Uri endpoint;

  /// Model identifier included in default evaluation requests.
  final String model;

  /// Additional HTTP headers sent with each request.
  final Map<String, String> headers;

  /// Request timeout duration.
  final Duration timeout;

  /// Optional injected HTTP client (useful for `MockClient` in tests).
  final http.Client? client;

  /// Optional custom request body builder.
  final Map<String, Object?> Function(
    String actualOutput,
    String expectedTarget,
    String model,
    double threshold,
  )?
  requestBodyBuilder;

  /// Optional custom response parser.
  final SemanticEvaluationResult Function(
    http.Response response,
    double threshold,
  )?
  responseParser;

  @override
  String get name => 'http-llm($model)';

  @override
  Future<SemanticEvaluationResult> evaluate({
    required String actualOutput,
    required dynamic expectedTarget,
    double similarityThreshold = 0.75,
  }) async {
    _validateThreshold(similarityThreshold);

    final String targetDescription = switch (expectedTarget) {
      SemanticIntent() => expectedTarget.toString(),
      RegExp() => 'Must match regular expression: /${expectedTarget.pattern}/',
      Iterable() =>
        'Must satisfy all items: ${expectedTarget.map((Object? e) => e.toString()).join(', ')}',
      String() => expectedTarget,
      _ => expectedTarget.toString(),
    };

    final Map<String, Object?> payload = requestBodyBuilder != null
        ? requestBodyBuilder!(
            actualOutput,
            targetDescription,
            model,
            similarityThreshold,
          )
        : _defaultRequestBody(
            actualOutput,
            targetDescription,
            model,
            similarityThreshold,
          );

    final http.Client activeClient = client ?? http.Client();
    final bool shouldCloseClient = client == null;

    try {
      final http.Response response = await activeClient
          .post(
            endpoint,
            headers: <String, String>{
              'Content-Type': 'application/json',
              ...headers,
            },
            body: jsonEncode(payload),
          )
          .timeout(timeout);

      if (response.statusCode < 200 || response.statusCode >= 300) {
        throw StateError(
          'HttpAiSemanticEvaluator received HTTP ${response.statusCode} '
          'from $endpoint: ${response.body}',
        );
      }

      if (responseParser != null) {
        return responseParser!(response, similarityThreshold);
      }
      return _defaultResponseParser(response, similarityThreshold);
    } finally {
      if (shouldCloseClient) {
        activeClient.close();
      }
    }
  }

  Map<String, Object?> _defaultRequestBody(
    String actualOutput,
    String expectedTarget,
    String model,
    double threshold,
  ) {
    return <String, Object?>{
      'model': model,
      'temperature': 0.0,
      'messages': <Map<String, String>>[
        <String, String>{
          'role': 'system',
          'content':
              'You are a strict semantic test evaluator. Respond ONLY with a '
              'JSON object containing keys: "score" (number from 0.0 to 1.0), '
              '"is_match" (boolean, true if score >= $threshold), and '
              '"rationale" (concise string explanation).',
        },
        <String, String>{
          'role': 'user',
          'content':
              'Expected intent or target:\n$expectedTarget\n\n'
              'Actual model output:\n$actualOutput',
        },
      ],
    };
  }

  SemanticEvaluationResult _defaultResponseParser(
    http.Response response,
    double threshold,
  ) {
    final Object? decoded = jsonDecode(response.body);
    if (decoded is! Map<String, dynamic>) {
      throw FormatException(
        'Expected JSON object from semantic evaluator endpoint, got '
        '${decoded.runtimeType}.',
      );
    }

    Map<String, dynamic> evaluationJson = decoded;

    // Unwrap OpenAI-compatible chat completion envelope if present.
    final Object? choices = decoded['choices'];
    if (choices is List && choices.isNotEmpty) {
      final Object? firstChoice = choices.first;
      if (firstChoice is Map<String, dynamic>) {
        final Object? message = firstChoice['message'];
        if (message is Map<String, dynamic>) {
          final Object? content = message['content'];
          if (content is String) {
            final String cleaned = _stripCodeFences(content);
            final Object? inner = jsonDecode(cleaned);
            if (inner is Map<String, dynamic>) {
              evaluationJson = inner;
            }
          }
        }
      }
    }

    final Object? rawScore = evaluationJson['score'];
    if (rawScore is! num) {
      throw FormatException(
        'Semantic evaluator response missing numeric "score": ${response.body}',
      );
    }

    final double score = rawScore.toDouble().clamp(0.0, 1.0);
    final Object? rawMatch = evaluationJson['is_match'];
    final bool isMatch = rawMatch is bool ? rawMatch : score >= threshold;
    final String? rationale = evaluationJson['rationale'] as String?;

    return SemanticEvaluationResult(
      score: score,
      isMatch: isMatch && score >= threshold,
      mode: name,
      rationale: rationale,
      metadata: <String, Object?>{
        'endpoint': endpoint.toString(),
        'model': model,
      },
    );
  }

  String _stripCodeFences(String text) {
    final String trimmed = text.trim();
    if (trimmed.startsWith('```')) {
      final List<String> lines = trimmed.split('\n');
      if (lines.length >= 3) {
        return lines.sublist(1, lines.length - 1).join('\n').trim();
      }
    }
    return trimmed;
  }
}

/// Test failure thrown when [expectSemanticMatch] fails to satisfy the
/// required similarity threshold or semantic constraints.
class SemanticMatchFailure extends TestFailure {
  /// Creates a [SemanticMatchFailure] with full comparison diagnostics.
  SemanticMatchFailure({
    required this.actualOutput,
    required this.expectedTarget,
    required this.similarityThreshold,
    required this.result,
    this.reason,
  }) : super(
         _buildMessage(
           actualOutput: actualOutput,
           expectedTarget: expectedTarget,
           similarityThreshold: similarityThreshold,
           result: result,
           reason: reason,
         ),
       );

  /// Actual string generated by the LLM or widget under test.
  final String actualOutput;

  /// Expected semantic intent, target string, regex, or keyword checklist.
  final dynamic expectedTarget;

  /// Minimum required similarity score in `[0.0, 1.0]`.
  final double similarityThreshold;

  /// Full evaluation result returned by the active [AiSemanticEvaluator].
  final SemanticEvaluationResult result;

  /// Optional caller-supplied reason string.
  final String? reason;

  static String _buildMessage({
    required String actualOutput,
    required dynamic expectedTarget,
    required double similarityThreshold,
    required SemanticEvaluationResult result,
    required String? reason,
  }) {
    final buffer = StringBuffer()
      ..writeln('Semantic match assertion failed (${result.mode} mode).');

    if (reason != null && reason.isNotEmpty) {
      buffer.writeln('Reason: $reason');
    }

    buffer
      ..writeln(
        'Score: ${result.score.toStringAsFixed(3)} '
        '(required >= ${similarityThreshold.toStringAsFixed(3)}, '
        'isMatch: ${result.isMatch})',
      )
      ..writeln('Expected: $expectedTarget')
      ..writeln('Actual:   ${_truncate(actualOutput)}');

    if (result.rationale != null && result.rationale!.isNotEmpty) {
      buffer.writeln('Details:  ${result.rationale}');
    }

    return buffer.toString().trimRight();
  }

  static String _truncate(String input, {int maxLength = 500}) {
    if (input.length <= maxLength) {
      return '"$input"';
    }
    return '"${input.substring(0, maxLength)}..." (${input.length} chars total)';
  }
}

/// Asserts that [actualOutput] semantically matches [expectedIntentOrTarget].
///
/// Operates in two modes:
/// 1. **Deterministic Fallback Mode** (when [evaluator] is `null`): uses
///    [DeterministicSemanticEvaluator] for token overlap / TF cosine similarity,
///    regex matching, keyword checklists, or [SemanticIntent] constraints.
/// 2. **Evaluator Mode** (when [evaluator] is provided): delegates evaluation
///    to the pluggable [AiSemanticEvaluator] (e.g., [HttpAiSemanticEvaluator],
///    [EmbeddingSemanticEvaluator], or a custom/mock implementation).
///
/// Parameters:
/// - [actualOutput]: The non-deterministic string produced by an LLM.
/// - [expectedIntentOrTarget]: Expected [String], [RegExp], `Iterable<Pattern>`
///   keyword checklist, or [SemanticIntent].
/// - [similarityThreshold]: Minimum similarity score in `[0.0, 1.0]` required
///   to pass (defaults to `0.75`).
/// - [evaluator]: Optional pluggable [AiSemanticEvaluator] instance.
/// - [reason]: Optional diagnostic message included when the assertion fails.
///
/// Returns the [SemanticEvaluationResult] on success, or throws
/// [SemanticMatchFailure] (a [TestFailure]) on mismatch.
Future<SemanticEvaluationResult> expectSemanticMatch(
  String actualOutput,
  dynamic expectedIntentOrTarget, {
  double similarityThreshold = 0.75,
  AiSemanticEvaluator? evaluator,
  String? reason,
}) async {
  _validateThreshold(similarityThreshold);

  final AiSemanticEvaluator activeEvaluator =
      evaluator ?? const DeterministicSemanticEvaluator();

  final SemanticEvaluationResult result = await activeEvaluator.evaluate(
    actualOutput: actualOutput,
    expectedTarget: expectedIntentOrTarget,
    similarityThreshold: similarityThreshold,
  );

  if (!result.isMatch || result.score < similarityThreshold) {
    throw SemanticMatchFailure(
      actualOutput: actualOutput,
      expectedTarget: expectedIntentOrTarget,
      similarityThreshold: similarityThreshold,
      result: result,
      reason: reason,
    );
  }

  return result;
}

/// Fluent extension adding [expectSemanticMatch] directly onto [String] values.
extension SemanticStringAssertions on String {
  /// Asserts that this string semantically matches [expectedIntentOrTarget].
  ///
  /// Delegates directly to top-level [expectSemanticMatch].
  Future<SemanticEvaluationResult> expectSemanticMatch(
    dynamic expectedIntentOrTarget, {
    double similarityThreshold = 0.75,
    AiSemanticEvaluator? evaluator,
    String? reason,
  }) {
    return _expectSemanticMatchTopLevel(
      this,
      expectedIntentOrTarget,
      similarityThreshold: similarityThreshold,
      evaluator: evaluator,
      reason: reason,
    );
  }
}

Future<SemanticEvaluationResult> _expectSemanticMatchTopLevel(
  String actualOutput,
  dynamic expectedIntentOrTarget, {
  required double similarityThreshold,
  AiSemanticEvaluator? evaluator,
  String? reason,
}) {
  return expectSemanticMatch(
    actualOutput,
    expectedIntentOrTarget,
    similarityThreshold: similarityThreshold,
    evaluator: evaluator,
    reason: reason,
  );
}

/// Synchronous deterministic [Matcher] for use with `expect(actual, matcher)`.
///
/// Prefer [matchesSemantically] to construct instances concisely.
class SemanticMatcher extends Matcher {
  /// Creates a synchronous [SemanticMatcher].
  SemanticMatcher(
    this.expectedIntentOrTarget, {
    this.similarityThreshold = 0.75,
    this.evaluator = const DeterministicSemanticEvaluator(),
  }) {
    _validateThreshold(similarityThreshold);
  }

  /// Expected [String], [RegExp], `Iterable<Pattern>`, or [SemanticIntent].
  final dynamic expectedIntentOrTarget;

  /// Minimum acceptable similarity score in `[0.0, 1.0]`.
  final double similarityThreshold;

  /// Deterministic evaluator used to compute the match synchronously.
  final DeterministicSemanticEvaluator evaluator;

  @override
  bool matches(Object? item, Map<dynamic, dynamic> matchState) {
    if (item is! String) {
      matchState['error'] = 'Expected a String, got ${item.runtimeType}.';
      return false;
    }

    final SemanticEvaluationResult result = evaluator.evaluateSync(
      actualOutput: item,
      expectedTarget: expectedIntentOrTarget,
      similarityThreshold: similarityThreshold,
    );
    matchState['result'] = result;
    return result.isMatch && result.score >= similarityThreshold;
  }

  @override
  Description describe(Description description) {
    return description.add(
      'semantically matches $expectedIntentOrTarget '
      '(threshold >= ${similarityThreshold.toStringAsFixed(2)})',
    );
  }

  @override
  Description describeMismatch(
    Object? item,
    Description mismatchDescription,
    Map<dynamic, dynamic> matchState,
    bool verbose,
  ) {
    final Object? error = matchState['error'];
    if (error != null) {
      return mismatchDescription.add(error.toString());
    }
    final result = matchState['result'] as SemanticEvaluationResult?;
    if (result == null) {
      return mismatchDescription.add('did not match.');
    }
    return mismatchDescription.add(
      'scored ${result.score.toStringAsFixed(3)} '
      '(${result.rationale ?? 'below threshold'})',
    );
  }
}

/// Returns a synchronous [Matcher] verifying deterministic semantic similarity
/// against [expectedIntentOrTarget].
///
/// Works directly with `expect(output, matchesSemantically(...))` as well as
/// `await expectLater(futureOutput, completion(matchesSemantically(...)))`.
Matcher matchesSemantically(
  dynamic expectedIntentOrTarget, {
  double similarityThreshold = 0.75,
  DeterministicSemanticEvaluator evaluator =
      const DeterministicSemanticEvaluator(),
}) {
  return SemanticMatcher(
    expectedIntentOrTarget,
    similarityThreshold: similarityThreshold,
    evaluator: evaluator,
  );
}

void _validateThreshold(double similarityThreshold) {
  if (similarityThreshold.isNaN ||
      similarityThreshold < 0.0 ||
      similarityThreshold > 1.0) {
    throw RangeError.range(
      similarityThreshold,
      0,
      1,
      'similarityThreshold',
      'similarityThreshold must be between 0.0 and 1.0.',
    );
  }
}
