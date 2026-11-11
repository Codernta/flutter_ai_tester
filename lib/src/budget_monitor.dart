import 'package:collection/collection.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:meta/meta.dart';

/// Pricing model used to compute USD inference cost from token counts.
@immutable
class AiTokenPricing {
  /// Creates an [AiTokenPricing] table using per-million-token USD rates.
  const AiTokenPricing({
    required this.promptCostPerMillionTokensUsd,
    required this.completionCostPerMillionTokensUsd,
  });

  /// USD cost per 1,000,000 input/prompt tokens.
  final double promptCostPerMillionTokensUsd;

  /// USD cost per 1,000,000 output/completion tokens.
  final double completionCostPerMillionTokensUsd;

  /// Calculates total inference cost in USD for [promptTokens] and
  /// [completionTokens].
  double calculateCostUsd({
    required int promptTokens,
    required int completionTokens,
  }) {
    if (promptTokens < 0 || completionTokens < 0) {
      throw ArgumentError('Token counts must be non-negative.');
    }
    return ((promptTokens * promptCostPerMillionTokensUsd) +
            (completionTokens * completionCostPerMillionTokensUsd)) /
        1000000.0;
  }
}

/// Performance and cost ceiling configuration for AI regression pipelines.
///
/// Any `null` threshold is treated as unconstrained.
@immutable
class PromptBudget {
  /// Creates a [PromptBudget] configuration.
  ///
  /// - [maxLatency]: Maximum acceptable end-to-end execution [Duration].
  /// - [maxPromptTokens]: Maximum acceptable input/prompt token count.
  /// - [maxCompletionTokens]: Maximum acceptable output/completion token count.
  /// - [maxTotalTokens]: Optional ceiling on combined `prompt + completion`
  ///   tokens.
  /// - [maxCostUsd]: Maximum acceptable estimated cost in US dollars.
  const PromptBudget({
    this.maxLatency,
    this.maxPromptTokens,
    this.maxCompletionTokens,
    this.maxTotalTokens,
    this.maxCostUsd,
  });

  /// Maximum allowed latency for the tested AI call or workflow.
  final Duration? maxLatency;

  /// Maximum allowed prompt (input) tokens.
  final int? maxPromptTokens;

  /// Maximum allowed completion (output) tokens.
  final int? maxCompletionTokens;

  /// Optional maximum allowed total (`prompt + completion`) tokens.
  final int? maxTotalTokens;

  /// Maximum allowed cost in USD.
  final double? maxCostUsd;

  /// Validates that all configured budget bounds are non-negative and finite.
  void validate() {
    if (maxLatency != null && maxLatency!.isNegative) {
      throw ArgumentError.value(
        maxLatency,
        'maxLatency',
        'maxLatency must not be negative.',
      );
    }
    if (maxPromptTokens != null && maxPromptTokens! < 0) {
      throw ArgumentError.value(
        maxPromptTokens,
        'maxPromptTokens',
        'maxPromptTokens must be non-negative.',
      );
    }
    if (maxCompletionTokens != null && maxCompletionTokens! < 0) {
      throw ArgumentError.value(
        maxCompletionTokens,
        'maxCompletionTokens',
        'maxCompletionTokens must be non-negative.',
      );
    }
    if (maxTotalTokens != null && maxTotalTokens! < 0) {
      throw ArgumentError.value(
        maxTotalTokens,
        'maxTotalTokens',
        'maxTotalTokens must be non-negative.',
      );
    }
    if (maxCostUsd != null && (maxCostUsd!.isNaN || maxCostUsd! < 0.0)) {
      throw ArgumentError.value(
        maxCostUsd,
        'maxCostUsd',
        'maxCostUsd must be a non-negative finite number.',
      );
    }
  }

  /// Serializes configured budget thresholds to JSON for CI reporting.
  Map<String, Object?> toJson() => <String, Object?>{
    if (maxLatency != null) 'maxLatencyMs': maxLatency!.inMilliseconds,
    if (maxPromptTokens != null) 'maxPromptTokens': maxPromptTokens,
    if (maxCompletionTokens != null) 'maxCompletionTokens': maxCompletionTokens,
    if (maxTotalTokens != null) 'maxTotalTokens': maxTotalTokens,
    if (maxCostUsd != null) 'maxCostUsd': maxCostUsd,
  };

  @override
  String toString() => 'PromptBudget(${toJson()})';
}

/// Immutable record of a single tracked AI invocation inside [AiTestMetrics].
@immutable
class AiMetricSample {
  /// Creates an [AiMetricSample].
  const AiMetricSample({
    required this.latency,
    this.promptTokens = 0,
    this.completionTokens = 0,
    this.costUsd = 0.0,
    this.label,
  });

  /// Observed wall-clock latency for this sample.
  final Duration latency;

  /// Prompt tokens consumed in this sample.
  final int promptTokens;

  /// Completion tokens generated in this sample.
  final int completionTokens;

  /// Cost in USD incurred by this sample.
  final double costUsd;

  /// Optional label (for example, step name or prompt variant).
  final String? label;

  /// Combined prompt and completion tokens for this sample.
  int get totalTokens => promptTokens + completionTokens;
}

/// Pairs an async operation's return [result] with its recorded [metrics].
@immutable
class AiTrackedExecution<T> {
  /// Creates an [AiTrackedExecution].
  const AiTrackedExecution({required this.result, required this.metrics});

  /// Value returned by the measured AI operation.
  final T result;

  /// Real-time performance and token metrics captured during execution.
  final AiTestMetrics metrics;
}

/// Tracks latency, token consumption, and USD cost for AI model calls in tests.
///
/// Can be constructed directly from known values or used as a real-time
/// performance tracker via [start], [stop], [recordUsage], and [measure].
class AiTestMetrics {
  /// Creates an [AiTestMetrics] instance with optional initial readings.
  ///
  /// If [costUsd] is omitted and [pricing] is supplied, initial USD cost is
  /// derived from [promptTokens] and [completionTokens].
  AiTestMetrics({
    Duration latency = Duration.zero,
    int promptTokens = 0,
    int completionTokens = 0,
    double? costUsd,
    AiTokenPricing? pricing,
    this.modelName,
    Map<String, Object?> metadata = const <String, Object?>{},
  }) : _accumulatedLatency = latency,
       _promptTokens = promptTokens,
       _completionTokens = completionTokens,
       _costUsd =
           costUsd ??
           (pricing?.calculateCostUsd(
                 promptTokens: promptTokens,
                 completionTokens: completionTokens,
               ) ??
               0.0),
       metadata = Map<String, Object?>.of(metadata) {
    _validateMetrics(
      latency: latency,
      promptTokens: promptTokens,
      completionTokens: completionTokens,
      costUsd: _costUsd,
    );
    if (latency > Duration.zero ||
        promptTokens > 0 ||
        completionTokens > 0 ||
        _costUsd > 0.0) {
      _samples.add(
        AiMetricSample(
          latency: latency,
          promptTokens: promptTokens,
          completionTokens: completionTokens,
          costUsd: _costUsd,
          label: modelName,
        ),
      );
    }
  }

  final Stopwatch _stopwatch = Stopwatch();
  Duration _accumulatedLatency;
  int _promptTokens;
  int _completionTokens;
  double _costUsd;
  final List<AiMetricSample> _samples = <AiMetricSample>[];

  /// Optional model identifier (for example, `gemini-2.5-flash`).
  final String? modelName;

  /// Arbitrary metadata attached to this metrics run.
  final Map<String, Object?> metadata;

  /// Executes [operation], measures wall-clock latency in real time, and
  /// populates an [AiTestMetrics] instance.
  ///
  /// Provide [extractor] to pull token counts and/or USD cost from the returned
  /// [T] object once [operation] completes.
  static Future<AiTrackedExecution<T>> measure<T>(
    Future<T> Function() operation, {
    AiMetricSample Function(T result, Duration elapsed)? extractor,
    AiTokenPricing? pricing,
    String? modelName,
    Map<String, Object?> metadata = const <String, Object?>{},
  }) async {
    final stopwatch = Stopwatch()..start();
    try {
      final T result = await operation();
      stopwatch.stop();
      final Duration elapsed = stopwatch.elapsed;

      if (extractor != null) {
        final AiMetricSample sample = extractor(result, elapsed);
        final double resolvedCost = sample.costUsd > 0.0
            ? sample.costUsd
            : (pricing?.calculateCostUsd(
                    promptTokens: sample.promptTokens,
                    completionTokens: sample.completionTokens,
                  ) ??
                  0.0);
        final metrics = AiTestMetrics(
          latency: sample.latency > Duration.zero ? sample.latency : elapsed,
          promptTokens: sample.promptTokens,
          completionTokens: sample.completionTokens,
          costUsd: resolvedCost,
          modelName: modelName ?? sample.label,
          metadata: metadata,
        );
        return AiTrackedExecution<T>(result: result, metrics: metrics);
      }

      return AiTrackedExecution<T>(
        result: result,
        metrics: AiTestMetrics(
          latency: elapsed,
          modelName: modelName,
          metadata: metadata,
        ),
      );
    } finally {
      if (stopwatch.isRunning) {
        stopwatch.stop();
      }
    }
  }

  /// Whether the real-time latency timer is currently active.
  bool get isRunning => _stopwatch.isRunning;

  /// Total observed latency, including any currently running [start] span.
  Duration get latency =>
      _accumulatedLatency +
      (_stopwatch.isRunning ? _stopwatch.elapsed : Duration.zero);

  /// Total prompt (input) tokens recorded across all runs.
  int get promptTokens => _promptTokens;

  /// Total completion (output) tokens recorded across all runs.
  int get completionTokens => _completionTokens;

  /// Combined prompt and completion tokens (`promptTokens + completionTokens`).
  int get totalTokens => _promptTokens + _completionTokens;

  /// Total accumulated cost in US dollars.
  double get costUsd => _costUsd;

  /// Unmodifiable list of individual samples recorded into this tracker.
  List<AiMetricSample> get samples =>
      UnmodifiableListView<AiMetricSample>(_samples);

  /// Starts real-time latency measurement.
  ///
  /// Throws [StateError] if [start] has already been called without a matching
  /// [stop].
  void start() {
    if (_stopwatch.isRunning) {
      throw StateError('AiTestMetrics timer is already running.');
    }
    _stopwatch
      ..reset()
      ..start();
  }

  /// Stops the active real-time latency timer, logs the elapsed span plus
  /// optional token/cost usage, and returns the measured lap [Duration].
  ///
  /// Throws [StateError] if [start] was not called.
  Duration stop({
    int promptTokens = 0,
    int completionTokens = 0,
    double? costUsd,
    AiTokenPricing? pricing,
    String? label,
  }) {
    if (!_stopwatch.isRunning) {
      throw StateError(
        'AiTestMetrics timer is not running. Call start() first.',
      );
    }
    _stopwatch.stop();
    final Duration lap = _stopwatch.elapsed;
    _stopwatch.reset();

    recordUsage(
      latency: lap,
      promptTokens: promptTokens,
      completionTokens: completionTokens,
      costUsd: costUsd,
      pricing: pricing,
      label: label,
    );
    return lap;
  }

  /// Appends a completed AI call's metrics to the running totals.
  void recordUsage({
    Duration latency = Duration.zero,
    int promptTokens = 0,
    int completionTokens = 0,
    double? costUsd,
    AiTokenPricing? pricing,
    String? label,
  }) {
    final double resolvedCost =
        costUsd ??
        (pricing?.calculateCostUsd(
              promptTokens: promptTokens,
              completionTokens: completionTokens,
            ) ??
            0.0);

    _validateMetrics(
      latency: latency,
      promptTokens: promptTokens,
      completionTokens: completionTokens,
      costUsd: resolvedCost,
    );

    _accumulatedLatency += latency;
    _promptTokens += promptTokens;
    _completionTokens += completionTokens;
    _costUsd += resolvedCost;

    _samples.add(
      AiMetricSample(
        latency: latency,
        promptTokens: promptTokens,
        completionTokens: completionTokens,
        costUsd: resolvedCost,
        label: label ?? modelName,
      ),
    );
  }

  /// Clears all accumulated latency, token counts, cost, and sample history.
  void reset() {
    _stopwatch
      ..stop()
      ..reset();
    _accumulatedLatency = Duration.zero;
    _promptTokens = 0;
    _completionTokens = 0;
    _costUsd = 0.0;
    _samples.clear();
  }

  /// Serializes these metrics into a JSON-friendly summary map.
  Map<String, Object?> toJson() => <String, Object?>{
    if (modelName != null) 'modelName': modelName,
    'latencyMs': latency.inMilliseconds,
    'promptTokens': promptTokens,
    'completionTokens': completionTokens,
    'totalTokens': totalTokens,
    'costUsd': costUsd,
    'sampleCount': _samples.length,
    if (metadata.isNotEmpty) 'metadata': metadata,
  };

  @override
  String toString() =>
      'AiTestMetrics(latency: ${latency.inMilliseconds}ms, '
      'promptTokens: $promptTokens, completionTokens: $completionTokens, '
      'costUsd: ${_formatUsd(costUsd)}'
      '${modelName == null ? '' : ', model: $modelName'})';

  static void _validateMetrics({
    required Duration latency,
    required int promptTokens,
    required int completionTokens,
    required double costUsd,
  }) {
    if (latency.isNegative) {
      throw ArgumentError.value(
        latency,
        'latency',
        'Latency must not be negative.',
      );
    }
    if (promptTokens < 0) {
      throw ArgumentError.value(
        promptTokens,
        'promptTokens',
        'promptTokens must be non-negative.',
      );
    }
    if (completionTokens < 0) {
      throw ArgumentError.value(
        completionTokens,
        'completionTokens',
        'completionTokens must be non-negative.',
      );
    }
    if (costUsd.isNaN || costUsd < 0.0) {
      throw ArgumentError.value(
        costUsd,
        'costUsd',
        'costUsd must be a non-negative finite number.',
      );
    }
  }
}

/// Details a single [PromptBudget] threshold breached by [AiTestMetrics].
@immutable
class BudgetViolation {
  /// Creates a [BudgetViolation].
  const BudgetViolation({
    required this.metric,
    required this.actualValue,
    required this.budgetLimit,
    required this.overage,
  });

  /// Name of the breached metric (for example, `maxLatency` or `maxCostUsd`).
  final String metric;

  /// Formatted observed value.
  final String actualValue;

  /// Formatted configured threshold.
  final String budgetLimit;

  /// Formatted overage delta and percentage.
  final String overage;

  /// Formats this violation as a concise single-line bullet.
  String format() =>
      '  - $metric: actual $actualValue exceeded budget limit $budgetLimit ($overage)';

  @override
  String toString() => format();
}

/// Test failure thrown when [expectWithinBudget] finds one or more budget
/// thresholds exceeded.
class BudgetExceededFailure extends TestFailure {
  /// Creates a [BudgetExceededFailure].
  BudgetExceededFailure({
    required this.actualMetrics,
    required this.budget,
    required this.violations,
    this.reason,
  }) : super(
         _buildMessage(
           actualMetrics: actualMetrics,
           budget: budget,
           violations: violations,
           reason: reason,
         ),
       );

  /// Observed metrics that breached [budget].
  final AiTestMetrics actualMetrics;

  /// Budget configuration enforced by the test.
  final PromptBudget budget;

  /// Every individual threshold violation detected.
  final List<BudgetViolation> violations;

  /// Optional caller-provided failure context.
  final String? reason;

  static String _buildMessage({
    required AiTestMetrics actualMetrics,
    required PromptBudget budget,
    required List<BudgetViolation> violations,
    required String? reason,
  }) {
    final buffer = StringBuffer()
      ..writeln(
        'AI PromptBudget exceeded (${violations.length} violation(s)'
        '${actualMetrics.modelName != null ? ' for model "${actualMetrics.modelName}"' : ''}):',
      );

    if (reason != null && reason.isNotEmpty) {
      buffer.writeln('Reason: $reason');
    }

    for (final BudgetViolation violation in violations) {
      buffer.writeln(violation.format());
    }

    buffer
      ..writeln('Observed Metrics: $actualMetrics')
      ..writeln('Configured Budget: $budget');

    return buffer.toString().trimRight();
  }
}

/// Evaluates [actualMetrics] against [budget] and returns any
/// [BudgetViolation]s without throwing.
List<BudgetViolation> evaluateBudgetViolations(
  AiTestMetrics actualMetrics,
  PromptBudget budget,
) {
  budget.validate();

  final violations = <BudgetViolation>[];
  final Duration actualLatency = actualMetrics.latency;

  if (budget.maxLatency != null && actualLatency > budget.maxLatency!) {
    final Duration delta = actualLatency - budget.maxLatency!;
    violations.add(
      BudgetViolation(
        metric: 'maxLatency',
        actualValue: _formatDuration(actualLatency),
        budgetLimit: _formatDuration(budget.maxLatency!),
        overage: _formatOverage(
          delta: delta.inMicroseconds.toDouble(),
          limit: budget.maxLatency!.inMicroseconds.toDouble(),
          formattedDelta: '+${_formatDuration(delta)}',
        ),
      ),
    );
  }

  if (budget.maxPromptTokens != null &&
      actualMetrics.promptTokens > budget.maxPromptTokens!) {
    final int delta = actualMetrics.promptTokens - budget.maxPromptTokens!;
    violations.add(
      BudgetViolation(
        metric: 'maxPromptTokens',
        actualValue: '${actualMetrics.promptTokens} tokens',
        budgetLimit: '${budget.maxPromptTokens} tokens',
        overage: _formatOverage(
          delta: delta.toDouble(),
          limit: budget.maxPromptTokens!.toDouble(),
          formattedDelta: '+$delta tokens',
        ),
      ),
    );
  }

  if (budget.maxCompletionTokens != null &&
      actualMetrics.completionTokens > budget.maxCompletionTokens!) {
    final int delta =
        actualMetrics.completionTokens - budget.maxCompletionTokens!;
    violations.add(
      BudgetViolation(
        metric: 'maxCompletionTokens',
        actualValue: '${actualMetrics.completionTokens} tokens',
        budgetLimit: '${budget.maxCompletionTokens} tokens',
        overage: _formatOverage(
          delta: delta.toDouble(),
          limit: budget.maxCompletionTokens!.toDouble(),
          formattedDelta: '+$delta tokens',
        ),
      ),
    );
  }

  if (budget.maxTotalTokens != null &&
      actualMetrics.totalTokens > budget.maxTotalTokens!) {
    final int delta = actualMetrics.totalTokens - budget.maxTotalTokens!;
    violations.add(
      BudgetViolation(
        metric: 'maxTotalTokens',
        actualValue: '${actualMetrics.totalTokens} tokens',
        budgetLimit: '${budget.maxTotalTokens} tokens',
        overage: _formatOverage(
          delta: delta.toDouble(),
          limit: budget.maxTotalTokens!.toDouble(),
          formattedDelta: '+$delta tokens',
        ),
      ),
    );
  }

  // Tolerance avoids spurious failures from IEEE-754 summation rounding.
  const double costTolerance = 1e-12;
  if (budget.maxCostUsd != null &&
      (actualMetrics.costUsd - budget.maxCostUsd!) > costTolerance) {
    final double delta = actualMetrics.costUsd - budget.maxCostUsd!;
    violations.add(
      BudgetViolation(
        metric: 'maxCostUsd',
        actualValue: _formatUsd(actualMetrics.costUsd),
        budgetLimit: _formatUsd(budget.maxCostUsd!),
        overage: _formatOverage(
          delta: delta,
          limit: budget.maxCostUsd!,
          formattedDelta: '+${_formatUsd(delta)}',
        ),
      ),
    );
  }

  return violations;
}

/// Asserts that [actualMetrics] stay within every threshold configured on
/// [budget].
///
/// Compares latency, prompt tokens, completion tokens, optional total tokens,
/// and USD cost. When any threshold is breached, throws a
/// [BudgetExceededFailure] (a [TestFailure]) listing every violated metric,
/// the observed value, the configured limit, and the exact overage.
void expectWithinBudget(
  AiTestMetrics actualMetrics,
  PromptBudget budget, {
  String? reason,
}) {
  final List<BudgetViolation> violations = evaluateBudgetViolations(
    actualMetrics,
    budget,
  );
  if (violations.isNotEmpty) {
    throw BudgetExceededFailure(
      actualMetrics: actualMetrics,
      budget: budget,
      violations: violations,
      reason: reason,
    );
  }
}

/// Returns a [Matcher] verifying that an [AiTestMetrics] instance satisfies
/// [budget].
Matcher isWithinBudget(PromptBudget budget) => _BudgetMatcher(budget);

class _BudgetMatcher extends Matcher {
  _BudgetMatcher(this.budget) {
    budget.validate();
  }

  final PromptBudget budget;

  @override
  bool matches(Object? item, Map<dynamic, dynamic> matchState) {
    if (item is! AiTestMetrics) {
      matchState['error'] = 'Expected AiTestMetrics, got ${item.runtimeType}.';
      return false;
    }

    final List<BudgetViolation> violations = evaluateBudgetViolations(
      item,
      budget,
    );
    matchState['violations'] = violations;
    return violations.isEmpty;
  }

  @override
  Description describe(Description description) =>
      description.add('satisfies $budget');

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
    final violations =
        (matchState['violations'] as List<BudgetViolation>?) ??
        const <BudgetViolation>[];
    if (violations.isEmpty) {
      return mismatchDescription.add('violated budget.');
    }
    return mismatchDescription.add(
      violations.map((BudgetViolation v) => v.format()).join('\n'),
    );
  }
}

String _formatDuration(Duration duration) {
  if (duration.inMilliseconds >= 1000) {
    final double seconds = duration.inMicroseconds / 1000000.0;
    return '${duration.inMilliseconds}ms (${seconds.toStringAsFixed(2)}s)';
  }
  if (duration.inMilliseconds > 0) {
    return '${duration.inMilliseconds}ms';
  }
  return '${duration.inMicroseconds}us';
}

String _formatUsd(double amount) {
  if (amount == 0.0) {
    return r'$0.0000';
  }
  if (amount < 0.01) {
    return '\$${amount.toStringAsFixed(6)}';
  }
  return '\$${amount.toStringAsFixed(4)}';
}

String _formatOverage({
  required double delta,
  required double limit,
  required String formattedDelta,
}) {
  if (limit <= 0.0) {
    return formattedDelta;
  }
  final double pct = (delta / limit) * 100.0;
  return '$formattedDelta / +${pct.toStringAsFixed(1)}%';
}
