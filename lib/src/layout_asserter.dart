import 'dart:async';
import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:flutter/rendering.dart';
import 'package:flutter/widgets.dart';
import 'package:flutter_test/flutter_test.dart';

/// Builder callback that converts a raw AI-generated JSON payload into a
/// Flutter widget tree under test.
typedef GenerativeWidgetBuilder = Widget Function(Map<String, dynamic> json);

/// Optional per-payload callback invoked after a generative widget tree has
/// rendered without framework errors or overflows.
typedef GenerativeLayoutCallback =
    FutureOr<void> Function(WidgetTester tester, Map<String, dynamic> payload);

/// Diagnostic snapshot of a [RenderObject] that experienced an overflow or
/// entered an error state during layout/painting.
@immutable
class OverflowDiagnostic {
  /// Creates an [OverflowDiagnostic] describing a failing render node.
  const OverflowDiagnostic({
    required this.renderObjectType,
    required this.summary,
    this.size,
    this.constraintsDescription,
    this.creatorDescription,
  });

  /// Runtime type name of the [RenderObject] (for example, `RenderFlex`).
  final String renderObjectType;

  /// Short diagnostic summary of the render object (for example,
  /// `RenderFlex#a1b2c OVERFLOWING`).
  final String summary;

  /// Laid-out size of the render object, if available.
  final Size? size;

  /// Box constraints applied during layout, if available.
  final String? constraintsDescription;

  /// Widget ancestry / creator chain for locating the offending widget.
  final String? creatorDescription;

  /// Formats this diagnostic entry as a multi-line human-readable block.
  String format() {
    final buffer = StringBuffer()..writeln('  - $summary ($renderObjectType)');
    if (size != null) {
      buffer.writeln('    Size: ${size!.width} x ${size!.height}');
    }
    if (constraintsDescription != null) {
      buffer.writeln('    Constraints: $constraintsDescription');
    }
    if (creatorDescription != null && creatorDescription!.isNotEmpty) {
      buffer.writeln('    Widget Ancestry: $creatorDescription');
    }
    return buffer.toString().trimRight();
  }

  @override
  String toString() => format();
}

/// Captures full diagnostic context for a single generative payload test run
/// that triggered a framework exception, build crash, or layout overflow.
@immutable
class LayoutFailureEntry {
  /// Creates a [LayoutFailureEntry] for a failing payload render pass.
  const LayoutFailureEntry({
    required this.payloadIndex,
    required this.payload,
    this.viewport,
    this.frameworkException,
    this.stackTrace,
    this.overflowDiagnostics = const <OverflowDiagnostic>[],
  });

  /// Zero-based index of the failing payload in the input payload list.
  final int payloadIndex;

  /// Exact AI-generated JSON payload that triggered the failure.
  final Map<String, dynamic> payload;

  /// Logical viewport size active during this test pass, if explicit
  /// `viewports` were configured.
  final Size? viewport;

  /// Framework or builder exception caught via `try/catch` or
  /// [WidgetTester.takeException].
  final Object? frameworkException;

  /// Stack trace recorded when the builder or callback threw synchronously or
  /// asynchronously.
  final StackTrace? stackTrace;

  /// Overflow or error-box diagnostics collected from the [RenderObject] tree.
  final List<OverflowDiagnostic> overflowDiagnostics;

  /// Formats this entry, including pretty-printed JSON of [payload].
  String format() {
    final encoder = JsonEncoder.withIndent(
      '  ',
      (Object? nonEncodable) => nonEncodable.toString(),
    );
    final String prettyPayload = encoder.convert(payload);

    final buffer = StringBuffer();
    final String viewportLabel = viewport == null
        ? ''
        : ' [viewport: ${viewport!.width.toStringAsFixed(0)}x${viewport!.height.toStringAsFixed(0)}]';
    buffer.writeln('Payload #$payloadIndex$viewportLabel');

    if (frameworkException != null) {
      buffer.writeln('  Exception: $frameworkException');
    }

    if (overflowDiagnostics.isNotEmpty) {
      buffer.writeln('  Render Tree Diagnostics:');
      for (final OverflowDiagnostic diagnostic in overflowDiagnostics) {
        buffer.writeln(diagnostic.format());
      }
    }

    if (stackTrace != null) {
      final List<String> topFrames = stackTrace
          .toString()
          .split('\n')
          .where((String line) => line.trim().isNotEmpty)
          .take(5)
          .toList();
      if (topFrames.isNotEmpty) {
        buffer.writeln('  Stack Trace (top frames):');
        for (final String frame in topFrames) {
          buffer.writeln('    $frame');
        }
      }
    }

    buffer.writeln('  Failing JSON Payload:');
    for (final String line in prettyPayload.split('\n')) {
      buffer.writeln('    $line');
    }

    return buffer.toString().trimRight();
  }

  @override
  String toString() => format();
}

/// Test failure thrown when [LayoutAsserter] detects exceptions or layout
/// overflows while rendering one or more generative JSON payloads.
class LayoutAssertionFailure extends TestFailure {
  /// Creates a [LayoutAssertionFailure] summarizing all failed layout runs.
  LayoutAssertionFailure({required this.totalRuns, required this.failures})
    : super(_buildMessage(totalRuns, failures));

  /// Total number of payload/viewport render passes executed (or attempted
  /// before `failFast` halted execution).
  final int totalRuns;

  /// Structured details for every failing payload render pass.
  final List<LayoutFailureEntry> failures;

  static String _buildMessage(
    int totalRuns,
    List<LayoutFailureEntry> failures,
  ) {
    final buffer = StringBuffer()
      ..writeln(
        'LayoutAsserter detected ${failures.length} layout/rendering '
        'failure(s) across $totalRuns test run(s):',
      )
      ..writeln();

    for (var i = 0; i < failures.length; i++) {
      buffer.writeln('--- Failure ${i + 1} of ${failures.length} ---');
      buffer.writeln(failures[i].format());
      if (i < failures.length - 1) {
        buffer.writeln();
      }
    }

    return buffer.toString().trimRight();
  }
}

/// Utility for stress-testing Generative UI widgets against batches of
/// AI-generated or synthetic JSON payloads.
///
/// Use [LayoutAsserter.testGenerativeLayout] (or the top-level
/// [testGenerativeLayout] helper) inside `testWidgets` or `integration_test`
/// suites to verify that varying LLM outputs render cleanly without Flutter
/// framework exceptions or [RenderFlex] layout overflows.
class LayoutAsserter {
  /// Creates a reusable [LayoutAsserter] configuration.
  ///
  /// - [viewports]: Optional logical screen sizes to test each payload against
  ///   (for example, narrow mobile vs. tablet layouts).
  /// - [pumpDuration]: Optional duration passed to `pump` (or timeout used by
  ///   `pumpAndSettle`) after pumping each payload.
  /// - [settle]: When `true`, calls `tester.pumpAndSettle` after mounting each
  ///   payload. Leave `false` if the widget contains infinite animations.
  /// - [failFast]: When `true`, stops on the first failing payload instead of
  ///   evaluating the remaining payloads.
  /// - [onPayloadRendered]: Optional callback executed after a payload passes
  ///   framework exception and overflow checks.
  const LayoutAsserter({
    this.viewports,
    this.pumpDuration,
    this.settle = false,
    this.failFast = false,
    this.onPayloadRendered,
  });

  /// Optional list of logical viewport [Size]s tested for every payload.
  final List<Size>? viewports;

  /// Optional duration used for an extra `pump` or `pumpAndSettle` step.
  final Duration? pumpDuration;

  /// Whether to settle pending animations after pumping each widget tree.
  final bool settle;

  /// Whether to abort immediately on the first failing payload.
  final bool failFast;

  /// Optional assertion hook run after each clean payload render.
  final GenerativeLayoutCallback? onPayloadRendered;

  /// Executes this [LayoutAsserter] against [payloads] using [builder].
  Future<void> run(
    WidgetTester tester,
    GenerativeWidgetBuilder builder,
    List<Map<String, dynamic>> payloads,
  ) {
    return testGenerativeLayout(
      tester,
      builder,
      payloads,
      viewports: viewports,
      pumpDuration: pumpDuration,
      settle: settle,
      failFast: failFast,
      onPayloadRendered: onPayloadRendered,
    );
  }

  /// Pumps every JSON payload in [payloads] through [builder] and asserts that:
  /// 1. No synchronous builder errors or Flutter framework exceptions occur.
  /// 2. No [RenderFlex] or container layout overflows appear in the
  ///    [RenderObject] tree.
  ///
  /// Parameters:
  /// - [tester]: Active [WidgetTester] from `testWidgets` or
  ///   `integration_test`.
  /// - [builder]: Function building the Generative UI widget for a single
  ///   decoded JSON payload.
  /// - [payloads]: Non-empty list of synthetic or LLM-generated JSON maps to
  ///   render and validate.
  /// - [viewports]: Optional list of logical screen [Size]s to test each
  ///   payload across multiple device dimensions.
  /// - [pumpDuration]: Optional extra pump duration after initial mount.
  /// - [settle]: Whether to call `tester.pumpAndSettle` after mounting.
  /// - [failFast]: Whether to throw on the first failure rather than
  ///   collecting all failing payloads.
  /// - [onPayloadRendered]: Optional hook for running custom assertions on
  ///   each cleanly rendered payload.
  ///
  /// Throws [ArgumentError] if [payloads] is empty or [viewports] is explicitly
  /// passed as an empty list. Throws [LayoutAssertionFailure] (a [TestFailure])
  /// if any payload causes an exception or layout overflow.
  static Future<void> testGenerativeLayout(
    WidgetTester tester,
    GenerativeWidgetBuilder builder,
    List<Map<String, dynamic>> payloads, {
    List<Size>? viewports,
    Duration? pumpDuration,
    bool settle = false,
    bool failFast = false,
    GenerativeLayoutCallback? onPayloadRendered,
  }) async {
    if (payloads.isEmpty) {
      throw ArgumentError.value(
        payloads,
        'payloads',
        'At least one JSON payload must be provided to testGenerativeLayout.',
      );
    }
    if (viewports != null && viewports.isEmpty) {
      throw ArgumentError.value(
        viewports,
        'viewports',
        'If provided, viewports must contain at least one Size.',
      );
    }

    final List<Size?> effectiveViewports = viewports ?? const <Size?>[null];
    final bool hasCustomViewports = viewports != null;
    final Size originalPhysicalSize = tester.view.physicalSize;
    final double originalDevicePixelRatio = tester.view.devicePixelRatio;

    final failures = <LayoutFailureEntry>[];
    var totalRuns = 0;

    try {
      for (final Size? viewport in effectiveViewports) {
        if (viewport != null) {
          tester.view.devicePixelRatio = 1.0;
          tester.view.physicalSize = viewport;
        }

        for (var i = 0; i < payloads.length; i++) {
          totalRuns++;
          final Map<String, dynamic> payload = payloads[i];

          final LayoutFailureEntry? failure = await _evaluateSinglePayload(
            tester: tester,
            builder: builder,
            payloadIndex: i,
            payload: payload,
            viewport: viewport,
            pumpDuration: pumpDuration,
            settle: settle,
            onPayloadRendered: onPayloadRendered,
          );

          if (failure != null) {
            failures.add(failure);
            if (failFast) {
              throw LayoutAssertionFailure(
                totalRuns: totalRuns,
                failures: failures,
              );
            }
          }
        }
      }
    } finally {
      // Drain any residual exception before unwinding so the test binding
      // teardown remains clean even when a LayoutAssertionFailure is expected.
      tester.takeException();
      if (hasCustomViewports) {
        tester.view.physicalSize = originalPhysicalSize;
        tester.view.devicePixelRatio = originalDevicePixelRatio;
      }
    }

    if (failures.isNotEmpty) {
      throw LayoutAssertionFailure(totalRuns: totalRuns, failures: failures);
    }
  }

  static Future<LayoutFailureEntry?> _evaluateSinglePayload({
    required WidgetTester tester,
    required GenerativeWidgetBuilder builder,
    required int payloadIndex,
    required Map<String, dynamic> payload,
    required Size? viewport,
    required Duration? pumpDuration,
    required bool settle,
    required GenerativeLayoutCallback? onPayloadRendered,
  }) async {
    Object? caughtException;
    StackTrace? caughtStackTrace;

    // Clear any stale exception left over from previous operations.
    tester.takeException();

    Widget? builtWidget;
    try {
      builtWidget = builder(payload);
    } catch (error, stackTrace) {
      caughtException = error;
      caughtStackTrace = stackTrace;
    }

    if (builtWidget != null) {
      try {
        // Wrap with a fresh KeyedSubtree per run so RenderObjects (including
        // DebugOverflowIndicatorMixin state) and widget State do not leak
        // across payloads.
        await tester.pumpWidget(
          KeyedSubtree(key: UniqueKey(), child: builtWidget),
        );

        if (settle) {
          await tester.pumpAndSettle(
            pumpDuration ?? const Duration(milliseconds: 100),
          );
        } else if (pumpDuration != null) {
          await tester.pump(pumpDuration);
        }
      } catch (error, stackTrace) {
        caughtException ??= error;
        caughtStackTrace ??= stackTrace;
      }
    }

    final dynamic frameworkException = tester.takeException();
    if (frameworkException != null) {
      caughtException ??= frameworkException;
    }

    final List<OverflowDiagnostic> overflowDiagnostics = builtWidget != null
        ? _collectOverflowDiagnostics(tester)
        : const <OverflowDiagnostic>[];

    if (caughtException != null || overflowDiagnostics.isNotEmpty) {
      return LayoutFailureEntry(
        payloadIndex: payloadIndex,
        payload: payload,
        viewport: viewport,
        frameworkException: caughtException,
        stackTrace: caughtStackTrace,
        overflowDiagnostics: overflowDiagnostics,
      );
    }

    if (onPayloadRendered != null) {
      try {
        await onPayloadRendered(tester, payload);
      } catch (error, stackTrace) {
        return LayoutFailureEntry(
          payloadIndex: payloadIndex,
          payload: payload,
          viewport: viewport,
          frameworkException: error,
          stackTrace: stackTrace,
        );
      } finally {
        final dynamic postCallbackException = tester.takeException();
        if (postCallbackException != null && caughtException == null) {
          // Ensure no lingering exception survives past callback execution.
          caughtException = postCallbackException;
        }
      }
      if (caughtException != null) {
        return LayoutFailureEntry(
          payloadIndex: payloadIndex,
          payload: payload,
          viewport: viewport,
          frameworkException: caughtException,
        );
      }
    }

    return null;
  }

  static List<OverflowDiagnostic> _collectOverflowDiagnostics(
    WidgetTester tester,
  ) {
    final diagnostics = <OverflowDiagnostic>[];
    final visited = <RenderObject>{};

    void visit(RenderObject node) {
      if (!visited.add(node)) {
        return;
      }

      final String shortDescription = node.toStringShort();
      final bool isOverflowing = shortDescription.contains('OVERFLOWING');
      final bool isErrorBox = node is RenderErrorBox;

      if (isOverflowing || isErrorBox) {
        Size? nodeSize;
        String? constraintsDescription;
        if (node is RenderBox && node.hasSize) {
          nodeSize = node.size;
          try {
            if (!node.debugNeedsLayout) {
              constraintsDescription = node.constraints.toString();
            }
          } catch (_) {
            // Constraints may be unavailable if layout aborted midway.
          }
        }

        final String? creatorDescription =
            kDebugMode && node.debugCreator != null
            ? node.debugCreator.toString()
            : null;

        diagnostics.add(
          OverflowDiagnostic(
            renderObjectType: node.runtimeType.toString(),
            summary: shortDescription,
            size: nodeSize,
            constraintsDescription: constraintsDescription,
            creatorDescription: creatorDescription,
          ),
        );
      }

      node.visitChildren(visit);
    }

    for (final RenderView renderView in tester.binding.renderViews) {
      visit(renderView);
    }

    return diagnostics;
  }
}

/// Convenience top-level wrapper for [LayoutAsserter.testGenerativeLayout].
///
/// Iterates through [payloads], renders the widget returned by [builder] via
/// [tester], and fails with a detailed [LayoutAssertionFailure] if any payload
/// throws a framework exception or produces a layout overflow.
Future<void> testGenerativeLayout(
  WidgetTester tester,
  GenerativeWidgetBuilder builder,
  List<Map<String, dynamic>> payloads, {
  List<Size>? viewports,
  Duration? pumpDuration,
  bool settle = false,
  bool failFast = false,
  GenerativeLayoutCallback? onPayloadRendered,
}) {
  return LayoutAsserter.testGenerativeLayout(
    tester,
    builder,
    payloads,
    viewports: viewports,
    pumpDuration: pumpDuration,
    settle: settle,
    failFast: failFast,
    onPayloadRendered: onPayloadRendered,
  );
}
