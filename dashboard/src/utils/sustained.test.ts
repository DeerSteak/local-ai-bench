import { describe, expect, it } from "vitest";
import { buildSustainedTimeline, flattenSustainedData, preferredTemperatureKey, sustainedSeries } from "./sustained";

describe("sustained utilities", () => {
  it("builds an aligned chart timeline while preserving missing channels", () => {
    const data = buildSustainedTimeline({ series: [
      { timestamp_sec: 0, tokens_per_sec: 42, gpu_die_c: 61, power_watts: 92 },
      { timestamp_sec: 30, tokens_per_sec: 39, gpu_die_c: null, power_watts: null },
    ] });
    expect(data).toEqual([
      { elapsed_min: 0, tokens_per_sec: 42, power_watts: 92, soc_package_c: undefined, cpu_package_c: undefined, gpu_die_c: 61, gpu_hotspot_c: undefined },
      { elapsed_min: 0.5, tokens_per_sec: 39, power_watts: null, soc_package_c: undefined, cpu_package_c: undefined, gpu_die_c: null, gpu_hotspot_c: undefined },
    ]);
    expect(preferredTemperatureKey(data)).toBe("gpu_die_c");
  });

  it("prefers a unified SoC package temperature when present", () => {
    const data = buildSustainedTimeline({
      series: [{ timestamp_sec: 0, soc_package_c: 43, cpu_package_c: 50 }],
    });
    expect(preferredTemperatureKey(data)).toBe("soc_package_c");
  });

  it("tolerates absent and malformed series", () => {
    expect(sustainedSeries(undefined)).toEqual([]);
    expect(sustainedSeries({ series: "old schema" })).toEqual([]);
    expect(preferredTemperatureKey([])).toBeNull();
  });

  it("flattens analysis and converts retention to percent", () => {
    const rows = flattenSustainedData([{ id: 7, data: { sustained: { model_a: {
      actual_duration_sec: 601, ambient_temp_c: 20, request_count: 42,
      valid_request_count: 41,
      analysis: { initial_tokens_per_sec: 50, steady_state_tokens_per_sec: 45,
        retention_ratio: 0.9, performance: "mild_degradation", cause: "temperature_correlated" },
    } } } }]);
    expect(rows[0]).toMatchObject({ _fileId: 7, model: "model_a", retention_pct: 90,
      request_count: 42, valid_request_count: 41,
      performance: "mild_degradation", cause: "temperature_correlated" });
  });

  it("preserves skipped and crashed outcomes for table rendering", () => {
    const rows = flattenSustainedData([{ id: 9, data: { sustained: {
      skipped_model: { skipped: "context_unsupported" },
      crashed_model: { crashed: "during warmup" },
    } } }]);

    expect(rows[0]).toMatchObject({ skipped: "context_unsupported", skip_detail: "context_unsupported" });
    expect(rows[1]).toMatchObject({ error: "during warmup" });
  });
});
