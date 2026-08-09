import { describe, it, expect } from "vitest";
import {
  vllmBenchSizeLabel, vllmBenchLatencyEntries, vllmBenchThroughputEntries,
  buildVllmBenchLatencyData, buildVllmBenchThroughputData,
  buildVllmBenchLatencyConfigs, buildVllmBenchThroughputConfigs,
  getVllmBenchModels, flattenVllmBenchData,
} from "./vllmbench";

const file = (id: string, hostname: string, modelData: object) => ({
  id, hostname, data: { vllmbench: { "gemma3-1b": modelData } },
});

const entries = {
  latency_entries: [
    { input_len: 512, output_len: 128, avg_latency_sec: 1.0, output_tps: 128 },
    { input_len: 512, output_len: 512, avg_latency_sec: 3.0, output_tps: 170.67 },
    { input_len: 2048, output_len: 128, avg_latency_sec: 2.0, output_tps: 64 },
  ],
  throughput_entries: [
    { input_len: 512, output_len: 128, requests_per_sec: 8, output_tps: 1024 },
    { input_len: 2048, output_len: 128, requests_per_sec: 4, output_tps: 512 },
  ],
};

describe("vllmBenchSizeLabel", () => {
  it("renders whole and fractional K sizes", () => {
    expect(vllmBenchSizeLabel(2048)).toBe("2K");
    expect(vllmBenchSizeLabel(512)).toBe("0.5K");
    expect(vllmBenchSizeLabel(undefined)).toBe("0K");
  });
});

describe("entry accessors", () => {
  it("returns an empty list rather than throwing on missing or malformed sections", () => {
    expect(vllmBenchLatencyEntries(undefined)).toEqual([]);
    expect(vllmBenchLatencyEntries({})).toEqual([]);
    expect(vllmBenchLatencyEntries({ latency_entries: "nope" })).toEqual([]);
    expect(vllmBenchThroughputEntries(null)).toEqual([]);
  });
});

describe("buildVllmBenchLatencyData", () => {
  it("orders rows by input length and keys each output length separately", () => {
    const rows = buildVllmBenchLatencyData([file("f0", "host", entries)], "gemma3-1b");
    expect(rows.map(r => r.promptLabel)).toEqual(["0.5K", "2K"]);
    expect(rows[0].f0_out128).toBe(1.0);
    expect(rows[0].f0_out512).toBe(3.0);
    expect(rows[1].f0_out128).toBe(2.0);
    expect(rows[1].f0_out512).toBeUndefined();
  });

  it("returns no rows for a model with no vllmbench data", () => {
    expect(buildVllmBenchLatencyData([file("f0", "host", {})], "gemma3-1b")).toEqual([]);
    expect(buildVllmBenchLatencyData([{ id: "f0", data: {} }], "gemma3-1b")).toEqual([]);
  });

  it("merges input sizes across files that swept different ranges", () => {
    const a = file("f0", "a", { latency_entries: [{ input_len: 512, output_len: 128, avg_latency_sec: 1 }] });
    const b = file("f1", "b", { latency_entries: [{ input_len: 8192, output_len: 128, avg_latency_sec: 9 }] });
    const rows = buildVllmBenchLatencyData([a, b], "gemma3-1b");
    expect(rows.map(r => r.promptLabel)).toEqual(["0.5K", "8K"]);
    expect(rows[0].f0_out128).toBe(1);
    expect(rows[0].f1_out128).toBeUndefined();
    expect(rows[1].f1_out128).toBe(9);
  });
});

describe("buildVllmBenchThroughputData", () => {
  it("plots the derived output-only rate", () => {
    const rows = buildVllmBenchThroughputData([file("f0", "host", entries)], "gemma3-1b");
    expect(rows.map(r => r.f0_out128)).toEqual([1024, 512]);
  });
});

describe("line configs", () => {
  it("names a single file's series by output length alone", () => {
    const files = [file("f0", "host", entries)];
    const data = buildVllmBenchLatencyData(files, "gemma3-1b");
    const configs = buildVllmBenchLatencyConfigs(files, "gemma3-1b", data);
    expect(configs.map(c => c.dataKey)).toEqual(["f0_out128", "f0_out512"]);
    expect(configs.map(c => c.name)).toEqual(["out128", "out512"]);
  });

  it("qualifies series with the system name once more than one file is loaded", () => {
    const files = [file("f0", "alpha", entries), file("f1", "beta", entries)];
    const data = buildVllmBenchLatencyData(files, "gemma3-1b");
    const configs = buildVllmBenchLatencyConfigs(files, "gemma3-1b", data);
    expect(configs).toHaveLength(4);
    expect(configs.every(c => c.name.includes("·"))).toBe(true);
  });

  it("drops a series whose column is entirely empty", () => {
    const files = [file("f0", "host", { latency_entries: [{ input_len: 512, output_len: 128 }] })];
    const data = buildVllmBenchLatencyData(files, "gemma3-1b");
    expect(buildVllmBenchLatencyConfigs(files, "gemma3-1b", data)).toEqual([]);
  });

  it("builds throughput configs from throughput entries, not latency ones", () => {
    const files = [file("f0", "host", entries)];
    const data = buildVllmBenchThroughputData(files, "gemma3-1b");
    const configs = buildVllmBenchThroughputConfigs(files, "gemma3-1b", data);
    expect(configs.map(c => c.dataKey)).toEqual(["f0_out128"]);
  });
});

describe("getVllmBenchModels", () => {
  it("unions models across files without duplicates", () => {
    const files = [
      { id: "f0", data: { vllmbench: { a: {}, b: {} } } },
      { id: "f1", data: { vllmbench: { b: {}, c: {} } } },
      { id: "f2", data: {} },
    ];
    expect(getVllmBenchModels(files)).toEqual(["a", "b", "c"]);
  });
});

describe("flattenVllmBenchData", () => {
  it("joins latency and throughput on the same input/output shape", () => {
    const rows = flattenVllmBenchData([file("f0", "host", entries)]);
    const shape = rows.find(r => r.input_len === 512 && r.output_len === 128);
    expect(shape.avg_latency_sec).toBe(1.0);
    expect(shape.requests_per_sec).toBe(8);
    expect(shape.throughput_output_tps).toBe(1024);
    expect(shape.latency_output_tps).toBe(128);
  });

  it("keeps a shape measured by only one of the two benchmarks", () => {
    const rows = flattenVllmBenchData([file("f0", "host", entries)]);
    const latencyOnly = rows.find(r => r.output_len === 512);
    expect(latencyOnly.avg_latency_sec).toBe(3.0);
    expect(latencyOnly.requests_per_sec).toBeUndefined();
  });

  it("returns nothing for files without a vllmbench section", () => {
    expect(flattenVllmBenchData([{ id: "f0", data: {} }])).toEqual([]);
  });
});
