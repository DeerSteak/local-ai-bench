import { describe, expect, it, vi } from "vitest";
import { fetchSelectedResultFiles } from "./autoload";

describe("fetchSelectedResultFiles", () => {
  it("does nothing without the explicit autoload query", async () => {
    const fetchFn = vi.fn();
    expect(await fetchSelectedResultFiles("", fetchFn)).toEqual([]);
    expect(fetchFn).not.toHaveBeenCalled();
  });

  it("loads every staged result in manifest order", async () => {
    const responses = [
      { ok: true, json: async () => ({ files: [
        { name: "one.json", url: "/__selected_results__/0.json" },
        { name: "two.json", url: "/__selected_results__/1.json" },
      ] }) },
      { ok: true, text: async () => "{\"one\":1}" },
      { ok: true, text: async () => "{\"two\":2}" },
    ];
    const files = await fetchSelectedResultFiles("?autoload=1", vi.fn(async () => responses.shift()));
    expect(files.map(file => file.name)).toEqual(["one.json", "two.json"]);
    expect(await files[1].text()).toBe('{"two":2}');
  });

  it("rejects manifest URLs outside the selected-results route", async () => {
    const fetchFn = vi.fn(async () => ({
      ok: true, json: async () => ({ files: [{ name: "bad.json", url: "/private/file" }] }),
    }));
    await expect(fetchSelectedResultFiles("?autoload=1", fetchFn))
      .rejects.toThrow("manifest is invalid");
  });

  it("rejects a manifest larger than the dashboard file limit", async () => {
    const fetchFn = vi.fn(async () => ({
      ok: true, json: async () => ({ files: Array(7).fill({
        name: "result.json", url: "/__selected_results__/0.json",
      }) }),
    }));
    await expect(fetchSelectedResultFiles("?autoload=1", fetchFn))
      .rejects.toThrow("manifest is invalid");
  });
});
