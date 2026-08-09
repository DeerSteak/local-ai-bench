// Single source of truth for the suite version is config.py's VERSION — parsed at
// build time so the dashboard never carries a third copy to keep in sync.
export function parseSuiteVersion(configSource: string | null | undefined): string | null {
  const match = /^VERSION\s*=\s*["']([^"']+)["']/m.exec(configSource || "");
  return match ? match[1] : null;
}
