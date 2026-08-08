#!/usr/bin/env node
// Blocks a commit that adds `any` to a new dashboard/src TS file, or increases
// an existing file's `any` count — see AGENTS.md's TypeScript section and
// docs/release-policy.md#any-ratchet-hook. Run from .githooks/pre-commit.

import { execFileSync } from "node:child_process";

function git(args) {
  try {
    return execFileSync("git", args, { encoding: "utf8" });
  } catch {
    return null;
  }
}

function countAny(text) {
  const stripped = text
    .replace(/\/\*[\s\S]*?\*\//g, "")
    .replace(/^\s*\/\/.*$/gm, "");
  return (stripped.match(/\bany\b/g) || []).length;
}

const staged = (git(["diff", "--cached", "--name-only", "--diff-filter=ACM"]) || "")
  .split("\n")
  .filter(f => /^dashboard\/src\/.*\.tsx?$/.test(f) && !f.endsWith(".d.ts"));

let failed = false;

for (const file of staged) {
  const newContent = git(["show", `:${file}`]);
  if (newContent === null) continue;
  const newCount = countAny(newContent);

  const oldContent = git(["show", `HEAD:${file}`]);
  if (oldContent === null) {
    if (newCount > 0) {
      console.error(`dashboard any-ratchet: ${file} is new and contains ${newCount} \`any\` — `
        + "new files must not use `any`; type it fully.");
      failed = true;
    }
    continue;
  }

  const oldCount = countAny(oldContent);
  if (newCount > oldCount) {
    console.error(`dashboard any-ratchet: ${file} \`any\` count increased ${oldCount} -> ${newCount} `
      + "— no new `any` in a file that already exists.");
    failed = true;
  }
}

if (failed) {
  console.error("Fix the file(s) above, or for a confirmed false positive, commit with --no-verify.");
  process.exit(1);
}
