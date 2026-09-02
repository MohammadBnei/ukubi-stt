// Parses the <script type="module"> out of web/index.html and asks node to check
// its syntax. Nothing else does: the page is include_str!'d into the binary, so a
// syntax error compiles cleanly, ships, and is visible only in a browser console.
//
// It has happened. `toPCM16` was both imported and redeclared, which is a
// SyntaxError — and because a duplicate declaration kills the module at PARSE
// time, the symptom was the entire page doing nothing at all, not the audio path
// misbehaving. It survived four releases because the only check anyone ran was
// that `/` returned 200, which it did the whole time.
import { readFileSync, writeFileSync, mkdtempSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { tmpdir } from "node:os";
import { join } from "node:path";

function check(name, src) {
  const file = join(mkdtempSync(join(tmpdir(), "sttweb-")), "page.mjs");
  writeFileSync(file, src);
  try {
    execFileSync(process.execPath, ["--check", file], { stdio: "pipe" });
    console.log(`${name}: syntax OK`);
  } catch (e) {
    console.error(`${name} has a syntax error:\n` + (e.stderr || e).toString());
    process.exit(1);
  }
}

const html = readFileSync("web/index.html", "utf8");
const m = html.match(/<script type="module">([\s\S]*?)<\/script>/);
if (!m) {
  console.error('no <script type="module"> found in web/index.html');
  process.exit(1);
}
// The import is KEPT verbatim. `node --check` parses without resolving modules,
// so an unreachable specifier is fine — and keeping it is the whole point, since
// stripping it would delete the very conflict a redeclaration creates. Verified
// both ways before trusting this.
const src = m[1];
check("web/index.html module", src);
// The vendored capture module is served separately at /stt-capture.js and is
// copied into two consumer repos, so a syntax error here breaks three things.
check("web/stt-capture.js", readFileSync("web/stt-capture.js", "utf8"));
