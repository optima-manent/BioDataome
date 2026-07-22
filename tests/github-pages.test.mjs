import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readdir, readFile } from "node:fs/promises";
import test from "node:test";

const output = new URL("../dist/pages/", import.meta.url);

test("GitHub Pages artifact is static, self-contained, and release-bound", async () => {
  const [html, manifestText, graphBytes, noJekyll] = await Promise.all([
    readFile(new URL("index.html", output), "utf8"),
    readFile(new URL("release-manifest.json", output), "utf8"),
    readFile(new URL("atlas-graph.json", output)),
    readFile(new URL(".nojekyll", output), "utf8"),
  ]);
  const manifest = JSON.parse(manifestText);

  assert.equal(noJekyll, "");
  assert.equal(manifest.schema, "cskl-atlas-static-graph-v3");
  assert.equal(manifest.node_count, 500);
  assert.equal(manifest.edge_count, 5344);
  assert.equal(createHash("sha256").update(graphBytes).digest("hex"), manifest.output_checksum);
  assert.match(html, /C-SKL Atlas · Biological Dataset Discovery/);
  assert.match(html, /Content-Security-Policy/);
  assert.match(html, /\.\/assets\//);
  assert.doesNotMatch(html, /https?:\/\/[^"']+\.(?:js|css)/i);
});

test("Pages bundle contains the measured release and no private build markers", async () => {
  const manifest = JSON.parse(
    await readFile(new URL("../app/data/atlas-graph.manifest.json", import.meta.url), "utf8"),
  );
  const graphBytes = await readFile(new URL("../app/data/atlas-graph.json", import.meta.url));
  const checksum = createHash("sha256").update(graphBytes).digest("hex");
  assert.equal(checksum, manifest.output_checksum);

  const assets = await readdir(new URL("assets/", output));
  const scripts = assets.filter((name) => name.endsWith(".js"));
  const styles = assets.filter((name) => name.endsWith(".css"));
  assert.ok(scripts.length > 0);
  assert.ok(styles.length > 0);

  const scriptText = await Promise.all(
    scripts.map((name) => readFile(new URL(`assets/${name}`, output), "utf8")),
  );
  assert.ok(scriptText.reduce((total, value) => total + Buffer.byteLength(value), 0) < 750_000);

  const publicText = [
    await readFile(new URL("index.html", output), "utf8"),
    ...scriptText,
    await readFile(new URL("release-manifest.json", output), "utf8"),
  ].join("\n");
  assert.match(publicText, /snapshot_ee201ff2e0991ea8fdb7bcad/);
  assert.match(publicText, /Server synthesis unavailable/);
  const privateMarkers = new RegExp(
    ["chat" + "gpt", "co" + "dex", "C:" + "\\\\Users\\\\", "Py" + "charmProjects", "sk-or-" + "v1"].join("|"),
    "i",
  );
  assert.doesNotMatch(publicText, privateMarkers);
});
