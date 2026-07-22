import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const repositoryRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const manifestPath = resolve(repositoryRoot, "app/data/atlas-graph.manifest.json");
const manifest = JSON.parse(await readFile(manifestPath, "utf8"));

if (manifest.schema !== "cskl-atlas-static-graph-v3") {
  throw new Error(`Unsupported static release schema: ${manifest.schema}`);
}
if (typeof manifest.artifact !== "string" || !manifest.artifact) {
  throw new Error("Static release manifest does not identify its artifact.");
}

const artifactPath = resolve(dirname(manifestPath), manifest.artifact);
const bytes = await readFile(artifactPath);
const checksum = createHash("sha256").update(bytes).digest("hex");
if (checksum !== manifest.output_checksum) {
  throw new Error(`Static release checksum mismatch: expected ${manifest.output_checksum}, got ${checksum}`);
}

const graph = JSON.parse(bytes.toString("utf8"));
if (graph.snapshot?.snapshot_id !== manifest.snapshot_id) {
  throw new Error("Static graph and manifest snapshot identifiers differ.");
}
if (graph.nodes?.length !== manifest.node_count || graph.edges?.length !== manifest.edge_count) {
  throw new Error("Static graph counts do not match the release manifest.");
}
if (manifest.node_count !== 500) {
  throw new Error(`The publication showcase expects 500 nodes, found ${manifest.node_count}.`);
}

console.log(
  `Verified ${manifest.snapshot_id}: ${manifest.node_count} nodes, ${manifest.edge_count} edges, sha256 ${checksum}`,
);
