import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { AtlasExplorer } from "../../app/components/AtlasExplorer";
import { adaptPublishedGraph } from "../../app/lib/api-graph";
import "../../app/globals.css";
import "./showcase.css";

type StaticManifest = {
  schema: string;
  artifact: string;
  output_checksum: string;
  snapshot_id: string;
  node_count: number;
  edge_count: number;
};

const container = document.getElementById("root");
if (!container) throw new Error("The static page is missing its application root.");

function digestHex(bytes: ArrayBuffer): Promise<string> {
  return crypto.subtle.digest("SHA-256", bytes).then((digest) =>
    [...new Uint8Array(digest)]
      .map((value) => value.toString(16).padStart(2, "0"))
      .join(""),
  );
}

async function loadRelease() {
  const [manifestResponse, graphResponse] = await Promise.all([
    fetch(new URL("./release-manifest.json", document.baseURI)),
    fetch(new URL("./atlas-graph.json", document.baseURI)),
  ]);
  if (!manifestResponse.ok || !graphResponse.ok) {
    throw new Error("The frozen Atlas release could not be loaded.");
  }

  const manifest = (await manifestResponse.json()) as StaticManifest;
  const graphBytes = await graphResponse.arrayBuffer();
  const checksum = await digestHex(graphBytes);
  if (manifest.schema !== "cskl-atlas-static-graph-v3") {
    throw new Error(`Unsupported static release schema: ${manifest.schema}`);
  }
  if (checksum !== manifest.output_checksum) {
    throw new Error("The frozen Atlas release failed its checksum validation.");
  }

  const payload = JSON.parse(new TextDecoder().decode(graphBytes));
  const graph = adaptPublishedGraph(
    payload as Parameters<typeof adaptPublishedGraph>[0],
  );
  if (
    graph.snapshotId !== manifest.snapshot_id ||
    graph.nodes.length !== manifest.node_count ||
    graph.edges.length !== manifest.edge_count
  ) {
    throw new Error("The graph payload does not match its release manifest.");
  }
  return graph;
}

try {
  const graph = await loadRelease();
  createRoot(container).render(
    <StrictMode>
      <AtlasExplorer graph={graph} synthesisEndpoint={null} />
    </StrictMode>,
  );
} catch (error) {
  const message = error instanceof Error ? error.message : "The Atlas release could not be loaded.";
  createRoot(container).render(
    <main className="showcase-loading showcase-error" role="alert">
      <strong>Release unavailable</strong>
      <span>{message}</span>
    </main>,
  );
}
