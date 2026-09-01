import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { AtlasExplorer } from "../../app/components/AtlasExplorer";
import { adaptPublishedGraph } from "../../app/lib/api-graph";
import "../../app/globals.css";

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
const root = createRoot(container);

function PrimerGraphic() {
  return (
    <div className="showcase-primer" aria-hidden="true">
      <span className="showcase-cluster-label showcase-cluster-a">Blood studies</span>
      <span className="showcase-cluster-label showcase-cluster-b">Neural studies</span>
      <svg className="showcase-links" viewBox="0 0 100 100" preserveAspectRatio="none">
        <path className="showcase-link showcase-link-a" d="M23 51 C27 58 31 66 35 71" />
        <path className="showcase-link showcase-link-b" d="M79 45 C83 52 86 61 88 69" />
        <path className="showcase-link showcase-link-c" d="M35 71 C49 59 64 48 79 45" />
      </svg>
      <span className="showcase-dot showcase-dot-a" />
      <span className="showcase-dot showcase-dot-b" />
      <span className="showcase-dot showcase-dot-c" />
      <span className="showcase-dot showcase-dot-d" />
    </div>
  );
}

function LoadingRelease() {
  return (
    <main className="showcase-loading" aria-live="polite" aria-busy="true">
      <PrimerGraphic />
      <strong>C-SKL Atlas</strong>
      <span>Checking the published 500-study map…</span>
      <small>Color shows a grouping. Shape shows anatomical context. Lines are evidence links.</small>
    </main>
  );
}

function ReleaseError({ message, onRetry }: { message: string; onRetry: () => void }) {
  return (
    <main className="showcase-loading showcase-error" role="alert">
      <PrimerGraphic />
      <strong>Release unavailable</strong>
      <span>{message}</span>
      <button type="button" onClick={onRetry}>
        Try again
      </button>
    </main>
  );
}

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

async function renderRelease() {
  root.render(<LoadingRelease />);
  try {
    const graph = await loadRelease();
    root.render(
      <StrictMode>
        <AtlasExplorer graph={graph} synthesisEndpoint={null} />
      </StrictMode>,
    );
  } catch (error) {
    const message =
      error instanceof Error ? error.message : "The Atlas release could not be loaded.";
    root.render(<ReleaseError message={message} onRetry={() => void renderRelease()} />);
  }
}

void renderRelease();
