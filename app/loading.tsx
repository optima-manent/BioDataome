import { AtlasPrimerGraphic } from "./components/AtlasIntro";

export default function Loading() {
  return (
    <main className="atlas-route-loading" aria-live="polite" aria-busy="true">
      <div>
        <span className="panel-eyebrow">C-SKL Atlas</span>
        <AtlasPrimerGraphic compact />
        <strong>Preparing the published snapshot…</strong>
        <small>Checking datasets, relationships, and provenance.</small>
      </div>
    </main>
  );
}
