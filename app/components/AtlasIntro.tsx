export function AtlasPrimerGraphic({ compact = false }: { compact?: boolean }) {
  return (
    <div className={`atlas-primer-graphic ${compact ? "compact" : ""}`} aria-hidden="true">
      <span className="primer-cluster primer-cluster-a">Blood & immune · 18</span>
      <span className="primer-cluster primer-cluster-b">Respiratory · 12</span>
      <i className="primer-edge edge-a" />
      <i className="primer-edge edge-b" />
      <i className="primer-edge edge-c dotted" />
      <b className="primer-node node-a circle" />
      <b className="primer-node node-b diamond" />
      <b className="primer-node node-c hexagon" />
      <b className="primer-node node-d square" />
      <span className="primer-pulse" />
    </div>
  );
}

export function AtlasIntro({
  open,
  onDismiss,
}: {
  open: boolean;
  onDismiss: () => void;
}) {
  if (!open) return null;
  return (
    <section
      className="atlas-intro ready"
      role="dialog"
      aria-modal="true"
      aria-labelledby="atlas-intro-title"
      aria-describedby="atlas-intro-description"
    >
      <div className="atlas-intro-card">
        <div className="intro-visual-wrap">
          <span className="intro-kicker">How to read the atlas</span>
          <AtlasPrimerGraphic />
        </div>
        <div className="intro-copy">
          <span className="panel-eyebrow">A quick orientation</span>
          <h1 id="atlas-intro-title">One map, four visual cues.</h1>
          <p id="atlas-intro-description">
            Each node is a GEO dataset. The map helps you find relationships worth inspecting;
            it does not claim mechanism or causality.
          </p>
          <div className="intro-cues">
            <div><i className="cue-node color" /><span><strong>Color</strong><small>The selected context—anatomy by default, or a review-aware clinical family.</small></span></div>
            <div><i className="cue-node shape" /><span><strong>Shape</strong><small>A broad anatomical context that stays readable at map scale.</small></span></div>
            <div><i className="cue-line" /><span><strong>Line</strong><small>A published relationship; use the q-value slider to keep stronger support.</small></span></div>
            <div><i className="cue-cluster" /><span><strong>Cluster label</strong><small>Open it to focus the map, then return to all datasets in one click.</small></span></div>
          </div>
          <div className="intro-action-row">
            <span role="status">Published snapshot ready</span>
            <button type="button" autoFocus onClick={onDismiss}>
              Open the atlas <b aria-hidden="true">→</b>
            </button>
          </div>
          <small className="intro-taxonomy-note">
            Clinical families are broad, versioned browsing groups. They do not assign diagnosis
            codes, and unreviewed labels stay visibly unreviewed.
          </small>
        </div>
      </div>
    </section>
  );
}
