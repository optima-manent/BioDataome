from cskl_atlas import static_export
from cskl_atlas.display_facets import (
    DISEASE_FAMILY_VERSION,
    TISSUE_SYSTEM_VERSION,
    derive_disease_family,
    derive_tissue_system,
)


def test_tissue_system_groups_synonymous_blood_sources() -> None:
    assert derive_tissue_system(["whole blood"]) == "Blood & immune"
    assert derive_tissue_system(["bone marrow / peripheral blood"]) == "Blood & immune"
    assert derive_tissue_system(["peripheral blood mononuclear cell"]) == "Blood & immune"


def test_tissue_system_preserves_mixed_and_in_vitro_distinctions() -> None:
    assert derive_tissue_system(["lung / liver"]) == "Mixed anatomical systems"
    assert derive_tissue_system(["lymphoblastoid cell line"]) == "Cell culture / in vitro"
    assert derive_tissue_system(["UBERON:0000178 / UBERON:0002371"]) == "Mixed / unspecified"


def test_tissue_system_version_is_explicit() -> None:
    assert TISSUE_SYSTEM_VERSION.startswith("atlas-tissue-system-")


def test_disease_family_uses_broad_icd_inspired_display_groups() -> None:
    assert derive_disease_family(["breast carcinoma"]) == "Neoplastic"
    assert derive_disease_family(["chronic obstructive pulmonary disease"]) == "Respiratory"
    assert derive_disease_family(["Parkinson's disease"]) == "Neurological & mental health"
    assert derive_disease_family(["atopic dermatitis"]) == "Immune & inflammatory"


def test_disease_family_preserves_uncertainty_and_mixed_groups() -> None:
    assert derive_disease_family([]) == "Unreviewed"
    assert derive_disease_family(["rare condition not otherwise classified"]) == "Other / unclassified"
    assert derive_disease_family(["asthma; obesity"]) == "Mixed disease families"
    assert derive_disease_family(["asthma; control"]) == "Respiratory"


def test_disease_family_version_is_explicit() -> None:
    assert DISEASE_FAMILY_VERSION.startswith("atlas-clinical-family-")


def test_static_export_identity_tracks_taxonomy_and_independent_calibration(monkeypatch) -> None:
    arguments = {
        "snapshot_manifest_checksum": "a" * 64,
        "independent_calibration_id": "calibration-independent-v1",
        "metadata_hashes": ["b" * 64],
        "explanation_hashes": ["c" * 64],
        "annotation_hash": "d" * 64,
        "ontology_audit_hash": "e" * 64,
    }
    baseline = static_export._static_dependency_hash(**arguments)
    assert static_export._display_facet_versions() == {
        "display_facet_version": TISSUE_SYSTEM_VERSION,
        "disease_family_version": DISEASE_FAMILY_VERSION,
    }

    monkeypatch.setattr(
        static_export,
        "DISEASE_FAMILY_VERSION",
        "atlas-clinical-family-test-next",
    )
    assert static_export._static_dependency_hash(**arguments) != baseline

    monkeypatch.setattr(static_export, "DISEASE_FAMILY_VERSION", DISEASE_FAMILY_VERSION)
    assert static_export._static_dependency_hash(
        **{**arguments, "independent_calibration_id": "calibration-independent-v2"}
    ) != baseline


def test_static_snapshot_provenance_freezes_publication_status() -> None:
    snapshot = {
        "snapshot_id": "snapshot-v1",
        "calibration_id": "calibration-v1",
        "independent_calibration_id": "calibration-independent-v1",
        "stratum": "GPL570:global",
        "policy_hash": "policy-v1",
        "layout_version": "layout-v1",
        "manifest_checksum": "a" * 64,
        "text_release_id": None,
        "created_at": "2026-01-01T00:00:00+00:00",
        "published_at": "2026-01-02T00:00:00+00:00",
        "status": "published",
    }
    published = static_export._static_snapshot_provenance(snapshot)
    superseded = static_export._static_snapshot_provenance(
        {**snapshot, "status": "superseded"}
    )

    assert published == superseded
    assert published["status"] == "published"
