from cskl_atlas.display_facets import TISSUE_SYSTEM_VERSION, derive_tissue_system


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
