from __future__ import annotations

from cskl_atlas.pathways import build_reactome_index, enrich_reactome


def test_reactome_index_uses_explicit_array_universe_and_bh(tmp_path):
    annotation = tmp_path / "annotation.tsv"
    annotation.write_text(
        "PROBEID\tENTREZID\tSYMBOL\tGENENAME\n"
        "p1\t1\tA\tGene A\n"
        "p2\t2\tB\tGene B\n"
        "p3\t3\tC\tGene C\n"
        "p4\t4\tD\tGene D\n"
        "p5\t5\tE\tGene E\n"
        "p6\t6\tF\tGene F\n",
        encoding="utf-8",
    )
    mapping = tmp_path / "reactome.tsv"
    mapping.write_text(
        "1\tR-HSA-1\thttps://example.test/1\tPath one\tTAS\tHomo sapiens\n"
        "2\tR-HSA-1\thttps://example.test/1\tPath one\tTAS\tHomo sapiens\n"
        "3\tR-HSA-1\thttps://example.test/1\tPath one\tTAS\tHomo sapiens\n"
        "4\tR-HSA-2\thttps://example.test/2\tPath two\tTAS\tHomo sapiens\n"
        "5\tR-HSA-2\thttps://example.test/2\tPath two\tTAS\tHomo sapiens\n"
        "999\tR-HSA-3\thttps://example.test/3\tOutside universe\tTAS\tHomo sapiens\n"
        "1\tR-MMU-1\thttps://example.test/mouse\tMouse\tTAS\tMus musculus\n",
        encoding="utf-8",
    )
    database = tmp_path / "reactome.sqlite"
    manifest_path = tmp_path / "manifest.json"
    manifest = build_reactome_index(
        mapping_path=mapping,
        annotation_path=annotation,
        database_path=database,
        manifest_path=manifest_path,
        release="fixture-1",
    )
    assert manifest["background_gene_count"] == 6
    assert manifest["mapped_gene_count"] == 5
    assert manifest["pathway_count"] == 2

    result = enrich_reactome(
        ["1", "2", "unknown"],
        database_path=database,
        minimum_overlap=2,
        minimum_pathway_size=2,
    )
    assert result["background_gene_count"] == 6
    assert result["input_gene_count"] == 3
    assert result["tested_gene_count"] == 2
    assert result["tested_pathway_count"] == 2
    assert [row["pathway_id"] for row in result["results"]] == ["R-HSA-1"]
    assert result["results"][0]["gene_ids"] == ["1", "2"]
    assert 0 <= result["results"][0]["p_value"] <= result["results"][0]["q_value"] <= 1


def test_reactome_build_is_replay_safe(tmp_path):
    annotation = tmp_path / "annotation.tsv"
    annotation.write_text(
        "PROBEID\tENTREZID\tSYMBOL\tGENENAME\np1\t1\tA\tGene A\n",
        encoding="utf-8",
    )
    mapping = tmp_path / "reactome.tsv"
    mapping.write_text(
        "1\tR-HSA-1\thttps://example.test/1\tPath one\tTAS\tHomo sapiens\n",
        encoding="utf-8",
    )
    arguments = {
        "mapping_path": mapping,
        "annotation_path": annotation,
        "database_path": tmp_path / "reactome.sqlite",
        "manifest_path": tmp_path / "manifest.json",
        "release": "fixture-1",
    }
    first = build_reactome_index(**arguments)
    second = build_reactome_index(**arguments)
    assert first == second
