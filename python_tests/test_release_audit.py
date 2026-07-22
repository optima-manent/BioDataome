import pytest
from cskl_atlas.catalog import Catalog
from cskl_atlas.release_audit import audit_release


def test_release_audit_rejects_unknown_profile(tmp_path) -> None:
    catalog = Catalog(tmp_path / "atlas.sqlite")
    catalog.initialize()
    with pytest.raises(ValueError, match="profile"):
        audit_release(catalog, snapshot_id="missing", profile="marketing")
