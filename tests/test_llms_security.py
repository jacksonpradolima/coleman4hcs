"""Documentation generation rejects path escapes and never follows output symlinks."""

import pytest

from scripts import generate_llms_txt as generator


def repository(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "docs").mkdir()
    (root / "README.md").write_text("# Synthetic CI documentation\n")
    (root / "docs" / "ci.md").write_text("Prioritize CI tests.\n")
    monkeypatch.setattr(generator, "ROOT", root)
    monkeypatch.setattr(generator, "OUT", root / "llms.txt")
    return root


def test_output_symlink_does_not_overwrite_outside_file(tmp_path, monkeypatch):
    root = repository(tmp_path, monkeypatch)
    outside = tmp_path / "outside.txt"
    outside.write_text("untouched")
    (root / "llms.txt").symlink_to(outside)
    with pytest.raises(ValueError, match="regular repository-local"):
        generator.main()
    assert outside.read_text() == "untouched"


def test_source_symlink_and_parent_directory_escape_are_rejected(tmp_path, monkeypatch):
    root = repository(tmp_path, monkeypatch)
    outside = tmp_path / "external"
    outside.mkdir()
    (outside / "outside.md").write_text("outside fixture")
    linked = root / "docs" / "linked.md"
    linked.symlink_to(outside / "outside.md")
    with pytest.raises(ValueError, match="symlinked source"):
        generator._read_text(linked)
    parent = root / "linked-directory"
    parent.symlink_to(outside, target_is_directory=True)
    escaped = parent / "outside.md"
    with pytest.raises(ValueError, match="escapes repository"):
        generator._read_text(escaped)


def test_interrupted_publication_keeps_previous_output_and_cleans_temporary(tmp_path, monkeypatch):
    root = repository(tmp_path, monkeypatch)
    output = root / "llms.txt"
    output.write_text("previous complete output")

    def interrupted(*_args):
        raise OSError("injected publication interruption")

    monkeypatch.setattr(generator.os, "replace", interrupted)
    with pytest.raises(OSError, match="injected"):
        generator.main()
    assert output.read_text() == "previous complete output"
    assert list(root.glob("*.tmp")) == []


def test_normal_generation_is_deterministic_and_replaces_a_complete_file(tmp_path, monkeypatch):
    root = repository(tmp_path, monkeypatch)
    generator.main()
    first = (root / "llms.txt").read_text()
    generator.main()
    assert (root / "llms.txt").read_text() == first
    assert "Prioritize CI tests." in first
    assert list(root.glob("*.tmp")) == []
