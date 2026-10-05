from triad_pcb.config import load_yaml, resolve_path, save_yaml


def test_config_inheritance_and_relative_resolution(tmp_path):
    base = tmp_path / "configs" / "base.yaml"
    child = tmp_path / "configs" / "child.yaml"
    save_yaml({"model": {"depth": 4, "width": 64}, "output": {"root": "outputs"}}, base)
    save_yaml({"inherits": "base.yaml", "model": {"depth": 6}}, child)

    config = load_yaml(child)

    assert config["model"] == {"depth": 6, "width": 64}
    assert resolve_path(config, config["output"]["root"]) == tmp_path / "outputs"
