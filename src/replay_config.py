"""Validated configuration for an isolated hsl-player bag replay."""

from pathlib import Path
import math

import yaml


FIELDS = {
    "clahe_enabled": ("clahe", "enabled", bool),
    "clip_limit": ("clahe", "clip_limit", float),
    "tile_width": ("clahe", "tile_grid_size", int),
    "white_enabled": ("white_enhancement", "enabled", bool),
    "min_value": ("white_enhancement", "min_value", int),
    "max_saturation": ("white_enhancement", "max_saturation", int),
    "strength": ("white_enhancement", "strength", float),
    "protect_green": ("white_enhancement", "protect_green", bool),
    "max_excess_green": ("white_enhancement", "max_excess_green", int),
}


def validate(values):
    if set(values) != set(FIELDS):
        raise ValueError("Parâmetros incompletos ou desconhecidos")
    for key, value in values.items():
        kind = FIELDS[key][2]
        if kind is bool and type(value) is not bool:
            raise ValueError(f"{key}: esperado booleano")
        if kind is not bool and (type(value) not in (int, float) or not math.isfinite(value)):
            raise ValueError(f"{key}: esperado número finito")
    if not (0 < values["clip_limit"] <= 100):
        raise ValueError("clip_limit deve estar entre 0 e 100")
    if not (1 <= values["tile_width"] <= 64) or int(values["tile_width"]) != values["tile_width"]:
        raise ValueError("tile_width deve ser inteiro de 1 a 64")
    for key in ("min_value", "max_saturation"):
        if not (0 <= values[key] <= 255) or int(values[key]) != values[key]:
            raise ValueError(f"{key} deve ser inteiro de 0 a 255")
    if not (0 <= values["strength"] <= 1):
        raise ValueError("strength deve estar entre 0 e 1")
    if not (-510 <= values["max_excess_green"] <= 510) or int(values["max_excess_green"]) != values["max_excess_green"]:
        raise ValueError("max_excess_green deve ser inteiro de -510 a 510")


def load(base_path):
    config = yaml.safe_load(Path(base_path).read_text())
    pre = config["segmentation_model"]["line_preprocessing"]
    values = {key: pre[group][name] for key, (group, name, _) in FIELDS.items() if key != "tile_width"}
    values["tile_width"] = pre["clahe"]["tile_grid_size"][0]
    validate(values)
    return config, values


def write_config(base, values, destination):
    validate(values)
    # Never mutate hsl-player's source or installed configuration.
    import copy
    config = copy.deepcopy(base)
    pre = config["segmentation_model"]["line_preprocessing"]
    for key, (group, name, _) in FIELDS.items():
        pre[group][name] = [int(values[key])] * 2 if key == "tile_width" else values[key]
    config["publish_preprocess_video"] = True
    config["use_depth"] = False
    Path(destination).write_text(yaml.safe_dump(config, sort_keys=False))
