"""Load the frozen six-hour forecast model for OCS-LDA cycling."""

from pathlib import Path

import torch
import yaml

from models.BohaiForecastNet import BohaiForecastNet


class AttrDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value


def load_config(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return AttrDict(yaml.safe_load(handle))


def load_model(cfg, checkpoint, device):
    model = BohaiForecastNet(cfg).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    state = ckpt["model_state"] if "model_state" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model
