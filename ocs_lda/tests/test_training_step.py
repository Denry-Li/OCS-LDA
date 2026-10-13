"""Exercise the packaged training forward/loss/backward path."""

from pathlib import Path

import torch
import yaml

from ocs_lda import OCSLDA
from ocs_lda.loss_bohai import StructuredAEReconstructionLoss
from ocs_lda.train_bohai import forward_model


def main() -> None:
    config_path = Path("configs/final/ocs_lda_d32_model.yaml")
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    torch.manual_seed(42)
    model = OCSLDA(cfg).train()
    x = torch.randn(1, 5, 40, 40)
    output = forward_model(model, x, {}, cfg)
    loss, _ = StructuredAEReconstructionLoss(cfg, cfg["target_vars"])(output, x)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    print(f"training step passed; loss={float(loss.detach()):.8f}")


if __name__ == "__main__":
    main()
