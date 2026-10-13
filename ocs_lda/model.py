from __future__ import annotations

import math
from collections import OrderedDict
from typing import Dict, Mapping, Optional

import torch
import torch.nn as nn

from ocs_lda.structured_autoencoder import StructuredAutoencoder
from ocs_lda.swin_components import cfg_get

Tensor = torch.Tensor


def _resolve_branch_value(value, branch_name: str, default: float) -> float:
    """Resolve a scalar or branch-name mapping from the model configuration."""
    if isinstance(value, Mapping):
        return float(value.get(branch_name, value.get("default", default)))
    return float(value)


class IndependentGatedSharedHubMixer(nn.Module):
    """Shared/private hub with one learnable private-to-shared gate per branch.

    Each private summary is independently gated before the common hub:

        shared_new = shared + F(shared, gate_1*m_1, ..., gate_n*m_n)
                            - F(shared, 0, ..., 0)

    The null-private response is subtracted so that all-zero gates produce an
    exact identity update for the shared branch. The common nonlinear hub and
    branch-specific FiLM paths condition the private outputs.

    Gates use a bounded signed tanh parameterization. The 1x1 projections learn
    response sign and structure; the gates provide an interpretable coupling
    strength and cannot grow without bound.
    """

    def __init__(
        self,
        branch_channels: Mapping[str, int],
        shared_name: str = "shared",
        hidden_ch: Optional[int] = None,
        gate_init=0.0,
        gate_max: float = 1.0,
        gate_trainable: bool = True,
    ):
        super().__init__()
        self.branch_channels = OrderedDict(
            (name, int(channels)) for name, channels in branch_channels.items()
        )
        self.shared_name = str(shared_name)
        if self.shared_name not in self.branch_channels:
            raise ValueError(
                f"shared_name={self.shared_name!r} not found in branch_channels"
            )

        self.private_names = [
            name for name in self.branch_channels if name != self.shared_name
        ]
        if not self.private_names:
            raise ValueError("At least one private branch is required.")

        self.shared_ch = self.branch_channels[self.shared_name]
        hidden_ch = int(hidden_ch or self.shared_ch)
        self.gate_max = float(gate_max)
        if not math.isfinite(self.gate_max) or self.gate_max <= 0:
            raise ValueError("hub_gate_max must be a finite positive scalar.")

        self.private_to_shared = nn.ModuleDict(
            {
                name: nn.Conv2d(
                    self.branch_channels[name],
                    self.shared_ch,
                    kernel_size=1,
                )
                for name in self.private_names
            }
        )

        hub_in_ch = self.shared_ch * (1 + len(self.private_names))
        self.shared_update = nn.Sequential(
            nn.Conv2d(hub_in_ch, hidden_ch, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_ch, self.shared_ch, kernel_size=1),
        )

        self.raw_gates = nn.ParameterDict()
        for name in self.private_names:
            initial_gate = _resolve_branch_value(gate_init, name, 0.0)
            raw_value = self._gate_to_raw(initial_gate)
            parameter = nn.Parameter(
                torch.tensor(raw_value, dtype=torch.get_default_dtype()),
                requires_grad=bool(gate_trainable),
            )
            self.raw_gates[name] = parameter

        # Shared-to-private FiLM conditions each private branch.
        self.film = nn.ModuleDict()
        for name in self.private_names:
            channels = self.branch_channels[name]
            mid = max(channels, self.shared_ch)
            modulator = nn.Sequential(
                nn.Conv2d(self.shared_ch, mid, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(mid, 2 * channels, kernel_size=1),
            )
            nn.init.zeros_(modulator[-1].weight)
            nn.init.zeros_(modulator[-1].bias)
            self.film[name] = modulator

    def _gate_to_raw(self, gate: float) -> float:
        """Convert an actual bounded gate value to its unconstrained parameter."""
        gate = float(gate)
        if not math.isfinite(gate):
            raise ValueError("Hub gate initial values must be finite.")
        ratio = gate / self.gate_max
        eps = 1e-6
        ratio = max(-1.0 + eps, min(1.0 - eps, ratio))
        return math.atanh(ratio)

    def gate_values(
        self,
        branch_mask: Optional[Mapping[str, float]] = None,
    ) -> "OrderedDict[str, Tensor]":
        """Return bounded gate values, optionally multiplied by a runtime mask."""
        values: "OrderedDict[str, Tensor]" = OrderedDict()
        for name in self.private_names:
            gate = self.gate_max * torch.tanh(self.raw_gates[name])
            if branch_mask is not None:
                gate = gate * float(branch_mask.get(name, 1.0))
            values[name] = gate
        return values

    def forward(
        self,
        branches: Mapping[str, Tensor],
        branch_mask: Optional[Mapping[str, float]] = None,
    ) -> Dict[str, Tensor]:
        shared = branches[self.shared_name]
        gates = self.gate_values(branch_mask=branch_mask)

        gated_summaries = []
        null_summaries = []
        for name in self.private_names:
            summary = self.private_to_shared[name](branches[name])
            gated_summaries.append(gates[name] * summary)
            null_summaries.append(torch.zeros_like(summary))

        hub_input = torch.cat([shared, *gated_summaries], dim=1)
        null_input = torch.cat([shared, *null_summaries], dim=1)
        private_conditioned_update = (
            self.shared_update(hub_input) - self.shared_update(null_input)
        )
        shared_new = shared + private_conditioned_update

        out: "OrderedDict[str, Tensor]" = OrderedDict()
        out[self.shared_name] = shared_new
        for name in self.private_names:
            private = branches[name]
            gamma_shift = self.film[name](shared_new)
            gamma, shift = torch.chunk(gamma_shift, chunks=2, dim=1)
            out[name] = private * (1.0 + gamma) + shift
        return out

    def extra_repr(self) -> str:
        return (
            f"shared_name={self.shared_name!r}, "
            f"private_names={self.private_names}, gate_max={self.gate_max}"
        )


class OCSLDA(StructuredAutoencoder):
    """OCS-LDA structured autoencoder with branch-independent hub gates."""

    def __init__(self, params, norm_layer=nn.LayerNorm):
        super().__init__(params=params, norm_layer=norm_layer)

        old_hub = self.bottleneck.branch_latent.hub
        gate_init = cfg_get(params, "hub_gate_init", 0.0)
        gate_max = float(cfg_get(params, "hub_gate_max", 1.0))
        gate_trainable = bool(cfg_get(params, "hub_gate_trainable", True))

        new_hub = IndependentGatedSharedHubMixer(
            branch_channels=self.branch_channels,
            shared_name="shared",
            gate_init=gate_init,
            gate_max=gate_max,
            gate_trainable=gate_trainable,
        )

        # Preserve the base initialization for unchanged Hub and FiLM tensors.
        transferable = OrderedDict(
            (key, value)
            for key, value in old_hub.state_dict().items()
            if key != "beta"
        )
        incompatible = new_hub.load_state_dict(transferable, strict=False)
        expected_missing = {
            f"raw_gates.{name}" for name in new_hub.private_names
        }
        unexpected_missing = set(incompatible.missing_keys) - expected_missing
        if unexpected_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "Unexpected Hub initialization mismatch: "
                f"missing={sorted(unexpected_missing)}, "
                f"unexpected={sorted(incompatible.unexpected_keys)}"
            )

        self.bottleneck.branch_latent.hub = new_hub
        self.branch_latent = self.bottleneck.branch_latent

    def hub_gate_values(
        self,
        detach: bool = False,
    ) -> "OrderedDict[str, Tensor]":
        values = self.bottleneck.branch_latent.hub.gate_values()
        if detach:
            return OrderedDict(
                (name, value.detach()) for name, value in values.items()
            )
        return values

    def decode(
        self,
        z: Tensor,
        output_size=None,
        return_parts: bool = False,
    ):
        output = super().decode(
            z=z,
            output_size=output_size,
            return_parts=return_parts,
        )
        if return_parts:
            output["hub_gates"] = self.hub_gate_values(detach=False)
        return output


__all__ = [
    "IndependentGatedSharedHubMixer",
    "OCSLDA",
]
