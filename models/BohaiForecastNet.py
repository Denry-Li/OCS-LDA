import torch
import torch.nn as nn

try:
    from models.SwimIRv2 import SwinTransformerBlock
except Exception:
    SwinTransformerBlock = None


class ConvResidualBlock(nn.Module):
    def __init__(self, channels: int, hidden_channels: int | None = None):
        super().__init__()
        hidden_channels = hidden_channels or channels
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.net(x)


class SwinStage(nn.Module):
    def __init__(self, dim: int, resolution: tuple[int, int], depth: int, heads: int, window_size: int):
        super().__init__()
        self.use_swin = SwinTransformerBlock is not None and depth > 0
        if self.use_swin:
            blocks = []
            for i in range(depth):
                blocks.append(
                    SwinTransformerBlock(
                        dim=dim,
                        input_resolution=resolution,
                        num_heads=heads,
                        window_size=window_size,
                        shift_size=0 if i % 2 == 0 else window_size // 2,
                        mlp_ratio=2.0,
                        qkv_bias=True,
                        drop=0.0,
                        attn_drop=0.0,
                        drop_path=0.0,
                    )
                )
            self.blocks = nn.ModuleList(blocks)
        else:
            self.blocks = nn.ModuleList([ConvResidualBlock(dim) for _ in range(max(depth, 1))])

    def forward(self, x):
        if self.use_swin:
            b, c, h, w = x.shape
            x_seq = x.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)
            for block in self.blocks:
                x_seq = block(x_seq, (h, w))
            return x_seq.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        for block in self.blocks:
            x = block(x)
        return x


class BohaiForecastNet(nn.Module):
    """Lightweight regional deterministic forecast operator.

    The model predicts a standardized-space residual by default:
        x_{t+1} = x_t + delta(x_t)
    This is deliberately smaller than the original LDA global forecast_net and
    is meant to be trained from scratch on 40x40, 5-variable regional samples.
    """

    def __init__(self, cfg):
        super().__init__()
        in_chans = int(cfg.get("in_chans", len(cfg.get("input_vars", [])) or 5))
        out_chans = int(cfg.get("out_chans", in_chans))
        img_h = int(cfg.get("img_size_y", cfg.get("img_size", [40, 40])[0]))
        img_w = int(cfg.get("img_size_x", cfg.get("img_size", [40, 40])[-1]))
        dim = int(cfg.get("forecast_dim", cfg.get("enc_dim", 64)))
        bottleneck_dim = int(cfg.get("forecast_bottleneck_dim", cfg.get("embed_dim", 128)))
        depths = list(cfg.get("forecast_depths", [2, 2, 2]))
        heads = list(cfg.get("forecast_heads", [2, 4, 4]))
        window = int(cfg.get("forecast_window_size", cfg.get("window_size", 5)))
        self.residual_prediction = bool(cfg.get("residual_prediction", True))
        self.in_chans = in_chans
        self.forecast_input_steps = int(cfg.get("forecast_input_steps", 1))
        self.use_time_features = bool(cfg.get("use_time_features", False))
        self.time_feature_channels = int(
            cfg.get("forecast_time_feature_channels", 4 if self.use_time_features else 0)
        )
        self.downsample_factor = int(cfg.get("forecast_downsample_factor", 2))
        if self.forecast_input_steps not in (1, 2):
            raise ValueError("forecast_input_steps must be 1 or 2.")
        if self.downsample_factor not in (1, 2):
            raise ValueError("forecast_downsample_factor must be 1 or 2.")
        stem_in_chans = in_chans * self.forecast_input_steps
        if self.use_time_features:
            stem_in_chans += self.time_feature_channels

        self.stem = nn.Sequential(
            nn.Conv2d(stem_in_chans, dim, kernel_size=3, padding=1),
            nn.GELU(),
            ConvResidualBlock(dim),
        )
        self.enc_stage = SwinStage(dim, (img_h, img_w), depths[0], heads[0], window)
        if self.downsample_factor == 2:
            self.down = nn.Sequential(
                nn.Conv2d(dim, bottleneck_dim, kernel_size=3, stride=2, padding=1),
                nn.GELU(),
            )
            self.up = nn.Sequential(
                nn.ConvTranspose2d(bottleneck_dim, dim, kernel_size=2, stride=2),
                nn.GELU(),
            )
            low_h = (img_h + 1) // 2
            low_w = (img_w + 1) // 2
        else:
            self.down = nn.Sequential(
                nn.Conv2d(dim, bottleneck_dim, kernel_size=1),
                nn.GELU(),
            )
            self.up = nn.Sequential(
                nn.Conv2d(bottleneck_dim, dim, kernel_size=1),
                nn.GELU(),
            )
            low_h = img_h
            low_w = img_w
        self.dyn_stage = SwinStage(
            bottleneck_dim,
            (low_h, low_w),
            depths[1] if len(depths) > 1 else depths[0],
            heads[1] if len(heads) > 1 else heads[0],
            window,
        )
        self.dec_stage = SwinStage(
            dim,
            (img_h, img_w),
            depths[2] if len(depths) > 2 else depths[-1],
            heads[2] if len(heads) > 2 else heads[-1],
            window,
        )
        self.head = nn.Conv2d(dim, out_chans, kernel_size=3, padding=1)

    def _prepare_input(self, previous_state, current_state, time_features):
        if self.forecast_input_steps == 1:
            if current_state is None:
                current_state = previous_state
            model_inputs = [current_state]
        else:
            if current_state is None:
                expected = self.in_chans * 2
                if previous_state.shape[1] != expected:
                    raise ValueError(
                        "Two-step ForecastNet requires previous_state and current_state "
                        f"or a concatenated tensor with {expected} channels."
                    )
                previous_state, current_state = torch.split(
                    previous_state, self.in_chans, dim=1
                )
            model_inputs = [previous_state, current_state]

        if self.use_time_features:
            if time_features is None:
                raise ValueError("time_features are required when use_time_features=true.")
            if time_features.shape[1] != self.time_feature_channels:
                raise ValueError(
                    f"Expected {self.time_feature_channels} time-feature channels, "
                    f"got {time_features.shape[1]}."
                )
            model_inputs.append(time_features)
        return torch.cat(model_inputs, dim=1), current_state

    def forward(self, previous_state, current_state=None, time_features=None):
        model_input, residual_base = self._prepare_input(
            previous_state, current_state, time_features
        )
        feat = self.stem(model_input)
        skip = self.enc_stage(feat)
        low = self.down(skip)
        low = self.dyn_stage(low)
        up = self.up(low)
        if up.shape[-2:] != residual_base.shape[-2:]:
            up = up[..., : residual_base.shape[-2], : residual_base.shape[-1]]
        feat = self.dec_stage(up + skip)
        delta = self.head(feat)
        if self.residual_prediction and delta.shape == residual_base.shape:
            return residual_base + delta
        return delta


class BohaiForecastOperator:
    def __init__(self, model: nn.Module):
        self.model = model

    def integrate(
        self,
        x,
        step: int,
        detach: bool = True,
        ckp: bool = False,
        previous=None,
        time_features=None,
    ):
        del ckp
        out = x
        single = out.ndim == 3
        if single:
            out = out.unsqueeze(0)
            if previous is not None and previous.ndim == 3:
                previous = previous.unsqueeze(0)
            if time_features is not None and time_features.ndim == 4:
                time_features = time_features.unsqueeze(0)
        if self.model.forecast_input_steps == 2 and previous is None:
            raise ValueError("previous state is required for a two-step forecast model.")
        for step_index in range(int(step)):
            feature = None
            if time_features is not None:
                feature = time_features[:, step_index]
            if self.model.forecast_input_steps == 2:
                next_state = self.model(previous, out, feature)
                previous, out = out, next_state
            else:
                out = self.model(out, time_features=feature)
            if detach:
                out = out.detach()
        return out.squeeze(0) if single else out
