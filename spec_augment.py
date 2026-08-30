from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Union, Optional
import torch 
import torch.nn.functional as F
import torch.nn as nn

# ---------------------------------------------------------------------------
# Policies from Park et al. (2019)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SpecAugmentPolicy:
    """
    SpecAugment policy.

    W  : time-warp parameter
    F  : maximum frequency-mask width
    mF : number of frequency masks
    T  : maximum time-mask width
    p  : maximum proportion of time steps that can be masked
    mT : number of time masks
    """
    W: int = 0
    F: int = 27
    mF: int = 2
    T: int = 100
    p: float = 1.0
    mT: int = 2


# Original SpecAugment policies.
LIBRISPEECH_BASIC = SpecAugmentPolicy(
    W=80, F=27, mF=1, T=100, p=1.0, mT=1
)

LIBRISPEECH_DOUBLE = SpecAugmentPolicy(
    W=80, F=27, mF=2, T=100, p=1.0, mT=2
)

SWITCHBOARD_MILD = SpecAugmentPolicy(
    W=40, F=15, mF=2, T=70, p=0.2, mT=2
)

SWITCHBOARD_STRONG = SpecAugmentPolicy(
    W=40, F=27, mF=2, T=70, p=0.2, mT=2
)


def _randint(
    low: int,
    high: int,
    *,
    generator: Optional[torch.Generator],
    device: torch.device,
) -> int:
    """Inclusive low, exclusive high."""
    if high <= low:
        return low

    return int(
        torch.randint(
            low,
            high,
            (1,),
            generator=generator,
            device=device,
        ).item()
    )


def frequency_mask(
    x: torch.Tensor,
    max_width: int,
    num_masks: int,
    *,
    mask_value: float = 0.0,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Apply independent frequency masks to each example.

    x:
        [B, F, T]

    Each mask has width sampled uniformly from [0, max_width].
    """
    if x.ndim != 3:
        raise ValueError("Expected x with shape [B, F, T].")

    x = x.clone()

    batch, n_freq, _ = x.shape

    max_width = min(max_width, n_freq)

    if max_width <= 0 or num_masks <= 0:
        return x

    for b in range(batch):
        for _ in range(num_masks):
            width = _randint(
                0,
                max_width + 1,
                generator=generator,
                device=x.device,
            )

            if width == 0:
                continue

            start = _randint(
                0,
                n_freq - width + 1,
                generator=generator,
                device=x.device,
            )

            x[b, start:start + width, :] = mask_value

    return x

# ---------------------------------------------------------------------------
# Time masking
# ---------------------------------------------------------------------------

def time_mask(
    x: torch.Tensor,
    max_width: int,
    num_masks: int,
    *,
    mask_proportion: float = 1.0,
    lengths: Optional[torch.Tensor] = None,
    mask_value: float = 0.0,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Apply independent time masks to each example.

    x:
        [B, F, T]

    lengths:
        Optional tensor [B] containing the valid number of frames for
        each utterance.

    mask_proportion:
        Maximum fraction of valid frames that may be masked.

    Notes
    -----
    The effective maximum mask width is:

        min(max_width, floor(mask_proportion * valid_length))

    This allows the canonical SpecAugment p parameter to be reproduced
    while also supporting variable-length utterances.
    """
    if x.ndim != 3:
        raise ValueError("Expected x with shape [B, F, T].")

    x = x.clone()

    batch, _, n_frames = x.shape

    if lengths is None:
        lengths = torch.full(
            (batch,),
            n_frames,
            dtype=torch.long,
            device=x.device,
        )
    else:
        lengths = lengths.to(device=x.device, dtype=torch.long)

        if lengths.shape != (batch,):
            raise ValueError("lengths must have shape [B].")

    for b in range(batch):
        valid_length = min(int(lengths[b].item()), n_frames)

        if valid_length <= 0:
            continue

        proportion_width = int(
            mask_proportion * valid_length
        )

        effective_max = min(
            max_width,
            proportion_width,
            valid_length,
        )

        if effective_max <= 0:
            continue

        for _ in range(num_masks):
            width = _randint(
                0,
                effective_max + 1,
                generator=generator,
                device=x.device,
            )

            if width == 0:
                continue

            start = _randint(
                0,
                valid_length - width + 1,
                generator=generator,
                device=x.device,
            )

            x[b, :, start:start + width] = mask_value

    return x


def time_warp_grid_sample(
    x: torch.Tensor,
    W: int,
    *,
    lengths: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
    padding_mode: str = "border",
    align_corners: bool = True,
) -> torch.Tensor:
    """
    SpecAugment TimeWarp using torch.grid_sample.

    Parameters
    ----------
    x:
        Input spectrograms with shape [B, n_mels, n_frames].

    W:
        Maximum time displacement.

    lengths:
        Optional valid lengths, shape [B].
        Padding after each valid length is left unchanged.

    generator:
        Optional torch.Generator for reproducible sampling.

    padding_mode:
        grid_sample padding mode. "border" is used to avoid introducing
        artificial zero-valued features.

    align_corners:
        Coordinate convention for grid_sample. True makes the mapping
        between spectrogram frame indices and normalized coordinates
        especially straightforward.

    Returns
    -------
    Tensor
        Warped spectrograms, same shape as x.
    """
    if x.ndim != 3:
        raise ValueError(
            f"Expected x with shape [B, F, T], got {tuple(x.shape)}"
        )

    if W < 0:
        raise ValueError("W must be >= 0.")

    B, n_mels, n_frames = x.shape

    if lengths is None:
        lengths = torch.full(
            (B,),
            n_frames,
            dtype=torch.long,
            device=x.device,
        )
    else:
        lengths = lengths.to(
            device=x.device,
            dtype=torch.long,
        )

        if lengths.shape != (B,):
            raise ValueError(
                f"Expected lengths with shape [{B}], "
                f"got {tuple(lengths.shape)}"
            )

        lengths = lengths.clamp(min=0, max=n_frames)

    if W == 0:
        return x

    output = x.clone()

    for b in range(B):
        T = int(lengths[b].item())

        # TimeWarp requires a meaningful midpoint and two regions.
        if T < 4:
            continue

        # The original operation samples the displacement of the
        # spectrogram midpoint.
        max_displacement = min(W, T // 2 - 1)

        if max_displacement <= 0:
            continue

        displacement = int(
            torch.randint(
                -max_displacement,
                max_displacement + 1,
                (1,),
                generator=generator,
                device=x.device,
            ).item()
        )

        if displacement == 0:
            continue

        center = T // 2
        warped_center = center + displacement

        # ------------------------------------------------------------------
        # Construct inverse mapping:
        #
        # output_time -> source_time
        #
        # The source midpoint (center) is mapped to warped_center.
        #
        # Left:
        #   source 0      -> target 0
        #   source center -> target warped_center
        #
        # Right:
        #   source center -> target warped_center
        #   source T-1    -> target T-1
        # ------------------------------------------------------------------

        target = torch.arange(
            T,
            dtype=torch.float32,
            device=x.device,
        )

        source = torch.empty_like(target)

        # Left side.
        left = target <= warped_center

        source[left] = (
            target[left]
            * center
            / warped_center
        )

        # Right side.
        right = ~left

        source[right] = (
            center
            + (target[right] - warped_center)
            * (T - 1 - center)
            / (T - 1 - warped_center)
        )

        # Convert source frame indices [0, T-1] to the normalized
        # grid_sample coordinate range [-1, 1].
        if align_corners:
            x_coord = (
                2.0 * source / (T - 1)
                - 1.0
            )
        else:
            x_coord = (
                2.0 * (source + 0.5) / T
                - 1.0
            )

        # The frequency coordinate is identity.
        y_coord = torch.linspace(
            -1.0,
            1.0,
            n_mels,
            device=x.device,
            dtype=x.dtype,
        )

        # [F, T] -> [1, F, T, 2]
        grid_y = y_coord[:, None].expand(
            n_mels,
            T,
        )

        grid_x = x_coord[None, :].expand(
            n_mels,
            T,
        )

        grid = torch.stack(
            [grid_x, grid_y],
            dim=-1,
        ).unsqueeze(0)

        # grid_sample expects [N, C, H, W].
        #
        # Interpret:
        #   H = frequency
        #   W = time
        #
        # C=1 because frequency is already represented by H.
        frame = x[b, :, :T].unsqueeze(0).unsqueeze(1)

        warped = F.grid_sample(
            frame,
            grid,
            mode="bilinear",
            padding_mode=padding_mode,
            align_corners=align_corners,
        )

        output[b, :, :T] = warped.squeeze(0).squeeze(0)

    return output


class SpecAugment(nn.Module):
    def __init__(
        self,
        policy,
        *,
        use_time_warp=True,
        mask_value=0.0,
    ):
        super().__init__()

        self.policy = policy
        self.use_time_warp = use_time_warp
        self.mask_value = mask_value

    def forward(
        self,
        x,
        *,
        lengths=None,
        generator=None,
    ):
        if self.use_time_warp and self.policy.W > 0:
            x = time_warp_grid_sample(
                x,
                self.policy.W,
                lengths=lengths,
                generator=generator,
            )

        x = frequency_mask(
            x,
            self.policy.F,
            self.policy.mF,
            mask_value=self.mask_value,
            generator=generator,
        )

        x = time_mask(
            x,
            self.policy.T,
            self.policy.mT,
            mask_proportion=self.policy.p,
            lengths=lengths,
            mask_value=self.mask_value,
            generator=generator,
        )

        return x

if __name__ == "__main__":
    B = 8
    n_mels = 80
    n_frames = 300

    x = torch.randn(B, n_mels, n_frames)

    generator = torch.Generator(device=x.device)
    generator.manual_seed(1234)

    x_warped = time_warp_grid_sample(
        x,
        W=0,
        generator=generator,
    )
    assert(torch.isclose(x, x_warped).all().item())
    # torch.Size([8, 80, 300])


    # And with variable-length utterances:

    lengths = torch.tensor([
        275, 231, 300, 187,
        264, 142, 298, 211,
    ])

    x_warped = time_warp_grid_sample(
        x,
        W=80,
        lengths=lengths,
        generator=generator,
    )

    print(x_warped.size())

