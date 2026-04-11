"""FLARE-based student model for overlap score regression."""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from typing import Any, Optional

import sys
import torch
import torch.nn as nn

from .base import OverlapStudent, StudentBatch, StudentPred
from .registry import register_student


def _unwrap_checkpoint_state_dict(ckpt: object) -> dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            maybe_sd = ckpt.get(key)
            if isinstance(maybe_sd, dict):
                return maybe_sd
        if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt
    raise KeyError(
        "Could not find a model state dict in checkpoint; "
        "expected one of ['model', 'state_dict', 'model_state_dict'] or a flat tensor dict."
    )


def _extract_backbone_state_dict(ckpt: object) -> dict[str, torch.Tensor]:
    state_dict = _unwrap_checkpoint_state_dict(ckpt)
    state_dict = {
        k[len("module."):] if k.startswith("module.") else k: v
        for k, v in state_dict.items()
    }

    backbone_state_dict = {
        k[len("backbone."):]: v
        for k, v in state_dict.items()
        if k.startswith("backbone.")
    }
    if backbone_state_dict:
        return backbone_state_dict
    return state_dict


def _default_init_checkpoint_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    candidates = (
        repo_root / "checkpoints" / "leangate.pt",
        repo_root
        / "checkpoints"
        / "iterative"
        / "flare_iter4_dec6_train_decoder_lr5e-5_img256_bs256x4_20ep"
        / "overlap_iterativemodel_dec6_full.pt",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "No LeanGate initialization checkpoint found. "
        f"Expected one of: {candidates[0]} or {candidates[1]}. "
        "Pass `flare_ckpt` explicitly if your checkpoint lives elsewhere."
    )


def _warn_if_curope2d_unavailable() -> None:
    # Best-effort warning only: inference can still proceed via the slow PyTorch RoPE2D fallback.
    try:
        from models.curope import curope as _kernels  # type: ignore[import-not-found]

        if not hasattr(_kernels, "rope_2d"):
            raise ImportError("missing symbol rope_2d")
        return
    except Exception as exc:
        repo_root = Path(__file__).resolve().parents[2]
        mast3r_curope_dir = (
            repo_root
            / "third_party"
            / "MASt3R-SLAM"
            / "thirdparty"
            / "mast3r"
            / "dust3r"
            / "croco"
            / "models"
            / "curope"
        )
        mast3r_prebuilt = sorted(mast3r_curope_dir.glob("curope*.so")) if mast3r_curope_dir.exists() else []
        if mast3r_prebuilt:
            mast3r_dir_str = str(mast3r_curope_dir)
            if mast3r_dir_str not in sys.path:
                sys.path.insert(0, mast3r_dir_str)
            try:
                # If MASt3R-SLAM has a built `curope*.so`, load it as a top-level module.
                # FLARE's curope2d.py will then pick it up via `import curope`.
                if "curope" in sys.modules:
                    del sys.modules["curope"]
                import curope as _curope_kernels  # type: ignore[import-not-found]

                if hasattr(_curope_kernels, "rope_2d"):
                    print(
                        "[FlareStudent] [info] Using MASt3R-SLAM-built `curope` extension for cuRoPE2D.",
                        file=sys.stderr,
                        flush=True,
                    )
                    return
            except Exception:
                # Fall through to warning below.
                pass

        curope_dir = (
            repo_root / "third_party" / "FLARE"
            / "dust3r"
            / "croco"
            / "models"
            / "curope"
        )
        py_tag = f"{sys.version_info.major}{sys.version_info.minor}"
        prebuilt = sorted(curope_dir.glob("curope.cpython-*.so")) if curope_dir.exists() else []
        prebuilt_tags = [p.name.split("curope.cpython-")[-1].split("-")[0] for p in prebuilt]
        mismatch_hint = ""
        if prebuilt_tags and all(tag != py_tag for tag in prebuilt_tags):
            mismatch_hint = (
                f" (found prebuilt tags={prebuilt_tags}, but running Python {sys.version_info.major}.{sys.version_info.minor})"
            )
        print(
            "[FlareStudent] [warn] cuRoPE2D requested but curope extension failed to load; "
            "falling back to slow PyTorch RoPE2D. "
            f"Reason: {type(exc).__name__}: {exc}{mismatch_hint}. "
            "To build in-place: "
            "`cd third_party/FLARE/dust3r/croco/models/curope && python3 setup.py build_ext --inplace`",
            file=sys.stderr,
            flush=True,
        )


class OverlapHeadBase(nn.Module):
    """Abstract overlap head interface."""

    def forward(self, view_tokens: torch.Tensor, trunk: nn.Module) -> torch.Tensor:  # pragma: no cover - interface
        raise NotImplementedError


class SimpleOverlapHead(OverlapHeadBase):
    """Single-step CLS-style overlap head (scheme 1)."""

    def __init__(self, hidden_dim: int, overlap_dim: int = 8) -> None:
        super().__init__()
        # overlap_dim kept for signature compatibility / future use
        self.score_token = nn.Parameter(torch.empty(1, 1, hidden_dim))
        nn.init.normal_(self.score_token, std=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, view_tokens: torch.Tensor, trunk: nn.Module) -> torch.Tensor:
        # view_tokens: [B, 2, C]
        B, _, C = view_tokens.shape
        score_token = self.score_token.expand(B, 1, C)
        tokens = torch.cat([score_token, view_tokens], dim=1)
        tokens = trunk(tokens)
        cls_feat = tokens[:, 0]
        score = self.mlp(cls_feat).squeeze(-1)
        return score


class IterativeOverlapHead(OverlapHeadBase):
    """Iterative pose-head-style overlap head (scheme 2)."""

    def __init__(self, hidden_dim: int, overlap_dim: int = 8, num_iters: int = 3) -> None:
        super().__init__()
        self.overlap_dim = overlap_dim
        self.num_iters = num_iters
        self.overlap_proj = nn.Linear(overlap_dim, hidden_dim)
        self.overlap_branch = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, overlap_dim),
        )
        self.readout = nn.Sequential(
            nn.Linear(overlap_dim, overlap_dim),
            nn.GELU(),
            nn.Linear(overlap_dim, 1),
        )

    def forward(self, view_tokens: torch.Tensor, trunk: nn.Module) -> torch.Tensor:
        # view_tokens: [B, 2, C]
        B, _, _ = view_tokens.shape
        overlap_enc = torch.zeros(B, 1, self.overlap_dim, device=view_tokens.device, dtype=view_tokens.dtype)
        tokens_view = view_tokens

        for _ in range(self.num_iters):
            ov_token = self.overlap_proj(overlap_enc)  # [B, 1, C]
            tokens_in = torch.cat([ov_token, tokens_view], dim=1)  # [B, 1 + 2, C]
            tokens_out = trunk(tokens_in)
            cls_out = tokens_out[:, 0, :]
            tokens_view = tokens_out[:, 1:, :]
            delta_ov = self.overlap_branch(cls_out)  # [B, overlap_dim]
            overlap_enc = overlap_enc + delta_ov.unsqueeze(1)

        score = self.readout(overlap_enc.squeeze(1)).squeeze(-1)
        return score


def _ensure_flare_path() -> None:
    """Ensure FLARE's `mast3r`/`dust3r`/`croco` are the ones being imported.

    This repo vendors two different projects that both ship a top-level `mast3r`
    package (`third_party/FLARE/...` and `third_party/MASt3R-SLAM/...`). When
    training/using `FlareStudent`, we must ensure we import *only* FLARE's copy;
    otherwise we can silently pick up the other implementation and its (possibly
    unpatched) CUDA extensions.
    """
    repo_root = Path(__file__).resolve().parents[2]
    flare_root = repo_root / "third_party" / "FLARE"
    dust3r_repo = flare_root / "dust3r"  # contains the `dust3r/` python package
    croco_repo = dust3r_repo / "croco"  # contains the `models/` package used by curope

    mast3r_slam_root = repo_root / "third_party" / "MASt3R-SLAM" / "thirdparty" / "mast3r"

    def _path_under(path: Path, root: Path) -> bool:
        try:
            path = path.resolve()
            root = root.resolve()
        except OSError:
            return False
        return root == path or root in path.parents

    def _purge_modules_from_root(root: Path, prefixes: tuple[str, ...]) -> None:
        root_resolved = root.resolve()
        for name, module in list(sys.modules.items()):
            if not any(name == p or name.startswith(p + ".") for p in prefixes):
                continue
            module_file = getattr(module, "__file__", None)
            if not module_file:
                continue
            try:
                module_path = Path(module_file).resolve()
            except OSError:
                continue
            if root_resolved in module_path.parents:
                del sys.modules[name]

    # If MASt3R-SLAM's `mast3r` was already imported, purge it so FLARE can take over.
    if mast3r_slam_root.exists():
        _purge_modules_from_root(mast3r_slam_root, prefixes=("mast3r", "dust3r", "croco", "models"))

        # Demote any MASt3R-SLAM `mast3r` paths so FLARE wins name resolution.
        demoted: list[str] = []
        kept: list[str] = []
        for p in sys.path:
            try:
                is_under = _path_under(Path(p), mast3r_slam_root)
            except (TypeError, ValueError):
                is_under = False
            if is_under:
                demoted.append(p)
            else:
                kept.append(p)
        sys.path[:] = kept + demoted

    # Prepend FLARE paths so its packages win.
    for p in (croco_repo, dust3r_repo, flare_root):
        p_str = str(p)
        if p_str in sys.path:
            sys.path.remove(p_str)
        sys.path.insert(0, p_str)

    # Sanity check: ensure `mast3r` resolves to FLARE, not MASt3R-SLAM.
    try:
        import mast3r as _mast3r_pkg  # type: ignore

        mast3r_file = getattr(_mast3r_pkg, "__file__", None)
        if mast3r_file and not _path_under(Path(mast3r_file), flare_root):
            raise ImportError(
                "Import conflict: expected FLARE's `mast3r` but got a different one: "
                f"{mast3r_file}. Ensure FLARE comes first on PYTHONPATH."
            )
    except ModuleNotFoundError:
        # Let the actual caller fail with a clearer traceback later.
        pass


def _load_flare_backbone(flare_ckpt: str, device: torch.device, *, enable_curope2d: bool) -> nn.Module:
    """
    Load FLARE AsymmetricMASt3R model for student usage.

    We assume geometry checkpoints without an 'args' field and directly
    instantiate the corresponding AsymmetricMASt3R architecture, then load
    its state dict in a tolerant way.
    """
    _ensure_flare_path()
    # Control whether FLARE tries to import/use cuRoPE2D.
    os.environ["ENABLE_CUROPE2D"] = "1" if enable_curope2d else "0"
    if enable_curope2d:
        _warn_if_curope2d_unavailable()
    import torch  # local import to avoid side effects at module import time
    from mast3r.model import AsymmetricMASt3R  # type: ignore[import]

    ckpt = torch.load(flare_ckpt, map_location="cpu")
    inf = float("inf")
    model = AsymmetricMASt3R(
        pos_embed="RoPE100",
        patch_embed_cls="ManyAR_PatchEmbed",
        img_size=(512, 512),
        head_type="catmlp+dpt",
        output_mode="pts3d+desc24",
        depth_mode=("exp", -inf, inf),
        conf_mode=("exp", 1, inf),
        enc_embed_dim=1024,
        enc_depth=24,
        enc_num_heads=16,
        dec_embed_dim=768,
        dec_depth=12,
        dec_num_heads=12,
        two_confs=True,
        desc_conf_mode=("exp", 0, inf),
    )

    state_dict = _extract_backbone_state_dict(ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


@dataclasses.dataclass
class FlareFrame:
    """Frame-like container holding FLARE encoder caches.

    Mirrors the MASt3R-SLAM pattern of caching `feat` / `pos` on the frame object
    so repeated pairwise scoring can reuse a persistent reference frame without
    re-encoding it.
    """

    img: torch.Tensor
    true_shape: torch.Tensor
    feat: Optional[torch.Tensor] = None
    pos: Optional[torch.Tensor] = None


@register_student("flare")
class FlareStudent(OverlapStudent):
    """FLARE student that regresses a single pairwise score from RGB pairs."""

    def __init__(
        self,
        device: Optional[str] = None,
        flare_ckpt: Optional[str] = None,
        train_trunk: bool = False,
        enable_curope2d: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        device_str = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device_str)

        def _trunc_normal_(tensor: torch.Tensor, *, std: float = 0.02) -> None:
            # Match common transformer init: trunc at +/- 2 std.
            a = -2.0 * std
            b = 2.0 * std
            torch.nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=a, b=b)

        def _init_module_trunc_normal(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                _trunc_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                return
            if isinstance(module, nn.MultiheadAttention):
                # MultiheadAttention stores params directly on the module.
                if getattr(module, "in_proj_weight", None) is not None:
                    _trunc_normal_(module.in_proj_weight)  # type: ignore[arg-type]
                if getattr(module, "in_proj_bias", None) is not None:
                    nn.init.zeros_(module.in_proj_bias)  # type: ignore[arg-type]
                if getattr(module, "out_proj", None) is not None:
                    _init_module_trunc_normal(module.out_proj)
                return
            if isinstance(module, nn.LayerNorm) and getattr(module, "elementwise_affine", False):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        head_type = int(kwargs.pop("overlap_head_type", 1))
        overlap_dim = int(kwargs.pop("overlap_dim", 8))
        overlap_iters = int(kwargs.pop("overlap_iters", 1))
        decoder_depth = kwargs.pop("decoder_depth", None)  # None = keep all 12 layers
        decoder_layer_scheme = kwargs.pop("decoder_layer_scheme", None)  # e.g. "skipmid"
        decoder_layer_indices = kwargs.pop("decoder_layer_indices", None)  # e.g. [3, 7, 12] (1-based) or [2, 6, 11]
        decoder_weight_init = kwargs.pop("decoder_weight_init", None)  # "pretrained" (default) or "random"
        decoder_init_seed = kwargs.pop("decoder_init_seed", None)  # ensure DDP-consistent init when random
        # When re-initializing decoder weights for ablations, keep FLARE's pretrained
        # pose tokens by default; randomizing them is an explicit opt-in.
        decoder_reset_pose_tokens = bool(kwargs.pop("decoder_reset_pose_tokens", False))
        train_decoder = bool(kwargs.pop("train_decoder", False))
        self.train_decoder = train_decoder

        _ensure_flare_path()

        if flare_ckpt is None:
            flare_ckpt = str(_default_init_checkpoint_path())

        # Load pretrained FLARE AsymmetricMASt3R model.
        self.backbone = _load_flare_backbone(flare_ckpt, device=self.device, enable_curope2d=enable_curope2d)

        # Optional: reduce decoder depth for ablation studies
        if decoder_depth is not None:
            decoder_depth = int(decoder_depth)
            if 0 < decoder_depth < 12:
                def _subset_modulelist(module_list: nn.ModuleList, indices: list[int]) -> nn.ModuleList:
                    return nn.ModuleList([module_list[i] for i in indices])

                total_depth = len(self.backbone.dec_blocks)
                if total_depth < decoder_depth:
                    raise ValueError(
                        f"decoder_depth={decoder_depth} exceeds backbone decoder depth={total_depth}"
                    )

                indices: list[int] | None = None
                if decoder_layer_scheme is None:
                    indices = list(range(decoder_depth))
                    print(
                        f"[FlareStudent] Reducing decoder depth from {total_depth} to {decoder_depth} layers",
                        flush=True,
                    )
                else:
                    scheme = str(decoder_layer_scheme).lower()
                    if scheme == "skipmid":
                        if decoder_depth != 6:
                            raise ValueError("decoder_layer_scheme='skipmid' requires decoder_depth=6")
                        if total_depth < 6:
                            raise ValueError(
                                f"decoder_layer_scheme='skipmid' requires backbone depth>=6, got {total_depth}"
                            )
                        indices = [0, 1, 2, total_depth - 3, total_depth - 2, total_depth - 1]
                        print(
                            f"[FlareStudent] Reducing decoder layers via scheme='skipmid': indices={indices} "
                            f"(from {total_depth})",
                            flush=True,
                        )
                    elif scheme in {"indices", "index", "custom"}:
                        if decoder_layer_indices is None:
                            raise ValueError(
                                "decoder_layer_scheme='indices' requires model.params.decoder_layer_indices "
                                "(e.g. [3,7,12] for 1-based or [2,6,11] for 0-based)."
                            )
                        if isinstance(decoder_layer_indices, str):
                            raw = decoder_layer_indices.strip()
                            if raw.startswith("[") and raw.endswith("]"):
                                raw = raw[1:-1]
                            parts = [p.strip() for p in raw.split(",") if p.strip()]
                            parsed = [int(p) for p in parts]
                        else:
                            parsed = [int(x) for x in decoder_layer_indices]

                        if not parsed:
                            raise ValueError("decoder_layer_indices must be a non-empty list of ints")
                        if len(set(parsed)) != len(parsed):
                            raise ValueError(f"decoder_layer_indices must be unique, got: {parsed}")

                        # Support both 1-based ([1..total_depth]) and 0-based ([0..total_depth-1]) inputs.
                        if all(1 <= x <= total_depth for x in parsed):
                            indices = [x - 1 for x in parsed]
                            scheme_note = "1-based"
                        elif all(0 <= x < total_depth for x in parsed):
                            indices = parsed
                            scheme_note = "0-based"
                        else:
                            raise ValueError(
                                f"decoder_layer_indices out of range for backbone depth={total_depth}: {parsed}"
                            )

                        if decoder_depth != len(indices):
                            raise ValueError(
                                f"decoder_depth={decoder_depth} must equal len(decoder_layer_indices)={len(indices)}"
                            )
                        print(
                            f"[FlareStudent] Reducing decoder layers via scheme='indices' ({scheme_note}): "
                            f"indices={indices} (from {total_depth})",
                            flush=True,
                        )
                    elif scheme in {"odd", "odd_layers", "odd_indices"}:
                        if decoder_depth != 6:
                            raise ValueError("decoder_layer_scheme='odd' requires decoder_depth=6")
                        odd_indices = [i for i in range(total_depth) if i % 2 == 1]
                        if len(odd_indices) < decoder_depth:
                            raise ValueError(
                                f"decoder_layer_scheme='odd' requires at least {decoder_depth} odd layers, "
                                f"but backbone depth={total_depth} provides only {len(odd_indices)}"
                            )
                        indices = odd_indices[:decoder_depth]
                        print(
                            f"[FlareStudent] Reducing decoder layers via scheme='odd': indices={indices} "
                            f"(from {total_depth})",
                            flush=True,
                        )
                    else:
                        raise ValueError(f"Unsupported decoder_layer_scheme: {decoder_layer_scheme!r}")

                if indices is None:
                    raise AssertionError("decoder layer indices not set")

                # Keep all decoder-related ModuleList aligned to the same selected layers.
                modulelist_attrs = (
                    "dec_blocks",
                    "dec_blocks2",
                    "dec_blocks_fine",
                    "dec_blocks_point_cross",
                    "cam_cond_encoder",
                    "cam_cond_embed",
                    # MASt3R/FLARE extra decoder variants (may or may not exist depending on model)
                    "dec_blocks_point",
                    "cam_cond_encoder_fine",
                    "cam_cond_embed_fine",
                    "cam_cond_encoder_point",
                    "cam_cond_embed_point",
                    "cam_cond_embed_point_pre",
                    "cam_cond_encoder_point_fine",
                    "cam_cond_embed_point_fine",
                )
                for attr in modulelist_attrs:
                    if not hasattr(self.backbone, attr):
                        continue
                    value = getattr(self.backbone, attr)
                    if isinstance(value, nn.ModuleList) and len(value) == total_depth:
                        setattr(self.backbone, attr, _subset_modulelist(value, indices))

                self.backbone.dec_depth = len(indices)

        # Optional: re-initialize decoder weights for ablations while keeping decoder_embed/dec_cam_norm pretrained.
        if decoder_weight_init is not None:
            init_kind = str(decoder_weight_init).lower()
            if init_kind in {"random", "trunc_normal", "trunc", "truncnormal"}:
                def _maybe_reset_parameters(module: nn.Module) -> None:
                    reset = getattr(module, "reset_parameters", None)
                    if callable(reset):
                        reset()

                devices: list[int] = []
                if self.device.type == "cuda" and self.device.index is not None:
                    devices = [int(self.device.index)]

                with torch.random.fork_rng(devices=devices, enabled=True):
                    if decoder_init_seed is not None:
                        seed_val = int(decoder_init_seed)
                        torch.manual_seed(seed_val)
                        if self.device.type == "cuda":
                            torch.cuda.manual_seed_all(seed_val)

                    if init_kind == "random":
                        print(
                            "[FlareStudent] Random-initializing decoder blocks for ablation "
                            "(keeps decoder_embed/dec_cam_norm pretrained)",
                            flush=True,
                        )
                        init_fn = _maybe_reset_parameters
                        init_tokens = lambda t: torch.nn.init.normal_(t, mean=0.0, std=1.0)
                    else:
                        print(
                            "[FlareStudent] Trunc-normal initializing decoder blocks for ablation "
                            "(keeps decoder_embed/dec_cam_norm pretrained)",
                            flush=True,
                        )
                        init_fn = _init_module_trunc_normal
                        init_tokens = lambda t: _trunc_normal_(t, std=0.02)

                    for attr in ("dec_blocks", "dec_blocks2", "cam_cond_encoder", "cam_cond_embed"):
                        if not hasattr(self.backbone, attr):
                            continue
                        value = getattr(self.backbone, attr)
                        if isinstance(value, nn.ModuleList):
                            for block in value:
                                block.apply(init_fn)

                    if decoder_reset_pose_tokens:
                        for attr in ("pose_token_ref", "pose_token_source"):
                            if not hasattr(self.backbone, attr):
                                continue
                            param = getattr(self.backbone, attr)
                            if isinstance(param, torch.nn.Parameter):
                                init_tokens(param)

        # Freeze FLARE backbone weights by default; we selectively unfreeze
        # the pose-head trunk if requested.
        for param in self.backbone.parameters():
            param.requires_grad = False

        if train_decoder:
            print("[FlareStudent] Unfreezing decoder blocks for ablation", flush=True)
            for attr in ("dec_blocks", "dec_blocks2", "cam_cond_encoder", "cam_cond_embed"):
                if not hasattr(self.backbone, attr):
                    continue
                value = getattr(self.backbone, attr)
                if isinstance(value, nn.ModuleList):
                    # NOTE: In FLARE's `_decoder`, `cam_cond_embed[i]` updates features that feed the *next*
                    # decoder layer. The last layer's cam_cond_embed does not affect the returned cam_tokens,
                    # so it receives no gradient; keep it frozen to avoid DDP unused-parameter errors.
                    if attr == "cam_cond_embed" and len(value) > 0:
                        for layer_idx, layer in enumerate(value):
                            if layer_idx >= len(value) - 1:
                                continue
                            for param in layer.parameters():
                                param.requires_grad = True
                    else:
                        for param in value.parameters():
                            param.requires_grad = True

            for attr in ("pose_token_ref", "pose_token_source"):
                if not hasattr(self.backbone, attr):
                    continue
                param = getattr(self.backbone, attr)
                if isinstance(param, torch.nn.Parameter):
                    param.requires_grad = True

        # Shortcuts to the pieces we actually use.
        self._encode_symmetrized = self.backbone._encode_symmetrized
        self._decoder = self.backbone._decoder
        self.trunk = self.backbone.pose_head.trunk

        if train_trunk:
            for param in self.trunk.parameters():
                param.requires_grad = True

        # Hidden dimension for tokens (decoder / trunk dim).
        hidden_dim = self.trunk[0].norm1.normalized_shape[0]

        # Overlap head selection
        if head_type == 1:
            self.head = SimpleOverlapHead(hidden_dim, overlap_dim)
        elif head_type == 2:
            self.head = IterativeOverlapHead(hidden_dim, overlap_dim, overlap_iters)
        else:
            raise ValueError(f"Unsupported overlap_head_type: {head_type}")

        self.to(self.device)

    def _decode_pair_to_tokens(
        self,
        feat1: torch.Tensor,
        pos1: torch.Tensor,
        feat2: torch.Tensor,
        pos2: torch.Tensor,
    ) -> torch.Tensor:
        # FLARE's decoder uses `torch.utils.checkpoint.checkpoint` internally. In re-entrant mode,
        # if all checkpoint inputs have `requires_grad=False`, the decoder block parameters can end
        # up receiving no gradients even when unfrozen, which breaks DDP (unused-parameter error).
        #
        # For the "train_decoder" ablation, we keep the encoder/backbone frozen; make the decoder
        # inputs gradient-enabled (without backpropagating into the frozen encoder) so decoder params
        # get proper gradients.
        if getattr(self, "train_decoder", False) and self.training:
            feat1 = feat1.detach().requires_grad_(True)
            feat2 = feat2.detach().requires_grad_(True)

        pose_tokens1, pose_tokens2 = self._decoder(feat1, pos1, feat2, pos2)

        last_ref = pose_tokens1[-1]  # [B, 1, C]
        last_cur = pose_tokens2[-1]  # [B*V, 1, C] when V>1

        B = int(last_ref.shape[0])
        views = int(feat2.shape[1])
        if views <= 0:
            raise ValueError(f"Expected feat2 with views>=1, got shape={tuple(feat2.shape)}")
        if last_cur.shape[0] != B * views:
            raise RuntimeError(
                f"Decoder output has unexpected batch size: got {last_cur.shape[0]}, expected {B * views} "
                f"(B={B}, views={views})"
            )

        # Reshape source tokens to [B, V, 1, C], broadcast ref tokens to [B, V, 1, C].
        last_cur = last_cur.reshape(B, views, last_cur.shape[-2], last_cur.shape[-1])
        last_ref = last_ref.unsqueeze(1).expand(-1, views, -1, -1)

        # Pair tokens: [B, V, 2, C] -> flatten to [B*V, 2, C].
        view_tokens = torch.cat([last_ref, last_cur], dim=2)
        return view_tokens.reshape(B * views, 2, view_tokens.shape[-1])

    def create_frame(self, image: torch.Tensor) -> FlareFrame:
        """Create a FlareFrame for caching encoder features.

        Args:
            image: RGB tensor of shape [3,H,W] or [B,3,H,W].
        """
        img = image
        if img.dim() == 3:
            img = img.unsqueeze(0)
        elif img.dim() != 4:
            raise ValueError(
                "Expected image tensor of shape [3,H,W] or [B,3,H,W], "
                f"got shape={tuple(img.shape)}"
            )

        img = img.to(self.device)
        B, _, H, W = img.shape
        true_shape = (
            torch.tensor([H, W], device=img.device, dtype=torch.int64)
            .unsqueeze(0)
            .expand(B, -1)
        )
        return FlareFrame(img=img, true_shape=true_shape)

    @torch.inference_mode()
    def encode_frame(self, frame: FlareFrame) -> None:
        """Populate `frame.feat` / `frame.pos` if missing (lazy encode)."""
        if frame.feat is not None and frame.pos is not None:
            return

        if frame.img.device != self.device:
            frame.img = frame.img.to(self.device)
        if frame.true_shape.device != self.device:
            frame.true_shape = frame.true_shape.to(self.device)

        view = [{"img": frame.img, "true_shape": frame.true_shape}]
        _, feat, pos, _, _, _, _ = self._encode_symmetrized(view)
        frame.feat = feat
        frame.pos = pos

    @torch.inference_mode()
    def forward_frames(self, ref_frame: FlareFrame, cur_frame: FlareFrame) -> StudentPred:
        """Forward pass using cached frames (reference can be reused across calls)."""
        self.encode_frame(ref_frame)
        self.encode_frame(cur_frame)
        if ref_frame.feat is None or ref_frame.pos is None:
            raise RuntimeError("ref_frame was not encoded")
        if cur_frame.feat is None or cur_frame.pos is None:
            raise RuntimeError("cur_frame was not encoded")

        view_tokens = self._decode_pair_to_tokens(
            ref_frame.feat,
            ref_frame.pos,
            cur_frame.feat,
            cur_frame.pos,
        )
        score = self.head(view_tokens, self.trunk)
        return StudentPred(
            overlap_score=score,
            pose_ij=None,
            extras={},
        )

    @torch.inference_mode()
    def forward_cached_ref(self, ref_frame: FlareFrame, cur_image: torch.Tensor) -> StudentPred:
        """Convenience wrapper: create + encode current frame, reuse cached ref."""
        cur_frame = self.create_frame(cur_image)
        return self.forward_frames(ref_frame, cur_frame)

    @torch.inference_mode()
    def forward_cached_ref_many(self, ref_frame: FlareFrame, cur_images: torch.Tensor) -> StudentPred:
        """One-to-many inference with cached ref; scores each candidate independently.

        This keeps batch throughput while matching per-frame logic:
        each candidate is decoded as an independent (ref, cur) pair (views=1),
        avoiding cross-candidate coupling inside the FLARE decoder.
        Returns overlap scores with shape [B, K].
        """
        self.encode_frame(ref_frame)
        if ref_frame.feat is None or ref_frame.pos is None:
            raise RuntimeError("ref_frame was not encoded")

        if cur_images.dim() == 4:
            cur_images = cur_images.unsqueeze(1)
        elif cur_images.dim() != 5:
            raise ValueError(f"Expected cur_images [B,3,H,W] or [B,K,3,H,W], got {tuple(cur_images.shape)}")

        B, K, _, H_cur, W_cur = cur_images.shape
        if ref_frame.feat.shape[0] != B:
            raise ValueError(
                f"Mismatched batch sizes: ref_frame B={ref_frame.feat.shape[0]} vs cur_images B={B}"
            )

        true_shape_cur = (
            torch.tensor([H_cur, W_cur], device=cur_images.device, dtype=torch.int64)
            .unsqueeze(0)
            .expand(B, -1)
        )
        views_cur = [{"img": cur_images[:, i], "true_shape": true_shape_cur} for i in range(K)]
        _, feat2, pos2, _, _, _, _ = self._encode_symmetrized(views_cur)  # [B,K,P,C], [B,K,P,2]

        ref_feat = ref_frame.feat
        ref_pos = ref_frame.pos
        if ref_feat is None or ref_pos is None:
            raise RuntimeError("ref_frame features are missing")

        # Convert [B,K,...] into independent batch items [B*K,1,...] so each
        # candidate uses views=1 in the decoder (no cross-candidate interaction).
        ref_feat_rep = ref_feat.expand(B, K, ref_feat.shape[-2], ref_feat.shape[-1]).reshape(
            B * K, 1, ref_feat.shape[-2], ref_feat.shape[-1]
        )
        ref_pos_rep = ref_pos.expand(B, K, ref_pos.shape[-2], ref_pos.shape[-1]).reshape(
            B * K, 1, ref_pos.shape[-2], ref_pos.shape[-1]
        )
        cur_feat_flat = feat2.reshape(B * K, 1, feat2.shape[-2], feat2.shape[-1])
        cur_pos_flat = pos2.reshape(B * K, 1, pos2.shape[-2], pos2.shape[-1])

        view_tokens = self._decode_pair_to_tokens(ref_feat_rep, ref_pos_rep, cur_feat_flat, cur_pos_flat)
        score = self.head(view_tokens, self.trunk).view(B, K)
        return StudentPred(
            overlap_score=score,
            pose_ij=None,
            extras={},
        )

    def _encode_pair_to_tokens(self, ref: torch.Tensor, cur: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Encode ref + cur(s) into trunk-ready tokens.

        Supports:
          - ref: [B,3,H,W], cur: [B,3,H,W]  -> K=1
          - ref: [B,3,H,W], cur: [B,K,3,H,W] -> K
        Returns:
          view_tokens: [B*K, 2, C]
          K: number of cur views
        """
        if ref.dim() != 4:
            raise ValueError(f"Expected ref shape [B,3,H,W], got {tuple(ref.shape)}")
        if cur.dim() == 4:
            cur = cur.unsqueeze(1)
        elif cur.dim() != 5:
            raise ValueError(f"Expected cur shape [B,3,H,W] or [B,K,3,H,W], got {tuple(cur.shape)}")

        B, K, _, H_cur, W_cur = cur.shape
        if ref.shape[0] != B:
            raise ValueError(f"Mismatched batch sizes: ref B={ref.shape[0]} vs cur B={B}")
        _, _, H_ref, W_ref = ref.shape

        true_shape_ref = (
            torch.tensor([H_ref, W_ref], device=ref.device, dtype=torch.int64)
            .unsqueeze(0)
            .expand(B, -1)
        )
        true_shape_cur = (
            torch.tensor([H_cur, W_cur], device=cur.device, dtype=torch.int64)
            .unsqueeze(0)
            .expand(B, -1)
        )

        views = [{"img": ref, "true_shape": true_shape_ref}]
        views.extend({"img": cur[:, i], "true_shape": true_shape_cur} for i in range(K))

        _, feat, pos, _, _, _, _ = self._encode_symmetrized(views)

        feat1 = feat[:, :1]
        feat2 = feat[:, 1:]
        pos1 = pos[:, :1]
        pos2 = pos[:, 1:]
        view_tokens = self._decode_pair_to_tokens(feat1, pos1, feat2, pos2)
        return view_tokens, K

    def forward(self, batch: StudentBatch) -> StudentPred:  # type: ignore[override]
        ref = batch.ref_image.to(self.device)
        cur = batch.cur_image.to(self.device)
        view_tokens, K = self._encode_pair_to_tokens(ref, cur)
        score = self.head(view_tokens, self.trunk)
        score = score.view(ref.shape[0], K)

        return StudentPred(
            overlap_score=score,
            pose_ij=None,
            extras={},
        )
