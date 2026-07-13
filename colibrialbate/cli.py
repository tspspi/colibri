from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import numpy as np

from .io import read_samples, write_direction, write_manifest
from .runtime import ServeEngine, default_coli, default_glm


def _load_prompts(path: str) -> list[str]:
    prompts = [line.strip() for line in Path(path).read_text().splitlines()]
    prompts = [line for line in prompts if line]
    if not prompts:
        raise ValueError(f"{path} contains no prompts")
    return prompts


def _collect(args: argparse.Namespace, prompts: list[str], output: Path) -> np.ndarray:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    engine = ServeEngine(
        model=args.model,
        glm=str(args.glm),
        layer=args.layer,
        collect_path=str(output),
        cap=args.cap,
        ebits=args.ebits,
        dbits=args.dbits,
        ram_gb=args.ram,
        ctx=args.ctx,
        think=args.think,
    )
    try:
        for prompt in prompts:
            engine.query(prompt)
    finally:
        engine.close()
    return read_samples(output)


def cmd_collect(args: argparse.Namespace) -> int:
    prompts = _load_prompts(args.prompts)
    samples = _collect(args, prompts, Path(args.output))
    print(f"collected {samples.shape[0]} samples of width {samples.shape[1]} into {args.output}")
    return 0


def _compute_direction(harmful: np.ndarray, harmless: np.ndarray) -> np.ndarray:
    direction = harmful.mean(axis=0) - harmless.mean(axis=0)
    norm = np.linalg.norm(direction)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("refusal direction has zero norm")
    return (direction / norm).astype(np.float32, copy=False)


def cmd_compute(args: argparse.Namespace) -> int:
    harmful = read_samples(args.harmful)
    harmless = read_samples(args.harmless)
    if harmful.shape[1] != harmless.shape[1]:
        raise ValueError("harmful and harmless sample widths differ")
    direction = _compute_direction(harmful, harmless)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_direction(out, direction)
    write_manifest(
        out.with_suffix(out.suffix + ".json"),
        {
            "direction_path": str(out),
            "layer": args.layer,
            "hidden_size": int(direction.shape[0]),
            "harmful_samples": int(harmful.shape[0]),
            "harmless_samples": int(harmless.shape[0]),
            "source": "colibrialbate",
        },
    )
    print(f"wrote direction to {out}")
    return 0


def cmd_fit(args: argparse.Namespace) -> int:
    outdir = Path(args.outdir)
    harmful_path = outdir / "harmful.samples.bin"
    harmless_path = outdir / "harmless.samples.bin"
    direction_path = outdir / "refusal_dir.bin"
    harmful = _collect(args, _load_prompts(args.harmful_prompts), harmful_path)
    harmless = _collect(args, _load_prompts(args.harmless_prompts), harmless_path)
    direction = _compute_direction(harmful, harmless)
    write_direction(direction_path, direction)
    write_manifest(
        outdir / "refusal_dir.json",
        {
            "direction_path": str(direction_path),
            "layer": args.layer,
            "hidden_size": int(direction.shape[0]),
            "harmful_samples": int(harmful.shape[0]),
            "harmless_samples": int(harmless.shape[0]),
            "harmful_path": str(harmful_path),
            "harmless_path": str(harmless_path),
            "source": "colibrialbate",
        },
    )
    print(f"wrote samples and direction into {outdir}")
    return 0


def _coli_env(direction: str, layer: int, scale: float) -> dict[str, str]:
    env = dict(os.environ)
    env["COLI_ALBATE_DIR"] = direction
    env["COLI_ALBATE_LAYER"] = str(layer)
    env["COLI_ALBATE_SCALE"] = str(scale)
    return env


def cmd_chat(args: argparse.Namespace) -> int:
    cmd = [str(args.coli), "chat", "--model", args.model]
    if args.ram:
        cmd += ["--ram", str(args.ram)]
    return subprocess.call(cmd, env=_coli_env(args.direction, args.layer, args.scale))


def cmd_run(args: argparse.Namespace) -> int:
    cmd = [str(args.coli), "run", "--model", args.model]
    if args.ram:
        cmd += ["--ram", str(args.ram)]
    cmd += list(args.prompt)
    return subprocess.call(cmd, env=_coli_env(args.direction, args.layer, args.scale))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="colibrialbate")
    sub = parser.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", required=True)
    common.add_argument("--layer", required=True, type=int)
    common.add_argument("--glm", type=Path, default=default_glm())
    common.add_argument("--cap", type=int, default=8)
    common.add_argument("--ebits", type=int, default=8)
    common.add_argument("--dbits", type=int, default=8)
    common.add_argument("--ram", type=int, default=0)
    common.add_argument("--ctx", type=int, default=4096)
    common.add_argument("--think", action="store_true")

    collect = sub.add_parser("collect", parents=[common])
    collect.add_argument("--prompts", required=True)
    collect.add_argument("--output", required=True)
    collect.set_defaults(func=cmd_collect)

    compute = sub.add_parser("compute")
    compute.add_argument("--harmful", required=True)
    compute.add_argument("--harmless", required=True)
    compute.add_argument("--output", required=True)
    compute.add_argument("--layer", required=True, type=int)
    compute.set_defaults(func=cmd_compute)

    fit = sub.add_parser("fit", parents=[common])
    fit.add_argument("--harmful-prompts", required=True)
    fit.add_argument("--harmless-prompts", required=True)
    fit.add_argument("--outdir", required=True)
    fit.set_defaults(func=cmd_fit)

    run_common = argparse.ArgumentParser(add_help=False)
    run_common.add_argument("--model", required=True)
    run_common.add_argument("--direction", required=True)
    run_common.add_argument("--layer", required=True, type=int)
    run_common.add_argument("--scale", type=float, default=1.0)
    run_common.add_argument("--coli", type=Path, default=default_coli())
    run_common.add_argument("--ram", type=int, default=0)

    chat = sub.add_parser("chat", parents=[run_common])
    chat.set_defaults(func=cmd_chat)

    run = sub.add_parser("run", parents=[run_common])
    run.add_argument("prompt", nargs=argparse.REMAINDER)
    run.set_defaults(func=cmd_run)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
