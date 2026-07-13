from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


END = b"\x01\x01END\x01\x01\n"
READY = b"\x01\x01READY\x01\x01\n"


def default_glm() -> Path:
    return Path(__file__).resolve().parents[1] / "c" / "glm"


def default_coli() -> Path:
    return Path(__file__).resolve().parents[1] / "c" / "coli"


def prompt_template(prompt: str, think: bool = False) -> str:
    think_block = "<think>" if think else "<think></think>"
    return f"[gMASK]<sop><|user|>{prompt}<|assistant|>{think_block}"


def _read_until(stream, sentinel: bytes) -> bytes | None:
    data = bytearray()
    while True:
        b = stream.read(1)
        if b == b"":
            return None
        data += b
        if data.endswith(sentinel):
            return bytes(data[:-len(sentinel)])


@dataclass
class ServeEngine:
    model: str
    glm: str
    layer: int
    collect_path: str | None = None
    direction_path: str | None = None
    scale: float = 1.0
    cap: int = 8
    ebits: int = 8
    dbits: int = 8
    ram_gb: int = 0
    ctx: int = 4096
    think: bool = False

    def __post_init__(self) -> None:
        env = dict(os.environ)
        env["SNAP"] = self.model
        env["SERVE"] = "1"
        env["NGEN"] = "1"
        env["TEMP"] = "0"
        env["KVSAVE"] = "0"
        env["DRAFT"] = "0"
        env["SPEC"] = "0"
        env["CTX"] = str(self.ctx)
        env["COLI_ALBATE_LAYER"] = str(self.layer)
        if self.ram_gb:
            env["RAM_GB"] = str(self.ram_gb)
        if self.collect_path:
            env["COLI_ALBATE_COLLECT"] = self.collect_path
        if self.direction_path:
            env["COLI_ALBATE_DIR"] = self.direction_path
            env["COLI_ALBATE_SCALE"] = str(self.scale)
        self.proc = subprocess.Popen(
            [self.glm, str(self.cap), str(self.ebits), str(self.dbits)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            bufsize=0,
        )
        if _read_until(self.proc.stdout, READY) is None:
            stderr = self.proc.stderr.read().decode("utf-8", "replace")
            raise RuntimeError(f"glm exited before READY\n{stderr}")
        self.proc.stdout.readline()

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                if self.proc.stdin:
                    self.proc.stdin.close()
            finally:
                self.proc.terminate()

    def query(self, prompt: str) -> None:
        if not self.proc.stdin or not self.proc.stdout:
            raise RuntimeError("serve process is not available")
        payload = prompt_template(prompt, think=self.think).encode("utf-8")
        header = f"\x02PROMPT {len(payload)} 1 0 0.9 0\n".encode("ascii")
        self.proc.stdin.write(header)
        self.proc.stdin.write(payload)
        self.proc.stdin.write(b"\n")
        self.proc.stdin.flush()
        if _read_until(self.proc.stdout, END) is None:
            stderr = self.proc.stderr.read().decode("utf-8", "replace")
            raise RuntimeError(f"glm exited while serving a prompt\n{stderr}")
        self.proc.stdout.readline()

