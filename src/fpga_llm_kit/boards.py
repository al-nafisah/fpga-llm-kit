"""FPGA boards, described by what a plan needs: DRAM, its bandwidth, on-chip RAM and DSPs.

Each board is a TOML file in boards/, every number with its source. Keys the
plan does not use, such as luts, are there for the reader. For a board that is
not there, copy the closest file, edit it, and pass its path to --board.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

MIB = 1024 * 1024
KBIT = 1024 // 8  # datasheets count on-chip RAM in Kb of 1024 bits


@dataclass(frozen=True)
class Board:
    key: str
    name: str
    part: str
    dsps: int
    bram_kbits: int
    uram_kbits: int
    dram_type: str
    dram_mib: int
    dram_bus_bits: int
    dram_mts: int        # transfers per second, millions
    reserved_mib: int    # DRAM kept by the operating system, if the board runs one

    @property
    def on_chip_bytes(self) -> int:
        return (self.bram_kbits + self.uram_kbits) * KBIT

    @property
    def dram_bytes_free(self) -> int:
        return (self.dram_mib - self.reserved_mib) * MIB

    @property
    def dram_peak_gbps(self) -> float:
        """Theoretical peak in GB/s: bus width in bytes times transfer rate."""
        return self.dram_bus_bits / 8 * self.dram_mts / 1000


def _parse(key: str, text: str) -> Board:
    t = tomllib.loads(text)
    fabric, dram = t["fabric"], t["dram"]
    return Board(
        key=key,
        name=t["name"],
        part=t["part"],
        dsps=fabric["dsps"],
        bram_kbits=fabric["bram_kbits"],
        uram_kbits=fabric.get("uram_kbits", 0),
        dram_type=dram["type"],
        dram_mib=dram["mib"],
        dram_bus_bits=dram["bus_bits"],
        dram_mts=dram["mts"],
        reserved_mib=dram.get("reserved_mib", 0),
    )


def _bundled() -> dict[str, str]:
    folder = resources.files(__package__) / "boards"
    return {f.name.removesuffix(".toml"): f.read_text() for f in folder.iterdir()
            if f.name.endswith(".toml")}


def all_boards() -> list[Board]:
    return [_parse(key, text) for key, text in sorted(_bundled().items())]


def board(name: str) -> Board:
    """A bundled board by key ("kv260"), or a TOML file by path."""
    path = Path(name)
    if path.suffix == ".toml" and path.is_file():
        return _parse(path.stem, path.read_text())
    bundled = _bundled()
    if name not in bundled:
        raise KeyError(f"no board {name!r}: choose one of {', '.join(sorted(bundled))}, "
                       "or pass the path to a .toml file")
    return _parse(name, bundled[name])
