import pytest

from fpga_llm_kit.boards import all_boards, board

TOML = """
name = "My board"
part = "XC7A35T"

[fabric]
luts = 20800
dsps = 90
bram_kbits = 1800

[dram]
type = "DDR3L"
mib = 256
bus_bits = 16
mts = 667
"""


def test_every_bundled_board_parses_with_plausible_numbers():
    boards = all_boards()
    assert boards
    for b in boards:
        assert b.dsps > 0 and b.on_chip_bytes > 0
        assert 0 < b.dram_peak_gbps < 1000
        assert 0 <= b.reserved_mib < b.dram_mib


def test_a_board_file_given_by_path(tmp_path):
    path = tmp_path / "mine.toml"
    path.write_text(TOML)
    b = board(str(path))
    assert (b.key, b.uram_kbits, b.reserved_mib) == ("mine", 0, 0)
    assert b.on_chip_bytes == 1800 * 1024 // 8
    assert b.dram_peak_gbps == pytest.approx(16 / 8 * 667 / 1000)


def test_an_unknown_board_names_the_choices():
    with pytest.raises(KeyError, match="kv260"):
        board("no-such-board")
