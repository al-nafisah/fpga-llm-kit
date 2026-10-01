"""The number format the hardware reads.

Weights. Each row of a matrix is cut into groups of GROUP inputs. A weight is a
signed integer code of 4 or 8 bits, every group has an unsigned 8-bit
sub-scale, and every row one scale kept as a 16-bit mantissa and an exponent:

    weight = code * sub_scale * row_mantissa * 2**row_exponent

Activations entering a matrix are int8 with one power-of-two scale per vector.
So the hardware sums code * activation within a group, multiplies that sum by
the group's sub-scale, adds up the groups, and applies the row scale once at
the end: integers throughout, and exact.

The KV cache is int8 with one power-of-two scale per head vector.

This module holds the sizes the planner needs; the quantizer that produces the
format is added with the reference model.
"""

GROUP = 32
SUBSCALE_BYTES = 1    # per group
ROW_SCALE_BYTES = 4   # per row: 16-bit mantissa and 8-bit exponent, padded to a word
NORM_BYTES = 2        # per norm weight
BIAS_BYTES = 4        # per bias
KV_BYTES = 1          # per cached key or value
KV_SCALE_BYTES = 1    # per cached head vector


def matrix_bytes(rows: int, cols: int, weight_bits: int) -> int:
    if cols % GROUP:
        raise ValueError(f"a matrix with {cols} inputs does not split into groups of {GROUP}")
    per_row = cols * weight_bits // 8 + cols // GROUP * SUBSCALE_BYTES + ROW_SCALE_BYTES
    return rows * per_row


def kv_bytes_per_token(layers: int, kv_heads: int, head_dim: int) -> int:
    """One key and one value vector per KV head per layer."""
    return layers * 2 * kv_heads * (head_dim * KV_BYTES + KV_SCALE_BYTES)
