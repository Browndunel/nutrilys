"""Formatage des nombres à la française pour les titres et les tableaux."""


def fmt_int(n: float) -> str:
    """1171729 -> '1 171 729'."""
    return f"{round(n):,}".replace(",", " ")


def fmt_pct(x: float, decimals: int = 0) -> str:
    """40.24 -> '40 %' ; avec decimals=1 -> '40,2 %'."""
    return f"{x:.{decimals}f}".replace(".", ",") + " %"


def fmt_num(x: float, decimals: int = 1) -> str:
    """2.35 -> '2,4'."""
    return f"{x:.{decimals}f}".replace(".", ",")
