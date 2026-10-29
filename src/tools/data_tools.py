"""Data Tools - Operații pe DataFrames."""
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

from .registry import register_tool


class JoinParams(BaseModel):
    left_key: str = Field(description="Coloana cheie din primul DataFrame")
    right_key: str = Field(description="Coloana cheie din al doilea DataFrame")
    how: Literal["inner", "left", "right", "outer"] = Field(
        default="inner", description="Tipul de join"
    )


class FilterParams(BaseModel):
    column: str = Field(description="Coloana pe care se aplică filtrul")
    operator: Literal["==", "!=", ">", "<", ">=", "<=", "contains"] = Field(
        description="Operatorul de comparație"
    )
    value: str = Field(description="Valoarea pentru comparație")


@register_tool
def join_data(inputs: list[pd.DataFrame], params: JoinParams) -> pd.DataFrame:
    """Combină două DataFrames pe baza unei chei comune."""
    if len(inputs) < 2:
        raise ValueError("join_data requires 2 DataFrames")
    return pd.merge(
        inputs[0], inputs[1],
        left_on=params.left_key,
        right_on=params.right_key,
        how=params.how,
    )


@register_tool
def filter_data(inputs: list[pd.DataFrame], params: FilterParams) -> pd.DataFrame:
    """Filtrează un DataFrame pe baza unei condiții."""
    if not inputs:
        raise ValueError("filter_data requires 1 DataFrame")

    df = inputs[0]
    col = df[params.column]
    value = params.value

    if params.operator in (">", "<", ">=", "<="):
        try:
            value = float(value)
        except ValueError:
            pass

    ops = {
        "==": lambda c, v: c == v,
        "!=": lambda c, v: c != v,
        ">": lambda c, v: c > v,
        "<": lambda c, v: c < v,
        ">=": lambda c, v: c >= v,
        "<=": lambda c, v: c <= v,
        "contains": lambda c, v: c.astype(str).str.contains(str(v), case=False, na=False),
    }

    return df[ops[params.operator](col, value)]
