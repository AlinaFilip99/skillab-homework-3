"""
Tool implementations — funcțiile efective ale tool-urilor.

Convenție: toate tools primesc params cu `input_dfs` (lista de DataFrames) + parametri specifici.
"""
import pandas as pd

from .registry import register_tool
from .params import JoinDataParams, FilterDataParams

VALID_JOINS = {"inner", "left", "right", "outer"}


@register_tool
def join_data(params: JoinDataParams) -> pd.DataFrame:
    """
    Combină două DataFrames pe baza unei chei comune (join).
    Suportă inner, left, right, outer join.

    Args:
        params.input_dfs: [left_df, right_df]
        params.left_key: coloana cheie din primul DataFrame
        params.right_key: coloana cheie din al doilea DataFrame
        params.how: tipul de join

    Returns:
        DataFrame rezultat după join
    """
    if len(params.input_dfs) != 2:
        raise ValueError(
            f"join_data necesită exact 2 DataFrames, primite: {len(params.input_dfs)}"
        )

    left_df, right_df = params.input_dfs

    if params.left_key not in left_df.columns:
        raise ValueError(f"Coloana '{params.left_key}' nu există în primul DataFrame")
    if params.right_key not in right_df.columns:
        raise ValueError(f"Coloana '{params.right_key}' nu există în al doilea DataFrame")
    if params.how not in VALID_JOINS:
        raise ValueError(
            f"Tip de join invalid: '{params.how}'. Valori permise: {sorted(VALID_JOINS)}"
        )

    return pd.merge(
        left_df,
        right_df,
        left_on=params.left_key,
        right_on=params.right_key,
        how=params.how,
    )


@register_tool
def filter_data(params: FilterDataParams) -> pd.DataFrame:
    """
    Filtrează un DataFrame pe baza unei condiții.
    Suportă operatori: ==, !=, >, <, >=, <=, contains.

    Args:
        params.input_dfs: [df]
        params.column: coloana pe care se aplică filtrul
        params.operator: operatorul de comparație
        params.value: valoarea pentru comparație

    Returns:
        DataFrame filtrat
    """
    if len(params.input_dfs) != 1:
        raise ValueError(
            f"filter_data necesită exact 1 DataFrame, primite: {len(params.input_dfs)}"
        )

    df = params.input_dfs[0]

    if params.column not in df.columns:
        raise ValueError(f"Coloana '{params.column}' nu există în DataFrame")

    col = df[params.column]
    op = params.operator
    value = params.value

    if op == "contains":
        mask = col.astype(str).str.contains(str(value), case=False, na=False, regex=False)
    else:
        # Dacă valoarea vine ca string (ex. din LLM) dar coloana e numerică,
        # o convertim ca să nu comparăm "5" cu 5.
        if pd.api.types.is_numeric_dtype(col) and isinstance(value, str):
            value = pd.to_numeric(value)

        if op == "==":
            mask = col == value
        elif op == "!=":
            mask = col != value
        elif op == ">":
            mask = col > value
        elif op == "<":
            mask = col < value
        elif op == ">=":
            mask = col >= value
        elif op == "<=":
            mask = col <= value
        else:
            raise ValueError(f"Operator nesuportat: '{op}'")

    return df[mask]