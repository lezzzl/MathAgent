"""normalize_select: формы benchmarks.select, переживающие `kedro run --params`."""

import click
import pytest
from kedro.framework.cli.utils import _split_params

from mathagent.pipelines.benchmarks.nodes import normalize_select

EXPECTED = ["aime26", "hmmt26", "imo_answerbench"]


def split_params(value: str) -> dict:
    """Прогоняет строку через тот же колбэк, что и CLI `kedro run --params`."""
    option = click.Option(["--params"])
    option.name = "params"
    return _split_params(click.Context(click.Command("run")), option, value)


def test_yaml_list_passes_through():
    assert normalize_select(list(EXPECTED)) == EXPECTED


@pytest.mark.parametrize(
    "value",
    ["[aime26;hmmt26;imo_answerbench]", "aime26;hmmt26;imo_answerbench",
     "[aime26|hmmt26|imo_answerbench]", "[aime26; hmmt26 ;imo_answerbench]"],
)
def test_delimited_string_forms(value):
    assert normalize_select(value) == EXPECTED


@pytest.mark.parametrize(
    "value",
    [
        "benchmarks.select=aime26;hmmt26;imo_answerbench",
        # OmegaConf разберёт скобки как список из одного склеенного элемента —
        # normalize_select режет разделители и внутри элементов
        "benchmarks.select=[aime26;hmmt26;imo_answerbench]",
    ],
)
def test_semicolon_form_survives_the_cli_splitter(value):
    params = split_params(value)
    assert normalize_select(params["benchmarks"]["select"]) == EXPECTED


def test_indexed_form_survives_the_cli_splitter():
    params = split_params(
        "benchmarks.select.0=aime26,benchmarks.select.1=hmmt26,"
        "benchmarks.select.2=imo_answerbench"
    )
    assert normalize_select(params["benchmarks"]["select"]) == EXPECTED


def test_indexed_form_keeps_order_beyond_nine():
    # сортировка по строкам поставила бы "10" перед "2"
    value = {str(i): f"bench{i}" for i in range(11)}
    assert normalize_select(value)[-1] == "bench10"


def test_comma_form_is_still_broken_by_the_cli():
    # первопричина исходной ошибки: запятая в скобках CLI не переживает
    with pytest.raises(click.UsageError, match="must contain a key and a value"):
        split_params("benchmarks.select=[aime26,hmmt26]")


def test_empty_select_rejected():
    with pytest.raises(ValueError, match="пуст"):
        normalize_select([])


def test_non_index_dict_rejected():
    with pytest.raises(ValueError, match="неиндексными"):
        normalize_select({"first": "aime26"})
