import inspect

from backend.tools import TOOL_DEFINITIONS, generate_chart


def test_generate_chart_tool_is_defined():
    names = {t["function"]["name"] for t in TOOL_DEFINITIONS}
    assert "generate_chart" in names


def test_generate_chart_schema_supports_grouped_bar():
    tool = next(t for t in TOOL_DEFINITIONS if t["function"]["name"] == "generate_chart")
    props = tool["function"]["parameters"]["properties"]
    assert "chart_type" in props
    assert set(props["chart_type"]["enum"]) >= {"line", "bar", "grouped_bar"}
    # grouped_bar accepts metrics array
    assert "metrics" in props


def test_generate_chart_signature_has_metrics_param():
    sig = inspect.signature(generate_chart)
    assert "metrics" in sig.parameters
