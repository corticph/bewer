from rich.console import Console
from rich.table import Table

__all__: list[str] = []  # Internal


def print_metric_table(
    rows: list[tuple[str, str, tuple[tuple[str, str, str], tuple[str, str, str] | None]]],
) -> None:
    """
    Print a table of registered metrics, their parameters and their values.

    Args:
        rows: The rows of the table, as (name, params, (main_row, example_row)). Two rows are
            printed per metric: one for the dataset-level metric and one for the example-level
            metric (if any). ``params`` is a comma-separated parameter summary shown once per
            metric, and each value row contains the main value and other values as strings.

    Prints:
        A table of registered metrics.
    """
    table = Table(title="Registered metrics", caption="* = required parameter")
    table.add_column("Name", justify="left", style="bright_cyan")
    table.add_column("Level", justify="left", style="bright_black", no_wrap=True)
    table.add_column("Main", style="bright_magenta")
    table.add_column("Other", justify="left", style="bright_black")
    table.add_column("Params", justify="left", style="bright_black")

    for metric_name, params, (main_row, example_row) in rows:
        end_section = True if example_row is None else False
        table.add_row(metric_name, "dataset", *main_row, params, end_section=end_section)
        if example_row is not None:
            table.add_row("", "example", *example_row, "", end_section=True)

    console = Console()
    console.print(table)
