from rich.console import Console
from rich.table import Table


def print_metric_table(
    rows: list[tuple[tuple[str, str, str], tuple[str, str, str] | None]],
) -> None:
    """
    Print a table of metric values.

    Args:
        title (str): The title of the table.
        rows (list[tuple[tuple[str, str, str], tuple[str, str, str] | None]]): The rows of the table. Two rows per
            metric: one for the main metric and one for the example metric (if any). Each triple contains the name,
            main value, and other values as a single comma-separated string.

    Prints:
        A table of metric values.
    """
    table = Table(title="Registered metrics")
    table.add_column("Name", justify="left", style="bright_cyan", no_wrap=True)
    table.add_column("Level", justify="left", style="bright_black", no_wrap=True)
    table.add_column("Main", style="bright_magenta")
    table.add_column("Other", justify="left", style="bright_black")

    for metric_name, (main_row, example_row) in rows:
        end_section = True if example_row is None else False
        table.add_row(metric_name, "dataset", *main_row, end_section=end_section)
        if example_row is not None:
            table.add_row("", "example", *example_row, end_section=True)

    console = Console()
    console.print(table)
