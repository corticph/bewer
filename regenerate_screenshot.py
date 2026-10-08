"""Regenerate the alignment-display.png screenshot for the README.

Run with a wide terminal (or COLUMNS=120) and capture the output as a PNG.
Example with tmux + capture:
    poetry run python regenerate_screenshot.py
"""

from bewer import Dataset


def main():
    dataset = Dataset(language="en")
    dataset.add(
        ref="an example with different types of errors",
        hyp="an odd example with diff types errors",
    )
    example = dataset[0]
    alignment = example.metrics.levenshtein().alignment
    print("\n\n")
    alignment.display()


if __name__ == "__main__":
    main()
