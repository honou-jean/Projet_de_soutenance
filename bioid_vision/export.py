"""Excel export for the results shown in the application's log panel."""

import pandas as pd


def export_lines_to_excel(lines, filepath, column_name="Informations"):
    """Write a list of text lines to a single-column .xlsx file."""
    df = pd.DataFrame(lines, columns=[column_name])
    df.to_excel(filepath, index=False)
