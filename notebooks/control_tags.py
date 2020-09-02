"""
Call with
cp timing-example.ipynb timing-example-copy.ipynb
python control_tags.py timing-example-copy.ipynb
"""

import nbformat as nbf
from fire import Fire

# Collect a list of all notebooks in the content folder
# from glob import glob
# notebooks = glob("./content/**/*.ipynb", recursive=True)

# Text to look for in adding tags
text_search_dict = {
    "# HIDDEN": "remove-cell",  # Remove the whole cell
    "# NO CODE": "remove-input",  # Remove only the input
    "# HIDE CODE": "hide-input",  # Hide the input w/ a button to show
}

text_search_dict = {
    "# HIDDEN": "hide_output",  # Remove the whole cell
    "# NO CODE": "hide_input",  # Remove only the input
    "# HIDE CODE": "collapse_hide",  # Hide the input w/ a button to show
}

valid_tags = (
    "hide_output hide_input collapse_hide collapse_show remove_cell".split()
)


def mod_cell(cell):
    lines = cell["source"].splitlines()
    if not lines:
        return cell
    fst_line = lines[0].strip()
    if not fst_line.startswith("#"):
        return cell
    tag = fst_line.lstrip("#").lstrip()
    if tag not in valid_tags:
        return cell

    if "metadata" not in cell:
        cell["metadata"] = {}
    cell["metadata"][tag] = True
    return cell


def mod_nb(ipath):
    """
    Search through each notebook and look for th text,
    add a tag if necessary
    """
    # for ipath in notebooks:
    print(ipath)
    ntbk = nbf.read(ipath, nbf.NO_CONVERT)
    for cell in ntbk.cells:

        cell = mod_cell(cell)
        # if "metadata" not in cell:
        #     cell["metadata"] = {}
        # cell_tags = cell["metadata"].get("tags", [])
        # for comment, cell_tag in text_search_dict.items():
        #     set_tag = comment in cell["source"]
        #     if set_tag:
        #         if cell_tag not in cell_tags:
        #             cell_tags.append(cell_tag)
        #         cell["metadata"][cell_tag] = True

        # if len(cell_tags) > 0:
        #     cell["metadata"]["tags"] = cell_tags

    nbf.write(ntbk, ipath)


if __name__ == "__main__":

    Fire(mod_nb)
