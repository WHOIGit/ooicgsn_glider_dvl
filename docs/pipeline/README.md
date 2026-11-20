# Processing pipeline diagrams

Flowcharts of the DVL processing pipeline: an overview of the data flow between the
notebooks, and one flowchart for each of notebooks 1, 2, and 4, with their decisions.

| File | What it is |
|---|---|
| `dvl-pipeline.drawio` | The diagrams, one page each, to edit by hand in draw.io |
| `build.py` | Builds `dvl-pipeline.html` from the node lists `NB1`, `NB2`, `NB4` and `overview()` |
| `dvl-pipeline.html` | The diagrams as a web page, with the legend, captions, and open questions |
| `export_drawio.py` | Wrote `dvl-pipeline.drawio` from `build.py` |

## Editing in draw.io

Open `dvl-pipeline.drawio` with the **Draw.io Integration** extension in VS Code
(`hediet.vscode-drawio`), the draw.io desktop app, or <https://app.diagrams.net>. The pages are
tabs at the bottom. The arrows are attached to the boxes, so they follow when a box is moved.
Notes beside the boxes (the numbers from GL379 deployment 8) are separate text boxes.

Once `dvl-pipeline.drawio` has been edited by hand, it is the source of those diagrams:
`export_drawio.py` refuses to overwrite it unless it is run with `--force`.

## Rebuilding the web page

```
cd docs/pipeline
uv run python build.py
```

Each flowchart is a list of steps drawn top to bottom: `proc` (a processing step with its
function), `dec` (a decision, with the branch to its right), `data` (files read or written), and
`group` (a section heading). Each string in `title`, `detail`, or `notes` is one line.

The page and the draw.io file are separate: a change made in one is not copied to the other.
