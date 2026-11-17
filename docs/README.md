Generating the docs
----------

Use [mkdocs](http://www.mkdocs.org/) structure to update the documentation. mkdocs is in the
`docs` dependency group, so run it with uv from this directory.

Build locally with:

    uv run --group docs mkdocs build

Serve locally with:

    uv run --group docs mkdocs serve

or `make docs` from the repository root.
