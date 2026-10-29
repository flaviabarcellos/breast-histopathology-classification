# Exploratory Notebooks

For project setup, architecture, dataset preparation, training, and evaluation, see the [main project README](../README.md).

Use this directory for exploratory data analysis and experiment notebooks. Reusable pipeline code lives in `src/breakhis/` and should be imported from the installed `breakhis` package.

## Start JupyterLab

Run from the repository root:

```bash
poetry install --all-groups
poetry run jupyter lab notebooks/
```

If Poetry is installed locally in this workspace, replace `poetry` with `.tools/poetry/bin/poetry`.

## Experiment guidelines

- Keep setup, dataset inspection, training, and evaluation in separate cells.
- Record configuration, random seed, magnification, class counts, and patient split membership.
- Use validation data to make experiment decisions and reserve test data for final evaluation.
- Import pipeline functions rather than maintaining separate copies of training logic.
- Clear outputs containing local paths or dataset examples before committing.

The original source notebook remains at its supplied location and is not included here. Opening a notebook does not automatically start a training job.
