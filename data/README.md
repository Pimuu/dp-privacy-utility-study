# Data

`raw/adult.csv` is the UCI **Adult / Census Income** dataset (Becker & Kohavi, 1996,
https://doi.org/10.24432/C5XW20, licensed CC BY 4.0): the original `adult.data`
(32,561 rows) and `adult.test` (16,281 rows) concatenated into 48,842 rows,
14 attributes + income label, no header, missing values written as `?`.

It was obtained from the public mirror
`https://raw.githubusercontent.com/jbrownlee/Datasets/master/adult-all.csv`
(the UCI host was unreachable from the build environment) and is committed here so
the study reproduces offline. SHA-256:
`21e0ea2f925a00338929a8c86c27354c72ac1c79819bcca81c7d91c3d64218c2`
(checked at load time by `src/data.py`; a mismatch triggers a warning).

If the file is deleted, `src/data.py` re-downloads it; if no download is possible,
the pipeline falls back to a synthetic bounded tabular dataset (`make_synthetic`).
