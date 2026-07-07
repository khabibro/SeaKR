# Datasets

Datasets should be downloaded directly on the machine that will use them.

Do not store datasets in Obsidian. Obsidian should contain notes only, not large research artifacts.

Use:

- `data/raw/` for original downloaded archives and source files.
- `data/processed/` for extracted or transformed datasets and retrieval indexes.

For Azure runs, clone the repository first, then download datasets on Azure into `$DATA_DIR`. This avoids pushing large files through Git and avoids copying unnecessary assets through Obsidian.
