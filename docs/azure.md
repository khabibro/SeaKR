# Azure Workflow

Use this workflow for full SeaKR reproduction runs:

1. Develop locally on Mac or Linux.
2. Commit and push code, configs, scripts, and documentation to GitHub.
3. Clone the repository on the Azure GPU machine.
4. Create the Python/CUDA environment on Azure.
5. Authenticate with Hugging Face on Azure if gated models are needed.
6. Download datasets directly on Azure into `$DATA_DIR`.
7. Download model checkpoints directly on Azure into `$MODEL_DIR`.
8. Run mini experiments first.
9. Run full experiments after mini checks pass.
10. Download only final outputs, tables, figures, and selected logs back to the Mac.

Do not upload datasets, model weights, or full run logs to GitHub or Obsidian.
