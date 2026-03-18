# real_experiments

This directory stores assets for real execution experiments.

Contents:

- `generate_real_experiment_assets.py`: regenerates network JSON files and Mermaid diagrams.
- `networks/`: 4 prepared network definitions for real execution planning.
- `diagrams/`: Mermaid source files describing each network.
- `results/`: templates for storing real execution outputs.

Important:

- These assets are for real-experiment preparation only.
- No Docker access, Metasploit RPC call, exploit execution, payload delivery,
  shell acquisition, service fingerprinting, or lateral movement has been
  performed by generating these files.
- Actual results should be appended to `results/real_experiments_results.csv`
  and `results/real_experiments_execution_log.json` only after live execution.
