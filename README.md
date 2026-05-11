# PWFEM: Probabilistic Wavelet Finite Element Method for Uncertainty-Aware Super-Resolution

This repository hosts the official implementation of **PWFEM**, a Probabilistic Wavelet Finite Element Method for uncertainty-aware super-resolution.  
The code serves as the official implementation of the algorithm described in the paper published in *[Computers & Structures ](https://www.sciencedirect.com/journal/computers-and-structures)*.

## Contents

- `run_all_pwfem_exps.py` — main script
- `scripts/` —model utilities
- `probsr/` — probabilistic utilities
- `model_zoo/` — Bayesian utilities
- `dataset_synthesis.py` — data synthesis
- `data/`— datasets
- `checkpoints/`— checkpoints

## Usage
Run in Python IDE:
```python
run_all_pwfem_exps.py
```

## Requirements

- `requirements.txt` 

## Citation
```latex
@article{zhong2026super,
title = {Super-Resolution for Low-Fidelity Physical System Observations with Probabilistic Wavelet Representation},
journal = {Computers \& structures},
author = {Zhirong Zhong and Zhongyi Zhang and Zhi Zhai and Meng Ma and Jinxin Liu},
}
```
## License

Released under the MIT License. See `LICENSE` for details.
