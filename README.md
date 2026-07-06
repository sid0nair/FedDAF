# FedDAF: Federated Domain Adaptation Using Model Functional Distance

Official implementation of **FedDAF**, a Federated Domain Adaptation (FDA) method that aggregates a target client's model with the global source model using a similarity score derived from **model functional distance** — the cosine similarity between mean gradient fields, normalized with a **Gompertz function**.

> Mrinmay Sen*, Sidhant Nair*, C Krishna Mohan. **FedDAF: Federated Domain Adaptation Using Model Functional Distance.** *(under review, Springer Machine Learning)*
> *Equal contribution.

## Overview

Federated Domain Adaptation lets a target client with limited labeled data improve its model by collaborating with source clients that have abundant data but a shifted data distribution. Most FDA methods either assume plenty of unlabeled target data, or fail to tailor how much source information gets transferred to the target's own objective.

FedDAF addresses both issues at once:

1. Each source client trains locally; the server aggregates local source models into a global source model.
2. The target client computes the **mean gradient field** of the global source model and of its own previous model, evaluated on its (limited) local target data.
3. The **cosine similarity** between these two mean gradient fields is converted to an angle, then normalized with a **Gompertz function** to produce an aggregation weight `α ∈ [0, 1]`.
4. The target and global source models are combined as `w = α·w_S + (1 − α)·w_T`, giving an *adapted target model* that sits at a similarity-informed point between the two objectives.
5. The adapted model is then fine-tuned on the target's local data.

This lets the target pull in exactly as much source information as its own objective supports, even with very few labeled target samples.

## Repository structure

The paper reports two experimental settings — **controlled domain shift** (CIFAR-10 with injected noise + label scarcity, Table 1) and **real-world domain shift** (PACS / VLCS / Office-Caltech-10, Table 2) — each with its own set of scripts, since the data pipeline differs (synthetic partitioning + noise vs. `ImageFolder` domains).

```
FedDAF/
├── src/
│   ├── cifar10_controlled_shift/   # Table 1: CIFAR-10, Dirichlet source partitioning + Gaussian noise
│   │   ├── FedDAF.py       # Proposed method
│   │   ├── FedAvg.py       # Baseline: vanilla FedAvg
│   │   ├── FedAvgFT.py     # Baseline: FedAvg + target fine-tuning
│   │   ├── FedDWA.py       # Baseline: FedDWA (Liu et al., IJCAI 2023)
│   │   ├── FedGP.py        # Baseline: FedGP (Jiang et al., ICLR 2024)
│   │   └── TargetOnly.py   # Baseline: target-only training (no federation)
│   └── real_domain_shift/          # Table 2: PACS, VLCS, Office-Caltech-10
│       ├── FedDAF.py
│       ├── FedAvg.py
│       ├── FedAvgFT.py
│       ├── FedDWA.py
│       ├── FedGP.py        # also runs the FedDA variant
│       └── TargetOnly.py
├── requirements.txt
├── LICENSE
└── README.md
```

Both tracks implement the same FedDAF method (Algorithms 1–3); only the data loading, partitioning, and domain-shift mechanism differ.

## Installation

```bash
git clone https://github.com/<your-username>/FedDAF.git
cd FedDAF
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Tested with PyTorch 1.12.1 (CUDA 10.2) on a single Tesla V100, per the paper's experimental setup. Newer PyTorch/torchvision releases should also work.

## Data

- **CIFAR-10** (`src/cifar10_controlled_shift/`) downloads automatically via `torchvision.datasets.CIFAR10(root='data', download=True)` the first time you run any script in this folder — no manual setup needed. Client partitioning (Dirichlet allocation across source clients), Gaussian-noise injection, and target label-scarcity splits are all handled inside the script via `--dir_alpha`, `--noise`, and `--degree_scarcity`.
  Source: Krizhevsky, A., Hinton, G. *Learning Multiple Layers of Features from Tiny Images.* 2009. https://www.cs.toronto.edu/~kriz/cifar.html

- **PACS / VLCS / Office-Caltech-10 / Office-Home** (`src/real_domain_shift/`) require manual download and are loaded via `torchvision.datasets.ImageFolder`, one subfolder per class within each domain folder. Point scripts at your local copy with `--data_root /path/to/data` (defaults to `./data`).

  | Dataset | Source | Download |
  |---|---|---|
  | PACS | Li, D., Yang, Y., Song, Y., Hospedales, T.M. *Deeper, Broader and Artier Domain Generalization.* ICCV 2017. | https://domaingeneralization.github.io/#data |
  | VLCS | Fang, C., Xu, Y., Rockmore, D.N. *Unbiased Metric Learning...* ICCV 2013. | via [DomainBed](https://github.com/facebookresearch/DomainBed) (`domainbed/scripts/download.py --dataset vlcs`) |
  | Office-Caltech-10 | Gong, B., Shi, Y., Sha, F., Grauman, K. *Geodesic Flow Kernel for Unsupervised Domain Adaptation.* CVPR 2012. | https://github.com/jindongwang/transferlearning/blob/master/data/dataset.md#office-caltech10 |
  | Office-Home *(supported in code, not reported in the paper)* | Venkateswara, H., et al. *Deep Hashing Network for Unsupervised Domain Adaptation.* CVPR 2017. | https://www.hemanthdv.org/officeHomeDataset.html |

  All datasets above are publicly released for research use by their original authors; please cite the corresponding papers if you use them.

## Usage

Each script is self-contained and run independently.

### CIFAR-10 controlled domain shift (Table 1)

```bash
python src/cifar10_controlled_shift/FedDAF.py \
    --dataset CIFAR10 \
    --num_users 10 \
    --dir_alpha 1 \
    --noise 0.9 \
    --degree_scarcity 0.25 \
    --batch_sizeS 64 \
    --batch_sizeT 16 \
    --global_epoch 50 \
    --learning_rate 0.01 \
    --learning_rate_target 0.001 \
    --k 5 \
    --mu 0.001 \
    --seed 50
```

Key arguments specific to this track:

| Argument | Description |
|---|---|
| `--num_users` | Number of source clients (10 in the paper) |
| `--dir_alpha` | Dirichlet concentration parameter for source partitioning (1 in the paper) |
| `--noise` | Std. dev. of Gaussian noise injected into target data — controls domain-shift severity (`{0.3, 0.6, 0.9}` in Table 1) |
| `--degree_scarcity` | Fraction of target training data used — controls label scarcity (`{0.05, 0.25, 0.5}` in Table 1) |
| `--k` | Gompertz function parameter (this is `µ` in the paper) |
| `--mu` | Proximal-regularization coefficient in target local training (this is `λ` in the paper's proximal term — unrelated to `--k`/`µ`; default `0` for this track) |
| `--device_server` / `--device_local` | GPU device indices — adjust for your hardware (single-GPU/CPU setups should edit the `.to(device)` calls or set both to the same index) |

Same convention for the other baselines in this folder: `FedAvg.py`, `FedAvgFT.py`, `FedDWA.py`, `FedGP.py`, `TargetOnly.py`.

### Real-world domain shift (Table 2)

```bash
python src/real_domain_shift/FedDAF.py \
    --dataset PACS \
    --data_root ./data \
    --batch_sizeS 32 \
    --batch_sizeT 8 \
    --global_epoch 50 \
    --learning_rate 0.01 \
    --learning_rate_target 0.001 \
    --k 5 \
    --seed 50
```

Supported `--dataset` values: `PACS`, `VLCS`, `offcie_caltech10` (Office-Caltech-10), `OfficeHome`.

| Argument | Description |
|---|---|
| `--dataset` | Which benchmark to run |
| `--data_root` | Root folder containing the datasets (see `data/README.md`) |
| `--batch_sizeS` / `--batch_sizeT` | Source / target local batch size |
| `--learning_rate` / `--learning_rate_target` | Source / target SGD learning rate |
| `--global_epoch` | Number of federated communication rounds |
| `--k` | Gompertz function parameter (`µ` in the paper) |
| `--mu` | Proximal-regularization coefficient (`λ` in the paper's proximal term; default `0.001` for this track — **note this differs from the CIFAR-10 track's default of `0`**) |
| `--seed` | Random seed (50 in all reported results) |

The target domain, class count, and train/test split percentage for each dataset are set at the top of the `if __name__ == '__main__':` block in each script — edit these directly to change which domain is held out as the target. Same CLI convention applies to `FedAvg.py`, `FedAvgFT.py`, `FedDWA.py`, `FedGP.py`, `TargetOnly.py` in this folder.

## Method-to-code map

Paths below are relative to `src/`; the same functions/structure appear in both `cifar10_controlled_shift/FedDAF.py` and `real_domain_shift/FedDAF.py`.

| Paper element | Code |
|---|---|
| Algorithm 2 (mean gradient field) | `FindPersonalizedModel` (gradient accumulation loops) in `FedDAF.py` |
| Algorithm 3 (Gompertz-normalized target aggregation, Eq. 2) | `FindPersonalizedModel` in `FedDAF.py` |
| Eq. 1 (final adapted model `w = αw_S + (1−α)w_T`) | same function, `mod = sim*flat_grad(...) + (1-sim)*flat_grad(...)` |
| Target local training (Section 4.2) | `local_update_target` in `FedDAF.py` — includes a FedProx-style proximal term (`args.mu`) not currently described in the paper; regularization anchor is the un-adapted global source model, not the adapted model used for initialization. Manuscript update in progress. |
| Source local training (Section 4.3) | `local_update_source` in `FedDAF.py` |
| Source model aggregation (Eq. 3) | `aggregate_with_softmax` in `FedDAF.py` — aggregates source models by softmax weighting over their distance to the target model, rather than the plain average currently in Eq. 3 of the paper. Manuscript update in progress. |

## Citation

If you use this code, please cite:

```bibtex
@article{sen2026feddaf,
  title   = {FedDAF: Federated Domain Adaptation Using Model Functional Distance},
  author  = {Sen, Mrinmay and Nair, Sidhant and Mohan, C Krishna},
  journal = {Machine Learning},
  year    = {2026},
  note    = {Under review}
}
```

## License

Released under the [MIT License](LICENSE).

## Contact

Sidhant Nair — sid.nairiitd@gmail.com
