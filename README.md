
# RadioGUNet: Group Equivariant Convolutional Networks for Pathloss Estimation

This repository contains code for training and evaluating the RadioGUNet model in pathloss estimation on RadioMapSeer dataset

## Getting Started

### Prerequisites

- `requirements.txt`

### Dataset

Download the **[RadioMapSeer](https://radiomapseer.github.io/)** dataset as described in the paper.

---

## Reproducing Results

The experiments can be reproduced by running the training and testing scripts with the appropriate parameters:

- **model_path:** Where to save model checkpoints and logs
- **dataset_path:** Path to the RadioMapSeer dataset
- **experiment_type:**  
    - `DPM_no_car`
    - `DPM_cars`
    - `IRT_no_car`
    - `IRT_cars`
- **symmetry_group:**  
    - `C2`, `D2`, `C4`, `D4`, `C8`, `D8` (as in the paper, can be extended to `C16` `D16` so long as it is supported in `e2cnn`)

---

### Training

Replace `<exp_type>` and `<group>` with your choices:

```bash
python3 train.py \
  --model_path ./results/<exp_type>_<group> \
  --dataset_path ./RadioMapSeer \
  --experiment_type <exp_type> \
  --symmetry_group <group>
````

#### Example: DPM with cars, D8 group

```bash
python3 train.py \
  --model_path ./results/DPM_cars_D8 \
  --dataset_path ./RadioMapSeer \
  --experiment_type DPM_cars \
  --symmetry_group D8
```

---

### Evaluation

After training, run evaluation on the test split:

```bash
python3 test.py \
  --model_path ./results/<exp_type>_<group> \
  --dataset_path ./RadioMapSeer \
  --experiment_type <exp_type> \
  --symmetry_group <group>
```

#### Example: IRT without cars, D4 group

```bash
python3 test.py \
  --model_path ./results/IRT_no_car_D4 \
  --dataset_path ./RadioMapSeer \
  --experiment_type IRT_no_car \
  --symmetry_group D4
```

---

## Notes

* The scripts automatically configure the dataset loader and model based on your selected experiment and symmetry group.
* Results, logs, and model checkpoints will be saved in the specified `model_path`.
* For additional settings (e.g., batch size, learning rate, model size), edit `train.py` directly.

---

## Citation


