# Early-Fault-Detection-in-PV-A-Case-Study-on-the-GPVS-Faults-Dataset
Unsupervised deep learning framework using autoencoders for early detection of PV inverter faults on the GPVS-Faults dataset. Learns healthy behavior and flags deviations, reducing reliance on labeled fault data or expert rules.




## Table of Contents
- [Objectives](#Objectives)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Contributors](#contributors)




## Objectives
- Develop and evaluate an unsupervised deep learning framework using autoencoders for early detection of PV inverter faults on the GPVS-Faults dataset.
- Learn normal inverter behavior from healthy data to automatically identify deviations.
- Provide an alternative to rule-based or supervised methods that require labeled fault data and expert knowledge.




## Installation
```bash
git clone https://github.com/NinjaZny/Early-Fault-Detection-in-PV-A-Case-Study-on-the-GPVS-Faults-Dataset.git
cd Early-Fault-Detection-in-PV-A-Case-Study-on-the-GPVS-Faults-Dataset
pip install -r requirements.txt
```




## Dataset

The GPVS-Faults dataset is **large (>100 MB)** to be stored in GitHub.  
Please download it directly from the official source:  
[GPVS-Faults on Mendeley Data](https://data.mendeley.com/datasets/n76t439f65/1)

### Steps:
1. Download the dataset from the link above.
2. Extract the files after download.
3. Place the extracted `.mat` files into the `data/` folder of this project.

### Expected Dataset Structure
- `data/`
  - `GPVS-Faults/`
    - `F0L.mat`
    - `F0M.mat`
    - ...
    - `F7M.mat`

---

## Project Structure
- `data/` — Dataset storage (ignored in Git)
- `src/` — Core source code (models, training, evaluation)
- `notebooks/` — Jupyter notebooks for experiments and visualization
- `outputs/` — Training outputs (ignored in Git)
- `requirements.txt` — Python dependencies list
- `.gitignore` — Git ignore rules
- `ISCF_ClarificationNote.pdf`
- `README.md` — Project documentation






## Contributors
- Aya Benkirane
- Shruti Debath
- Maoye Guan
- Ningyuan Zhang