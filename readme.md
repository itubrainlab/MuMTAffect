Sure! Here's the full content formatted in Markdown for direct use in a `README.md` file:

```markdown
# AFFEC Dataset Processing and Training Instructions

This repository contains code for processing the **AFFEC** dataset and training models using the processed multimodal data. Follow the instructions below to prepare the data and run training.

---

## 📦 1. Download the Dataset

Download the following components from the AFFEC dataset hosted on Zenodo:

- **Eye Tracking Data**
- **Pupil Data**
- **Face Analysis Data**
- **Electrodermal Activity (EDA) and Physiological Sensors**
- **Self-Annotations**

You can download the dataset from the following link:

🔗 [https://zenodo.org/records/14794876](https://zenodo.org/records/14794876)

Once downloaded, extract the contents into a directory of your choice.

> ✅ Make sure the folder contains:
> - `participants.tsv`
> - Subfolders for each participant (e.g., `sub-xxx/`) with their respective data files

---

## 🧪 2. Generate the Pickle Dataset

The `pickle_generation.py` script processes the multimodal sensor data and merges it into a single pickle file (`dataset.pkl`), which can then be used for training.

Open your terminal and run:

```bash
python pickle_generation.py --dataset_path /path/to/dataset
```

Replace `/path/to/dataset` with the actual path where you extracted the dataset.

> 🗃️ After this step, a file named `dataset.pkl` will be created in the dataset directory.

---

## 🧠 3. Train the Model

Once the dataset is processed, you can run the training pipeline. Make sure the correct path to the pickle file is provided.

Run the following command:

```bash
python multiphase_simple.py --data_path dataset.pkl
```

> 📝 This script trains a model on the dataset and saves the evaluation results in:
> - `results.csv`

---

## ✅ Summary

| Step | Description |
|------|-------------|
| 1️⃣   | Download AFFEC dataset from Zenodo |
| 2️⃣   | Run `pickle_generation.py` to preprocess the data |
| 3️⃣   | Run `multiphase_simple.py` to train and evaluate the model |

---



