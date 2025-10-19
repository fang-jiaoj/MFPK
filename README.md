# MFPK
![framework](images/workflow.png)
MFPK is a multi-fidelity transfer-learning framework for predicting intravenous PK parameters across multiple species, including humans, dogs, monkeys, rats and mice. It incorporates graph-based, motif-based, and three-dimensional structure-based molecular representations to capture comprehensive, multi-scale chemical information.

## Quick Start
👉 We strongly recommend using our [**MFPK Web Server**](https://lmmd.ecust.edu.cn/MFPK/) for new prediction.

## Reporduce the results of MFPK
📜 We provide raw datasets for humans (`Human_data`) and animals (`Animal_data`) in `Data/Raw_data`, along with preprocessed datasets (`multitask_datasets.csv`) in `Data/MTL_data`. To reproduce the research results, you will need to generate the feature library (.lmdb) first. However, all code is integrated in `Code/MFPK_finetune.py`. Use the following commands for training and evaluation:

```
python MFPK_finetune.py
```

The best model checkpoints and prediction results will be saved under the 
- `Model/Finetune`
- directories `Result/`

For external evaluation, you should ensure that the external datasets are placed under the following directories: `/Data/`.

```
python MFPK_predict.py
```
This script will load all five pre-trained MFPK models and generate average prediction results for the compounds. The predicted multi-species PK parameters and overall metrics for each compound will be saved to the following directory: `Result/test_datasets_prediction/`

## License
This project is licensed under the [**MIT License**](https://github.com/fang-jiaoj/MFPK/blob/main/LICENSE)

