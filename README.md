# Human Activity Recognition for Management of Lower Back Pain

Lower back pain is an increasingly common, expensive, and debilitating condition that affects all age groups. Inspired by the works of Sani, Wiratunga, et al. on the [SELFBACK](https://link.springer.com/chapter/10.1007/978-3-319-61030-6_23) system for Lower Back Pain management, my objective through this project is to develop a subject-indepent (unseen user) classifier that identifies four physical activity types (sedentary, walking, ascending/descending stairs and running) through a wrist-bound accelerometer's readings. The dataset was compiled by the team of the original paper and sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/selfBACK).


The project is built in Python 3.10.5 and repository organization is as follows:
1. `data/` - Stores all raw and preprocessed data files.
2. `figures/` - Stores all visualizations; from EDA to model result comparisons
3. `results/` - Trained models and comparison results
4. `report/` - Reports on development pipeline, methodology, and model results.
5. `src/` - This directory contains all the code for: data wrangling, EDA and feature engineering in `eda-ftrengg.ipynb`; group based splittinbg, preprocessing and model development in `model-development.ipynb`; and model evaluations, result visualiztion and feature importances in `model-evaluation.ipynb`.


The key packages used in this project are the following<sup>*</sup>:
- detecta 0.0.5
- matplotlib 3.5.2
- numpy	1.22.4
- pandas 1.4.2
- pickle 4.0
- scikit-learn 1.1.1
- scipy 1.9.1
- seaborn 0.11.2
- shap 0.40.0
- xgboost 1.5.1




<sup>*</sup> Check file environment.yml for complete configuration and package dependencies