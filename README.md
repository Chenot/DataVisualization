# 📊 Open Science Visualization Toolkit

**Author:** Quentin Chenot  
**License:** MIT  
**Version:** 1.0  
**Date:** 15 September 2025  

This repository provides **flexible plotting functions** designed for transparent and reproducible research workflows:

- **Correlation plots** (`plot_correlation`)  
- **Likert plots** (`plot_likert`)  
- **Raincloud plots** (`plot_raincloud`)  

All functions are implemented in **Python**, rely on standard scientific libraries, and include example datasets + usage scripts to support reproducibility.  

---

## ✨ Features

- 📈 **Correlation plots** with regression lines, marginal densities, and annotated *r, r², p-values*  
- 📝 **Likert plots** for visualizing questionnaire data on diverging scales  
- 🌧 **Raincloud plots** (half-violin + boxplot + jitter) with optional ANOVA results  
- 💾 Export figures (PDF, PNG, etc.)
---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/YourUsername/OpenSciencePlots.git
cd OpenSciencePlots
```

Install required dependencies:

```bash
pip install -r requirements.txt
```

Minimal requirements:
- `matplotlib`
- `seaborn`
- `pandas`
- `numpy`
- `scipy`
- `statsmodels`

---

## 🚀 Usage

Each script is standalone and can be used independently. Example:

```python
from correlation_plot import plot_correlation
import seaborn as sns

df = sns.load_dataset("mpg").dropna(subset=["horsepower", "mpg"])
plot_correlation(df, "horsepower", "mpg",
                 x_label="Horsepower", y_label="Miles per Gallon",
                 save_path="output/correlation_plot.png")
```

More examples are included in each script’s `__main__` block.

---

## 📊 Example Outputs

### Correlation Plot
![Correlation Plot](output/correlation_plot.png)

### Likert Plot
![Likert Plot](output/likert_plot.png)

### Raincloud Plot (One-Way)
![Raincloud One-Way](output/raincloud_plot_1way.png)

### Raincloud Plot (Two-Way)
![Raincloud Two-Way](output/raincloud_plot_2way.png)


---

## 📜 License

This project is licensed under the **MIT License**.  
You are free to use, modify, and distribute this software with attribution.  
