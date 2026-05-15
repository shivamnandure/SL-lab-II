# 📌 Pre-Installation Commands for Abalone Practical

## ✅ For VS Code Terminal / CMD / PowerShell

Use this command before running the `.py` file:

```bash id="1o0w5s"
pip install pandas numpy matplotlib seaborn scikit-learn
```

If `pip` does not work, use:

```bash id="xf6t85"
python -m pip install pandas numpy matplotlib seaborn scikit-learn
```

---

# 📌 Run Python File in VS Code / CMD

```bash id="p2p3jz"
python filename.py
```

Example:

```bash id="0gfr4m"
python abalone.py
```

---

# 📌 For Anaconda Prompt / Anaconda Package Manager

Use this command:

```bash id="r6l1y9"
conda install pandas numpy matplotlib seaborn scikit-learn
```

If package solving takes too long, use:

```bash id="sokx2o"
conda install -c conda-forge pandas numpy matplotlib seaborn scikit-learn
```

---

# 📌 Run Jupyter Notebook (.ipynb)

First install Jupyter (if not installed):

```bash id="jz6c1o"
conda install jupyter
```

Then launch notebook:

```bash id="qk0rtv"
jupyter notebook
```

Open your `.ipynb` file and run cells using:

```text
Shift + Enter
```

---

# 📌 For Google Colab

Usually no installation is required because most libraries are pre-installed.

If needed:

```python id="2xgkg7"
!pip install pandas numpy matplotlib seaborn scikit-learn
```

---

# 📌 Libraries Used in This Practical

| Library        | Purpose                     |
| -------------- | --------------------------- |
| `pandas`       | Dataset handling            |
| `numpy`        | Numerical operations        |
| `matplotlib`   | Plotting graphs             |
| `seaborn`      | Advanced visualization      |
| `scikit-learn` | Machine Learning algorithms |

---

# 📌 Quick Viva Point

## Why install libraries?

Because Python requires external packages for:

* Data analysis
* Visualization
* Machine Learning implementation

Without these libraries, the program will not run properly.
