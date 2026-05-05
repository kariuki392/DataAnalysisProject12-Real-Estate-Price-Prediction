## 🏡 Real Estate Price Prediction — Ames Housing
### ✨ Feature Engineering & Regression Modelling for Property Valuation
Welcome to the **Real Estate Price Prediction** project — an end‑to‑end, notebook‑driven workflow that predicts residential property **sale prices** in **Ames, Iowa** using rich property characteristics and modern regression techniques.
This project walks through **data cleaning, exploratory data analysis, feature engineering, model training, and business interpretation**, turning raw housing data into an explainable, production‑ready modelling pipeline.  
---
## 📊 Project Overview
- **Goal**: Build a robust model to **predict house sale prices** and understand **what truly drives value** in residential real estate.
- **Dataset**: **Ames Housing Dataset**
  - 🧱 **2,930** residential properties  
  - 🧬 **82** raw features (structural, location, quality, and amenity characteristics)  
  - 📅 Sales between **2006–2010** in Ames, Iowa
- **Best Model**: **XGBoost Regressor**
  - ⭐ **R² ≈ 0.94** on the test set  
  - 📉 **RMSE ≈ \$20K**, **MAE ≈ \$12K**  
  - Typical error of about **7–11%** on a \$180K home — strong enough for **automated valuation models (AVMs)**.
---
## 🗂️ Repository Structure
- `notebooks/`
  - `real_estate_price_prediction.ipynb`  
    The **main analysis notebook**:
    - Data loading & cleaning  
    - Exploratory data analysis (EDA)  
    - Feature engineering  
    - Model training & evaluation  
    - Interpretation & conclusions
- `data/`
  - `raw/`
    - `AmesHousing.csv` — original **raw dataset** used in the notebook.
> 💡 All core logic currently lives inside the Jupyter notebook, making the project easy to explore and extend interactively.
---
## 🧰 Tech Stack
- **Language**
  - 🐍 Python 3.12
- **Data & Analysis**
  - `pandas`, `numpy`
- **Visualization**
  - `matplotlib`, `seaborn`
- **Machine Learning**
  - `scikit-learn`
  - `xgboost`
- **Utilities**
  - `warnings`, `matplotlib.ticker`
The notebook is styled for **clean visuals** via:
- Custom **color palette** (`PALETTE`)
- Tweaked `matplotlib` and `seaborn` themes for **publication-quality plots**
---
## 🔍 Analysis Pipeline
The notebook is structured in clear, numbered sections:
### 1️⃣ Setup & Data Loading
- Loads `AmesHousing.csv` with `pandas`.
- Cleans column names (replaces spaces and slashes with underscores).
- Prints:
  - Dataset shape (**2,930 x 82**)
  - **Sale price range** and **median price**
- Prepares the environment:
  - Imports all key libraries
  - Sets plotting style and figure DPI for crisp visuals
---
### 2️⃣ Exploratory Data Analysis (EDA)
- Summary statistics for major predictors:
  - `Overall_Qual`, `Gr_Liv_Area`, `Year_Built`, `SalePrice`
- Visuals (in the notebook) focus on:
  - Distribution of **sale prices**
  - Relationship between **quality**, **size**, **age**, and **price**
  - Neighbourhood‑level price differences
Purpose: develop **intuition** on how structural and locational features influence price before modelling.
---
### 3️⃣ Missing Data Analysis & Cleaning
Key decisions:
- **High-null categorical fields** like:
  - `Pool QC`, `Misc Feature`, `Alley`, `Fence`
  - Interpreted as **“absence” rather than missingness**  
    → Imputed with `'None'` instead of dropping rows.
- **Numeric nulls**
  - Filled with **median values** using `SimpleImputer`.
- Ensures a **complete, model-ready dataset** without aggressive dropping of observations.
---
### 4️⃣ Feature Engineering 💡
This section is the **heart of the modelling power**.
Engineered features aim to reflect **what buyers actually care about**, such as:
- 🧱 **Total living space**: combinations of basement, 1st, and 2nd floor areas
- 📏 **Total area / lot use**: more nuanced measures of usable house area
- 🕰️ **Age features**:
  - Time since construction
  - Time since last remodel
- 🏅 **Quality & condition composites**:
  - Overall quality indices  
  - Luxury/amenity indicators (e.g., garage, fireplaces)
- 🏡 **Neighbourhood and location signals**
These crafted features allow simpler models (like linear methods) to **capture non‑linear relationships** via smart transformations.
---
### 5️⃣ Feature Correlation Analysis
- Computes and visualizes **correlations** between engineered features and `SalePrice`.
- Identifies:
  - Strong positive drivers (e.g. **Overall quality**, **total square footage**)
  - Weaker or redundant fields that add limited signal
- Used to **guide model simplification** and avoid unnecessary noise.
---
### 6️⃣ Key Price Driver Visualisations 📈
Rich visual storytelling around the data, e.g.:
- **Price vs. Overall Quality**
- **Price vs. Total Living Area**
- **Price by Neighbourhood**
- **Impact of Garage capacity, Year built, and Remodel status**
- **Seasonality**: Prices across months of sale
These plots help bridge the gap between **raw numbers** and **business meaning**:
- Which improvements matter the most?
- How much does location change pricing?
- When is the best time to sell?
---
### 7️⃣ Regression Model Building 🤖
The modelling suite includes:
- **Linear Models**
  - Ridge Regression
  - Lasso Regression
  - ElasticNet
- **Tree‑based Models**
  - Random Forest Regressor
  - Gradient Boosting Regressor
  - XGBoost Regressor
Core modelling choices:
- Target variable is transformed to **log(SalePrice)**:
  - Reduces skew
  - Stabilizes variance
  - Produces more Gaussian‑like residuals
- All features are:
  - Appropriately **encoded**
  - **Scaled** via `StandardScaler` or `RobustScaler`
- Evaluation approach:
  - Train/test split
  - **Cross‑validation (KFold)** to verify generalisation
---
### 8️⃣ Model Evaluation & Interpretation ✅
For each model, the notebook reports:
- **Test R²**
- **Cross-validated R²**
- **RMSE (Root Mean Squared Error)**
- **MAE (Mean Absolute Error)**
Approximate performance summary:
| Model          | Test R² | CV R² | RMSE   | MAE   |
|----------------|--------:|------:|-------:|------:|
| Ridge          |   ~0.89 | ~0.88 | ~\$26K | ~\$17K |
| Lasso          |   ~0.89 | ~0.88 | ~\$26K | ~\$17K |
| ElasticNet     |   ~0.89 | ~0.88 | ~\$26K | ~\$17K |
| Random Forest  |   ~0.93 | ~0.91 | ~\$21K | ~\$13K |
| **XGBoost**    | **~0.94** | **~0.92** | **~\$20K** | **~\$12K** |
- **Winner**: 🏆 **XGBoost** with the best balance of bias/variance and strong error metrics.
- Interprets what **these metrics mean in dollars and business impact**.
---
### 9️⃣ Ridge & Lasso Coefficient Interpretation 🧠
- **Lasso** performs **feature selection**, driving many coefficients to zero:
  - Keeps **~60–70** of the 90+ engineered features
  - Confirms that **signal is concentrated** in a subset of meaningful attributes.
- **Ridge** spreads weights across correlated predictors, showing:
  - How groups of features (e.g., multiple size metrics) collectively share importance.
- Comparison:
  - Similar R² for both Ridge & Lasso
  - Different sparsity levels → trade‑off between **interpretability** and **stability**.
---
### 🔟 Key Findings & Business Takeaways 💼
From the notebook’s conclusion:
#### 🏆 Top Value Drivers
1. **Overall Quality**  
   - Strongest single predictor of price  
   - A one‑point jump on the 1–10 quality scale adds **≈ \$15K–\$25K**.
2. **Total Square Footage**  
   - Buyers pay **almost linearly for more space**.
3. **Neighbourhood**  
   - Price difference between top and bottom neighbourhoods exceeds **\$150K** at the median.
4. **Age / Year Built**  
   - Each additional decade of age **reduces expected price by ≈ \$5K–\$8K**.
5. **Garage Capacity**  
   - 2‑car garages command a **\$25K–\$30K premium** over no garage.
6. **Remodelling**  
   - Remodelled properties sell for about **\$15K more** than similar unremodelled homes.
7. **Seasonality**  
   - **May–July** shows consistently higher sale prices than **January–March**.
#### 🧮 Regularisation Insights
- **Lasso**:
  - Selects a **compact, interpretable feature set**.
  - Shows many original variables are **redundant or low signal**.
- **Ridge & Lasso**:
  - Achieve **similar R²**, supporting the idea that the **true signal is robust** across correlated predictors.
#### 📈 Business Applications
This modelling framework can be repurposed for:
- 🏦 **Automated Valuation Models (AVMs)** for mortgage & insurance use‑cases  
- 🏘️ **Investor tools** to flag **under‑priced** listings vs model predictions  
- 🛠️ **Renovation ROI calculators** to quantify the expected impact of upgrades  
- 💰 **Pricing strategy assistants** for sellers and agents to set **data‑driven listing prices**
---
## 🚀 Getting Started
### 1️⃣ Clone or Download the Project
```bash
git clone <your-repo-url>
cd realestateanalysis