# deep-learning-challenge
Module 21
#  Alphabet Soup Charity Fund Prediction

##  Overview

The purpose of this project is to build and evaluate a deep learning model to predict whether organizations applying to **Alphabet Soup** will receive funding. The goal is to help Alphabet Soup optimize its decision-making process by identifying patterns in successful applications.

---

## Data Preprocessing

###  Target Variable
- `IS_SUCCESSFUL`: Binary outcome — 1 for funded, 0 for not funded.

###  Feature Variables
- `APPLICATION_TYPE`
- `AFFILIATION`
- `CLASSIFICATION`
- `USE_CASE`
- `ORGANIZATION`
- `STATUS`
- `INCOME_AMT`
- `SPECIAL_CONSIDERATIONS`
- `ASK_AMT`

###  Removed Variables
- `EIN`: Non-predictive ID.
- `NAME`: Text identifier with no modeling value.

###  Preprocessing Steps
- Consolidated infrequent values in `APPLICATION_TYPE` and `CLASSIFICATION` under `"Other"`.
- Used `pd.get_dummies()` for one-hot encoding of categorical features.
- Applied `StandardScaler` to normalize numerical data.
- Split the data into training and testing sets (75/25 split).

---

## Neural Network Model

###  Model Architecture

| Layer        | Units | Activation | Dropout |
|--------------|-------|------------|---------|
| Dense        | 80    | ReLU       | 0.2     |
| Dense        | 30    | ReLU       | 0.2     |
| Dense        | 10    | ReLU       | —       |
| Output Layer | 1     | Sigmoid    | —       |

### Compilation Details
- **Loss Function:** Binary Crossentropy
- **Optimizer:** Adam
- **Epochs:** 100
- **Metrics:** Accuracy

###  Final Performance
- **Test Accuracy:** `72%`



---

##  Optimization Efforts

- Replaced rare categorical values with `"Other"` to reduce feature dimensionality.
- Added Dropout layers to reduce overfitting.
- Increased training epochs to allow deeper learning.
- Scaled numerical features for stable learning.
- Expanded network from 2 to 3 hidden layers.

---

##  Summary and Recommendation

- The final deep learning model achieved **72% accuracy**, just under the 75% target.
- Although the performance was close to the goal, further improvements could be achieved with different modeling approaches.



---

##  Project Structure

├── charity_data.csv
├── alphabet_soup_model.ipynb
├── model_summary.png
└── README.md

