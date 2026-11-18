# Data

The dataset is not included in this repository. To run the notebook:

1. Download **Telco Customer Churn (IBM dataset)** from Kaggle:
   <https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset>
2. Put `Telco_customer_churn.xlsx` in this folder:

   ```
   data/Telco_customer_churn.xlsx
   ```

The notebook expects the Excel version with 7,043 rows and 33 columns (including `Churn Value`, `Churn Score`,
`CLTV` and `Churn Reason`). It is not the smaller CSV that is also widely shared under the name "Telco Customer
Churn", which has different column names.

Excel files in this folder are ignored by git.
