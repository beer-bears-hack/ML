import os
import yaml
import argparse
import pandas as pd
    
class Dataset_proc:
    def __init__(self):
        with open("config/ml_config.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        try:
            print("Loading raw data...")
            self.raw_cte_df = pd.read_excel(config["raw_cte_path"])
            self.raw_contracts_df = pd.read_excel(config["raw_contracts_path"])
        except Exception as e:
            print(f"ERROR: in loading raw df: {e}")
            return None

        self.PREP_CTE_PATH = config["prep_cte_path"]
        self.PREP_CONTRACTS_PATH = config["prep_contracts_path"]

    def _preprocess_cte(self):
        print("Preprocessing cte data...")
        
        #Fixing the corrupted data
        unnamed_cols = ['Unnamed: 5', 'Unnamed: 6', 'Unnamed: 7', 'Unnamed: 8', 'Unnamed: 9', 'Unnamed: 10']
        mask = self.raw_cte_df[unnamed_cols].notna().any(axis=1)
        corrupted_data = self.raw_cte_df[mask]
        rows_to_drop = corrupted_data["Идентификатор СТЕ"].to_list()
        self.raw_cte_df = self.raw_cte_df[~self.raw_cte_df["Идентификатор СТЕ"].isin(rows_to_drop)]
        self.raw_cte_df.drop(columns=unnamed_cols, inplace=True)

        #Renaming
        self.raw_cte_df = self.raw_cte_df.rename(columns={
            'Идентификатор СТЕ': "CTE_id", 
            'Наименование СТЕ': "CTE_name",
            'Категория': "category",
            'Производитель': "manufacturer",
            'характеристики СТЕ': "characteristics"
        })

        #Fixing & Dropping missing data
        for cte_id in [39114363, 36715295]:
            mask = self.raw_cte_df["CTE_id"] == cte_id
            self.raw_cte_df.loc[mask, "characteristics"] = (
                self.raw_cte_df.loc[mask, "manufacturer"]
                .astype(str)
                .apply(lambda x: x.split("\t")[1])
            )
            self.raw_cte_df.loc[mask, "manufacturer"] = (
                self.raw_cte_df.loc[mask, "manufacturer"]
                .astype(str)
                .apply(lambda x: x.split("\t")[0])
            )
        self.raw_cte_df.dropna(inplace=True)

        #Saving preprocessed data
        self.raw_cte_df.to_csv(self.PREP_CTE_PATH, index=False)
        print(f"Preprocessed CTE data is saved {self.PREP_CTE_PATH}")

    def _preprocess_contracts(self):
        print("Preprocessing contracts data...")

        #Fixing the corrupted data
        unnamed_cols = ['Unnamed: 17',
            'Unnamed: 18', 'Unnamed: 19', 'Unnamed: 20', 'Unnamed: 21',
            'Unnamed: 22', 'Unnamed: 23', 'Unnamed: 24', 'Unnamed: 25',
            'Unnamed: 26', 'Unnamed: 27', 'Unnamed: 28', 'Unnamed: 29',
            'Unnamed: 30', 'Unnamed: 31', 'Unnamed: 32', 'Unnamed: 33',
            'Unnamed: 34', 'Unnamed: 35']
        corrupted_data = self.raw_contracts_df[unnamed_cols].notna().any(axis=1)
        self.raw_contracts_df = self.raw_contracts_df[~corrupted_data]
        self.raw_contracts_df.drop(columns=unnamed_cols, inplace=True)
    
        #Deduplicating and Dropping useless columns
        self.raw_contracts_df.drop_duplicates(inplace=True)
        self.raw_contracts_df.drop(columns=["Единица измерения", "ИНН заказчика", "ИНН поставщика"], inplace=True)

        #Renaming
        self.raw_contracts_df = self.raw_contracts_df.rename(columns={
            "Идентификатор контракта": "contract_id",
            "Наименование закупки": "purchase_name",
            "Способ закупки": "purchase_type",
            "Регион заказчика": "customer_region",
            "Регион поставщика": "supplier_region",
            "Количество": "quantity",
            "Начальная стоимость контракта": "initial_contract_price",
            "Стоимость контракта после заключения": "final_contract_price",
            "% снижения": "discount_percent",
            "Ставка НДС": "nds_rate",
            "Дата заключения контракта": "contract_date",
            "Идентификатор СТЕ по контракту": "CTE_id",
            "Наименование позиции СТЕ": "CTE_name",
            "Цена за единицу": "unit_price",
        })

        #Filtering absent CTE_ids
        prep_cte_df = self.raw_cte_df.copy()
        valid_cte_ids = set(prep_cte_df["CTE_id"].unique())
        self.raw_contracts_df = self.raw_contracts_df[self.raw_contracts_df["CTE_id"].isin(valid_cte_ids)]

        #Saving preprocessed data
        self.raw_contracts_df.to_csv(self.PREP_CONTRACTS_PATH, index=False)
        print(f"Preprocessed contracts data is saved {self.PREP_CONTRACTS_PATH}")

    def preprocess(self):
        self._preprocess_cte()
        self._preprocess_contracts()

if __name__ == "__main__":
    dataset_proc = Dataset_proc()
    dataset_proc.preprocess()
