import warnings

import matplotlib
from sklearn.model_selection import train_test_split

from util import load_data

warnings.filterwarnings("ignore")
matplotlib.use("Qt5Agg")


def main():
    df = load_data("pbc.csv")
    print(df.shape)
    print(f"First 5 columns:\n{df.head()}")
    df_dev, df_test = train_test_split(df, test_size=0.2)
    df_train, df_val = train_test_split(df_dev, test_size=0.25)

    print("\nTotal number of patients:", df.shape[0])
    print("Total number of patients in training set:", df_train.shape[0])
    print("Total number of patients in validation set:", df_val.shape[0])
    print("Total number of patients in test set:", df_test.shape[0])


if __name__ == "__main__":
    main()
