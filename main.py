import warnings

import matplotlib
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter
from sklearn.model_selection import train_test_split

from util import *
from setup_r import *

matplotlib.use("Qt5Agg")
warnings.filterwarnings("ignore")


def main():
    df = load_data("pbc.csv")
    print(df.shape)
    print(f"First 5 columns of DataFrame:\n{df.head()}")
    df_dev, df_test = train_test_split(df, test_size=0.2)
    df_train, df_val = train_test_split(df_dev, test_size=0.25)

    print("\nTotal number of patients:", df.shape[0])
    print("Total number of patients in training set:", df_train.shape[0])
    print("Total number of patients in validation set:", df_val.shape[0])
    print("Total number of patients in test set:", df_test.shape[0])

    continuous_columns = [
        "age",
        "bili",
        "chol",
        "albumin",
        "copper",
        "alk_phos",
        "ast",
        "trig",
        "platelet",
        "protime",
    ]

    df_train.loc[:, continuous_columns] = df_train.loc[:, continuous_columns].astype(
        float
    )

    df_val.loc[:, continuous_columns] = df_val.loc[:, continuous_columns].astype(float)

    df_test.loc[:, continuous_columns] = df_test.loc[:, continuous_columns].astype(
        float
    )

    mean = df_train.loc[:, continuous_columns].mean()
    std = df_train.loc[:, continuous_columns].std()
    df_train.loc[:, continuous_columns] = (
        df_train.loc[:, continuous_columns] - mean
    ) / std

    df_val.loc[:, continuous_columns] = (df_val.loc[:, continuous_columns] - mean) / std
    df_test.loc[:, continuous_columns] = (
        df_test.loc[:, continuous_columns] - mean
    ) / std

    to_encode = ["edema", "stage"]

    one_hot_train = to_one_hot(df_train, to_encode)
    one_hot_val = to_one_hot(df_val, to_encode)
    one_hot_test = to_one_hot(df_test, to_encode)
    print(one_hot_val.columns.tolist())
    print(f"There are {len(one_hot_val.columns)} columns")
    print(one_hot_train.shape)
    print(one_hot_train.head())
    cph = CoxPHFitter()
    cph.fit(
        one_hot_train,
        duration_col="time",
        event_col="status",
        fit_options={"step_size": 0.1},
    )

    cph.print_summary()
    cph.plot_covariate_groups("trt", values=[0, 1])
    plt.show()

    # Train
    scores = cph.predict_partial_hazard(one_hot_train)
    cox_train_scores = harrell_c(
        one_hot_train["time"].values, scores.values, one_hot_train["status"].values
    )
    # Validation
    scores = cph.predict_partial_hazard(one_hot_val)
    cox_val_scores = harrell_c(
        one_hot_val["time"].values, scores.values, one_hot_val["status"].values
    )
    # Test
    scores = cph.predict_partial_hazard(one_hot_test)
    cox_test_scores = harrell_c(
        one_hot_test["time"].values, scores.values, one_hot_test["status"].values
    )

    print("Train:", cox_train_scores)
    print("Val:", cox_val_scores)
    print("Test:", cox_test_scores)

    model = forest.rfsrc(
        ro.Formula("Surv(time, status) ~ ."),
        data=df_train,
        ntree=300,
        nodedepth=5,
        seed=-1,
    )

    print(f"\nModel summary:\n{model}")

    result = R.predict(model, newdata=df_val)
    scores = np.array(result.rx("predicted")[0])

    print("Cox Model Validation Score:", cox_val_scores)
    print(
        "Survival Forest Validation Score:",
        harrell_c(df_val["time"].values, scores, df_val["status"].values),
    )

    result = R.predict(model, newdata=df_test)
    scores = np.array(result.rx("predicted")[0])

    print("Cox Model Test Score:", cox_test_scores)
    print(
        "Survival Forest Validation Score:",
        harrell_c(df_test["time"].values, scores, df_test["status"].values),
    )

    vimps = np.array(forest.vimp(model).rx("importance")[0])

    y = np.arange(len(vimps))
    plt.barh(y, np.abs(vimps))
    plt.yticks(y, df_train.drop(["time", "status"], axis=1).columns)
    plt.title("VIMP (absolute value)")
    plt.show()


if __name__ == "__main__":
    main()
