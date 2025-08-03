import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pathlib import Path


current_path = Path.cwd()
# Get the root directory
root_directory = current_path
print(root_directory)


df = pd.read_csv(
    root_directory / "data" / "01_extracted_data" / "stores_sales_forecasting.csv",
    encoding="ISO-8859-1",
)
df = df.drop(
    [
        "Row ID",
        "Order ID",
        "Customer ID",
        "Product ID",
        "Country",
        "Category",
        "Ship Mode",
        "Ship Date",
        "Customer Name",
        "Segment",
        "City",
        "Postal Code",
        "Region",
        "Product Name",
        "Discount",
        "Profit",
    ],
    axis=1,
)


print(df.columns)
df = df.sort_values("Order Date")
df["Order Date"] = pd.to_datetime(df["Order Date"])
df_qte_pivot = (
    df.groupby([pd.Grouper(key="Order Date", freq="M"), "Sub-Category"])["Quantity"]
    .sum()
    .unstack(fill_value=0)
)

df_ts_month = df.groupby(pd.Grouper(key="Order Date", freq="M"))["Sales"].sum()

df = df_qte_pivot.join(df_ts_month).reset_index()
df["month"] = df["Order Date"].dt.month
df["year"] = df["Order Date"].dt.year


# Encodage cyclic du mois (saisonnalité annuelle)
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

# Créer les lags pour intégrer l'historique (ici 1 à 12 mois)
for lag in range(1, 7):
    df[f"sales_lag_{lag}"] = df["Sales"].shift(lag)

# Tendance (trend) : on peut simplement utiliser l'ordre temporel
df["trend"] = np.arange(len(df))

# Supprimer les lignes avec NaN (car lags au début)
df = df.dropna()


df = df.drop(
    columns=["Order Date", "month", "year"]
)  # Garde uniquement features utiles

print(df.head(10))

X = df.drop(columns=["Sales"])
y = df["Sales"]

tscv = TimeSeriesSplit(n_splits=6)
rmse_scores = []

for fold, (train_index, test_index) in enumerate(tscv.split(X), 1):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model = xgb.XGBRegressor(
        objective="reg:absoluteerror",
        n_estimators=3500,
        learning_rate=0.03,
        max_depth=19,
        random_state=42,
        # early_stopping_rounds=30
    )

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    mask = y_test != 0
    mape = np.mean(np.abs((y_test[mask] - preds[mask]) / y_test[mask])) * 100

    print(f"MAE: {mae:.2f}")
    print(f"MAPE: {mape:.2f}%")
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    rmse_scores.append(rmse)

    print(f"Fold {fold} RMSE: {rmse:.2f}")

print(f"\nAverage RMSE: {np.mean(rmse_scores):.2f}")


"""
X = df.drop(columns=['Sales'])
y = df['Sales']

tscv = TimeSeriesSplit(n_splits=3)
mae_scores = []
rmse_scores = []

for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Création du dataset LightGBM
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test)

    # Paramètres LightGBM
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.05,
        'num_leaves': 12,
        'verbose': -1,
        'seed': 42
    }

    # Entraînement avec early stopping
    model = lgb.train(
        params,
        train_data,
        num_boost_round=2000,
        valid_sets=[valid_data],
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(50)]
    )

    preds = model.predict(X_test, num_iteration=model.best_iteration)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    mae_scores.append(mae)
    rmse_scores.append(rmse)

    print(f"Fold {fold} — MAE: {mae:.2f}, RMSE: {rmse:.2f}")

print(f"\nAverage MAE: {np.mean(mae_scores):.2f}")
print(f"Average RMSE: {np.mean(rmse_scores):.2f}")




# Supposons df avec colonne 'Sales' (target), index datetime et colonnes exogènes

X = df.drop(columns=['Sales'])
y = df['Sales']
print(np.min(y))
tscv = TimeSeriesSplit(n_splits=3)

mae_scores = []
rmse_scores = []

for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Modèle SARIMAX : p,d,q et P,D,Q saisonnier à adapter à tes données
    model = SARIMAX(
        y_train,
        exog=X_train,
        order=(1,2,1),
        seasonal_order=(2,1,0,12),  # 12 pour saisonnalité annuelle si données mensuelles
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    model_fit = model.fit(disp=False)

    # Prédictions sur la période test avec exogènes test
    preds = model_fit.forecast(steps=len(y_test), exog=X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    mae_scores.append(mae)
    rmse_scores.append(rmse)

    print(f"Fold {fold} — MAE: {mae:.2f}, RMSE: {rmse:.2f}")

print(f"\nAverage MAE: {np.mean(mae_scores):.2f}")
print(f"Average RMSE: {np.mean(rmse_scores):.2f}")
"""
