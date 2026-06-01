"""
forecaster.py — Ансамблевый прогноз продаж по SKU
Горизонт: 1-3 месяца | Факторы: цена, промо, склад, реклама, сезонность

Модели в ансамбле:
  1. LightGBM       — основная, хорошо на коротких рядах с экзогенными
  2. SARIMAX        — только если история >= 24 мес (иначе нестабильна)
  3. Prophet        — baseline с сезонностью и праздниками
  4. ExponentialSmoothing — fallback при малой истории (< 12 мес)

Финальный прогноз = взвешенное среднее по 1/MAPE каждой модели.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from typing import Optional
from dataclasses import dataclass, field

# ── Метрики ────────────────────────────────────────────────────────────────
def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true > 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# ── Результат одной модели ──────────────────────────────────────────────────
@dataclass
class ModelResult:
    name: str
    forecast: np.ndarray          # прогноз на horizon шагов
    ci_lower: np.ndarray          # нижняя граница 80% CI
    ci_upper: np.ndarray          # верхняя граница 80% CI
    val_mape: float               # MAPE на валидационном окне
    val_rmse: float
    feature_importance: dict = field(default_factory=dict)
    notes: str = ""


# ── Подготовка признаков ────────────────────────────────────────────────────
KNOWN_FEATURES = [
    # Цена и скидки (lag1 — чтобы не смотреть в будущее)
    "avg_price_wb", "avg_price_ozon",
    "planned_discount_next_month",   # план на следующий месяц — известен заранее
    # Реклама
    "ads_budget_wb", "ads_budget_ozon",
    # Склад
    "stock_wb", "stock_ozon", "inventory_turnover_days",
    # Промо
    "promo_plan_next_month",         # план — известен заранее
    "promo_intensity_index",
    # Сезонность и внешние
    "seasonality_index", "is_q4",
    "usd_rate", "consumer_demand_index",
]

def build_features(df: pd.DataFrame, target: str = "revenue_total") -> pd.DataFrame:
    """
    Строит матрицу признаков из сырых данных.
    Правило no-leakage: рыночные факторы (цена, реклама, склад) сдвигаются на lag1.
    Плановые переменные (промо, скидка на следующий месяц) — не сдвигаются,
    они известны заранее.
    """
    d = df.copy()

    # Лаговые признаки для рыночных переменных
    lag1_cols = [
        "avg_price_wb", "avg_price_ozon",
        "ads_budget_wb", "ads_budget_ozon",
        "stock_wb", "stock_ozon", "inventory_turnover_days",
        "usd_rate", "consumer_demand_index",
        "promo_intensity_index",
    ]
    for col in lag1_cols:
        if col in d.columns:
            d[f"{col}_lag1"] = d[col].shift(1)

    # Плановые — можно использовать напрямую
    for col in ["planned_discount_next_month", "promo_plan_next_month"]:
        if col in d.columns:
            d[f"{col}_known"] = d[col]

    # Временные признаки
    d["month"]     = d.index.month
    d["month_sin"] = np.sin(2 * np.pi * d["month"] / 12)
    d["month_cos"] = np.cos(2 * np.pi * d["month"] / 12)
    d["trend_idx"] = np.arange(len(d))

    # Скользящие средние целевой переменной (только прошлое)
    if target in d.columns:
        d["target_ma3"] = d[target].shift(1).rolling(3, min_periods=1).mean()
        d["target_ma6"] = d[target].shift(1).rolling(6, min_periods=1).mean()
        d["target_lag1"] = d[target].shift(1)
        d["target_lag2"] = d[target].shift(2)
        d["target_lag3"] = d[target].shift(3)

    return d


def select_features(df: pd.DataFrame, target: str, min_importance: float = 0.01) -> list[str]:
    """
    Отбор признаков через LightGBM importance.
    Возвращает список признаков с importance > порога.
    """
    from lightgbm import LGBMRegressor

    candidates = [c for c in df.columns if c != target
                  and not c.startswith("_")
                  and df[c].notna().sum() > 5]

    X = df[candidates].fillna(0)
    y = df[target]

    lgb = LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
    lgb.fit(X, y)

    imp = dict(zip(candidates, lgb.feature_importances_ / lgb.feature_importances_.sum()))
    selected = [f for f, v in sorted(imp.items(), key=lambda x: -x[1]) if v >= min_importance]

    # Минимум 3 признака
    if len(selected) < 3:
        selected = sorted(imp, key=lambda x: -imp[x])[:3]

    return selected, imp


# ══════════════════════════════════════════════════════════════════════════
# МОДЕЛИ
# ══════════════════════════════════════════════════════════════════════════

def _train_val_split(y: pd.Series, X: pd.DataFrame, val_months: int = 3):
    n = len(y)
    n_val = min(val_months, max(1, n // 5))
    n_train = n - n_val
    return (y.iloc[:n_train], X.iloc[:n_train],
            y.iloc[n_train:], X.iloc[n_train:])


# ── 1. LightGBM ────────────────────────────────────────────────────────────
def fit_lgbm(y: pd.Series, X: pd.DataFrame,
             horizon: int, future_X: pd.DataFrame) -> ModelResult:
    from lightgbm import LGBMRegressor

    y_tr, X_tr, y_val, X_val = _train_val_split(y, X)

    model = LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )

    # log-transform для стабильности
    model.fit(X_tr, np.log1p(y_tr),
              eval_set=[(X_val, np.log1p(y_val))],
              callbacks=[])

    val_pred = np.expm1(model.predict(X_val))
    v_mape = mape(y_val.values, val_pred)
    v_rmse = rmse(y_val.values, val_pred)

    # Прогноз: рекурсивный на horizon шагов
    preds = []
    X_fut = future_X.copy().fillna(0)
    for step in range(horizon):
        row = X_fut.iloc[[step]] if step < len(X_fut) else X_fut.iloc[[-1]]
        p = float(np.expm1(model.predict(row)[0]))
        preds.append(max(0, p))

    # CI через quantile regression
    q_low = LGBMRegressor(
        objective="quantile", alpha=0.1, n_estimators=200,
        learning_rate=0.05, num_leaves=15, verbose=-1, random_state=0
    )
    q_high = LGBMRegressor(
        objective="quantile", alpha=0.9, n_estimators=200,
        learning_rate=0.05, num_leaves=15, verbose=-1, random_state=0
    )
    q_low.fit(X_tr, np.log1p(y_tr))
    q_high.fit(X_tr, np.log1p(y_tr))

    ci_l, ci_h = [], []
    for step in range(horizon):
        row = X_fut.iloc[[step]] if step < len(X_fut) else X_fut.iloc[[-1]]
        ci_l.append(max(0, float(np.expm1(q_low.predict(row)[0]))))
        ci_h.append(max(0, float(np.expm1(q_high.predict(row)[0]))))

    imp = dict(zip(X_tr.columns,
                   model.feature_importances_ / model.feature_importances_.sum()))

    return ModelResult(
        name="LightGBM",
        forecast=np.array(preds),
        ci_lower=np.array(ci_l),
        ci_upper=np.array(ci_h),
        val_mape=v_mape,
        val_rmse=v_rmse,
        feature_importance=imp,
    )


# ── 2. SARIMAX ─────────────────────────────────────────────────────────────
def fit_sarimax(y: pd.Series, X: pd.DataFrame,
                horizon: int, future_X: pd.DataFrame) -> Optional[ModelResult]:
    """Только если история >= 24 месяцев, иначе None."""
    if len(y) < 24:
        return None

    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from pmdarima import auto_arima

    y_tr, X_tr, y_val, X_val = _train_val_split(y, X)
    y_tr_log = np.log1p(y_tr)

    try:
        auto = auto_arima(
            y_tr_log, exogenous=X_tr,
            seasonal=True, m=12,
            max_p=2, max_q=2, max_P=1, max_Q=1,
            d=None, max_d=1, D=None,
            stepwise=True, suppress_warnings=True,
            error_action="ignore", information_criterion="bic",
        )
        order, s_order = auto.order, auto.seasonal_order
    except Exception:
        order, s_order = (1, 1, 0), (0, 1, 0, 12)

    model = SARIMAX(y_tr_log, exog=X_tr, order=order, seasonal_order=s_order,
                    enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)

    # Валидация
    val_pred_log = res.get_forecast(steps=len(y_val), exog=X_val).predicted_mean
    val_pred = np.expm1(val_pred_log.values)
    v_mape = mape(y_val.values, val_pred)
    v_rmse = rmse(y_val.values, val_pred)

    # Прогноз
    X_fc = future_X.iloc[:horizon].fillna(0) if len(future_X) >= horizon else \
        pd.concat([future_X, pd.DataFrame([future_X.iloc[-1]] * (horizon - len(future_X)))])

    fc = res.get_forecast(steps=horizon, exog=X_fc)
    forecast = np.expm1(fc.predicted_mean.values)
    ci = fc.conf_int(alpha=0.2)
    ci_l = np.expm1(ci.iloc[:, 0].values)
    ci_h = np.expm1(ci.iloc[:, 1].values)

    return ModelResult(
        name="SARIMAX",
        forecast=np.maximum(0, forecast),
        ci_lower=np.maximum(0, ci_l),
        ci_upper=np.maximum(0, ci_h),
        val_mape=v_mape, val_rmse=v_rmse,
        notes=f"order={order}, seasonal={s_order}",
    )


# ── 3. Prophet ─────────────────────────────────────────────────────────────
def fit_prophet(y: pd.Series, X: pd.DataFrame,
                horizon: int, future_X: pd.DataFrame) -> Optional[ModelResult]:
    try:
        from prophet import Prophet
    except ImportError:
        return None

    y_tr, X_tr, y_val, X_val = _train_val_split(y, X)

    df_train = pd.DataFrame({"ds": y_tr.index, "y": y_tr.values})

    regressors = [c for c in X_tr.columns
                  if c in ["ads_budget_wb_lag1", "ads_budget_ozon_lag1",
                            "planned_discount_next_month_known",
                            "seasonality_index", "promo_plan_next_month_known"]]

    for r in regressors:
        df_train[r] = X_tr[r].values

    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        interval_width=0.8,
        changepoint_prior_scale=0.1,
    )
    for r in regressors:
        m.add_regressor(r)

    m.fit(df_train)

    # Валидация
    df_val = pd.DataFrame({"ds": y_val.index})
    for r in regressors:
        df_val[r] = X_val[r].values if r in X_val.columns else 0
    val_fc = m.predict(df_val)
    val_pred = np.maximum(0, val_fc["yhat"].values)
    v_mape = mape(y_val.values, val_pred)
    v_rmse = rmse(y_val.values, val_pred)

    # Будущие даты
    last_date = y.index[-1]
    future_dates = pd.date_range(last_date, periods=horizon + 1, freq="MS")[1:]
    df_future = pd.DataFrame({"ds": future_dates})
    for r in regressors:
        df_future[r] = future_X[r].values[:horizon] if r in future_X.columns else 0

    fc = m.predict(df_future)
    return ModelResult(
        name="Prophet",
        forecast=np.maximum(0, fc["yhat"].values),
        ci_lower=np.maximum(0, fc["yhat_lower"].values),
        ci_upper=np.maximum(0, fc["yhat_upper"].values),
        val_mape=v_mape, val_rmse=v_rmse,
    )


# ── 4. ExponentialSmoothing (fallback) ────────────────────────────────────
def fit_ets(y: pd.Series, horizon: int) -> ModelResult:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    y_tr, _, y_val, _ = _train_val_split(y, pd.DataFrame(index=y.index))

    seasonal = "add" if len(y_tr) >= 24 else None
    model = ExponentialSmoothing(
        y_tr, trend="add", seasonal=seasonal,
        seasonal_periods=12 if seasonal else None,
    ).fit(optimized=True)

    val_pred = model.forecast(len(y_val))
    v_mape = mape(y_val.values, np.maximum(0, val_pred.values))
    v_rmse = rmse(y_val.values, np.maximum(0, val_pred.values))

    # Дообучаем на всей истории
    full_model = ExponentialSmoothing(
        y, trend="add", seasonal=seasonal,
        seasonal_periods=12 if seasonal else None,
    ).fit(optimized=True)
    fc = full_model.forecast(horizon).values

    # CI: ±1.28σ остатков (80%)
    resid_std = np.std(full_model.resid)
    z = 1.28
    ci_l = np.maximum(0, fc - z * resid_std)
    ci_h = fc + z * resid_std

    return ModelResult(
        name="ETS",
        forecast=np.maximum(0, fc),
        ci_lower=ci_l, ci_upper=ci_h,
        val_mape=v_mape, val_rmse=v_rmse,
        notes=f"seasonal={'add' if seasonal else 'none'}",
    )


# ══════════════════════════════════════════════════════════════════════════
# АНСАМБЛЬ
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class EnsembleResult:
    sku: str
    horizon: int
    forecast_dates: pd.DatetimeIndex
    forecast: np.ndarray
    ci_lower: np.ndarray
    ci_upper: np.ndarray
    ensemble_mape: float             # взвешенный MAPE ансамбля
    models: list[ModelResult]        # результаты отдельных моделей
    weights: dict[str, float]        # веса в ансамбле
    top_features: list[str]
    feature_importance: dict
    history_y: pd.Series             # для графика
    history_X: pd.DataFrame


def ensemble_forecast(
    sku_df: pd.DataFrame,
    target: str = "revenue_total",
    horizon: int = 3,
    future_exog: Optional[pd.DataFrame] = None,
) -> EnsembleResult:
    """
    Главная функция: принимает данные по одному SKU, возвращает EnsembleResult.

    sku_df: DataFrame с DatetimeIndex (monthly), содержит target и экзогенные столбцы.
    future_exog: DataFrame на horizon строк с плановыми значениями факторов.
                 Если None — используется последнее наблюдение (наивный прогноз факторов).
    """
    sku_id = sku_df.get("sku_id", pd.Series(["unknown"])).iloc[0]

    # Строим признаки
    feat_df = build_features(sku_df, target).dropna(subset=[target])
    feat_df = feat_df.select_dtypes(include=[np.number])

    if len(feat_df) < 6:
        raise ValueError(f"SKU {sku_id}: недостаточно данных ({len(feat_df)} мес)")

    # Отбор признаков
    feature_cols, imp = select_features(feat_df, target)
    X = feat_df[feature_cols].fillna(0)
    y = feat_df[target]

    # Строим future_X (плановые значения на horizon шагов)
    if future_exog is not None:
        future_X = build_features(
            pd.concat([sku_df.tail(6), future_exog]).drop_duplicates(), target
        ).iloc[-horizon:][feature_cols].fillna(0)
    else:
        # Fallback: скользящее среднее последних 3 точек
        last_vals = X.tail(3).mean()
        future_X = pd.DataFrame([last_vals] * horizon, columns=feature_cols)
        # Обновляем временные признаки
        last_date = y.index[-1]
        future_dates = pd.date_range(last_date, periods=horizon + 1, freq="MS")[1:]
        future_X.index = future_dates
        future_X["month"] = future_dates.month
        future_X["month_sin"] = np.sin(2 * np.pi * future_X["month"] / 12)
        future_X["month_cos"] = np.cos(2 * np.pi * future_X["month"] / 12)
        future_X["trend_idx"] = np.arange(len(feat_df), len(feat_df) + horizon)
        # Лаговые target-признаки
        if "target_lag1" in feature_cols:
            future_X["target_lag1"] = float(y.iloc[-1])
        if "target_ma3" in feature_cols:
            future_X["target_ma3"] = float(y.tail(3).mean())
        if "target_ma6" in feature_cols:
            future_X["target_ma6"] = float(y.tail(6).mean())

    # Запускаем модели
    models: list[ModelResult] = []

    # LightGBM — всегда
    try:
        models.append(fit_lgbm(y, X, horizon, future_X))
    except Exception as e:
        print(f"  LightGBM error: {e}")

    # SARIMAX — если история >= 24 мес
    try:
        r = fit_sarimax(y, X, horizon, future_X)
        if r: models.append(r)
    except Exception as e:
        print(f"  SARIMAX error: {e}")

    # Prophet
    try:
        r = fit_prophet(y, X, horizon, future_X)
        if r: models.append(r)
    except Exception as e:
        print(f"  Prophet error: {e}")

    # ETS — всегда как fallback
    try:
        models.append(fit_ets(y, horizon))
    except Exception as e:
        print(f"  ETS error: {e}")

    if not models:
        raise RuntimeError(f"SKU {sku_id}: все модели упали")

    # Веса = 1/MAPE, нормализованные
    raw_w = {}
    for m in models:
        if np.isfinite(m.val_mape) and m.val_mape > 0:
            raw_w[m.name] = 1.0 / m.val_mape
        else:
            raw_w[m.name] = 0.0

    total_w = sum(raw_w.values())
    if total_w == 0:
        weights = {m.name: 1/len(models) for m in models}
    else:
        weights = {name: w / total_w for name, w in raw_w.items()}

    # Ансамблевый прогноз
    ensemble_fc = np.zeros(horizon)
    ensemble_cl = np.zeros(horizon)
    ensemble_ch = np.zeros(horizon)
    for m in models:
        w = weights[m.name]
        ensemble_fc += w * m.forecast
        ensemble_cl += w * m.ci_lower
        ensemble_ch += w * m.ci_upper

    # MAPE ансамбля на валидации (взвешенный)
    ensemble_val_mape = sum(weights[m.name] * m.val_mape for m in models
                            if np.isfinite(m.val_mape))

    last_date = y.index[-1]
    forecast_dates = pd.date_range(last_date, periods=horizon + 1, freq="MS")[1:]

    return EnsembleResult(
        sku=str(sku_id),
        horizon=horizon,
        forecast_dates=forecast_dates,
        forecast=ensemble_fc,
        ci_lower=ensemble_cl,
        ci_upper=ensemble_ch,
        ensemble_mape=ensemble_val_mape,
        models=models,
        weights=weights,
        top_features=feature_cols[:10],
        feature_importance=imp,
        history_y=y,
        history_X=X,
    )
