"""
forecaster.py — Ансамблевый прогноз продаж
Поддерживает дневные данные с автоагрегацией в месячные.
Колонки входного файла (русские названия из тестового файла):
  Дата, SKU, Выручка/день, Продажи/день, Цена, Цена со скидкой маркетплейса,
  Цена без скидки, На складе, Рекламный бюджет, Отзывы, Рейтинг
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from typing import Optional
from dataclasses import dataclass, field

# ── Маппинг колонок (русские → внутренние имена) ──────────────────────────
COL_MAP = {
    'Дата':                           'date',
    'SKU':                            'sku_id',
    'Выручка/день':                   'revenue_total',
    'Продажи/день':                   'qty',
    'Цена':                           'avg_price',
    'Цена со скидкой маркетплейса':   'avg_price_discount',
    'Цена без скидки':                'avg_price_base',
    'На складе':                      'stock',
    'Рекламный бюджет':               'ads_budget',
    'Отзывы':                         'reviews',
    'Рейтинг':                        'rating',
    'Упущенная прибыль':              'lost_revenue',
}

# Также поддерживаем уже переименованные (английские) колонки
COL_MAP_EN = {v: v for v in COL_MAP.values()}


def load_and_prepare(file_path_or_buffer) -> pd.DataFrame:
    """
    Загружает файл (Excel или CSV), переименовывает колонки,
    агрегирует дневные данные в месячные.
    Возвращает DataFrame с колонками: sku_id, date (monthly), revenue_total, ...
    """
    if hasattr(file_path_or_buffer, 'name'):
        name = file_path_or_buffer.name
    else:
        name = str(file_path_or_buffer)

    if name.endswith('.csv'):
        df = pd.read_csv(file_path_or_buffer)
    else:
        df = pd.read_excel(file_path_or_buffer)

    # Переименовываем колонки
    rename = {}
    for col in df.columns:
        if col in COL_MAP:
            rename[col] = COL_MAP[col]
        elif col in COL_MAP_EN:
            rename[col] = COL_MAP_EN[col]
    df = df.rename(columns=rename)

    # Обязательные колонки
    if 'date' not in df.columns:
        raise ValueError(f"Не найдена колонка с датой. Есть: {list(df.columns)}")
    if 'revenue_total' not in df.columns:
        raise ValueError(f"Не найдена колонка выручки. Есть: {list(df.columns)}")

    df['date'] = pd.to_datetime(df['date'])

    # Если нет SKU — создаём один
    if 'sku_id' not in df.columns:
        df['sku_id'] = 'SKU_1'

    # Определяем частоту данных
    df_sorted = df.sort_values('date')
    median_diff = df_sorted['date'].diff().dt.days.median()
    is_daily = median_diff is not None and median_diff <= 7

    if is_daily:
        # Агрегируем в месячные
        agg_dict = {'revenue_total': 'sum'}
        for col in ['qty', 'lost_revenue', 'ads_budget']:
            if col in df.columns:
                agg_dict[col] = 'sum'
        for col in ['avg_price', 'avg_price_discount', 'avg_price_base', 'stock', 'rating']:
            if col in df.columns:
                agg_dict[col] = 'mean'
        for col in ['reviews']:
            if col in df.columns:
                agg_dict[col] = 'last'

        df = df.groupby(
            ['sku_id', pd.Grouper(key='date', freq='MS')]
        ).agg(agg_dict).reset_index()

    return df.sort_values(['sku_id', 'date']).reset_index(drop=True)


# ── Метрики ────────────────────────────────────────────────────────────────
def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true > 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


@dataclass
class ModelResult:
    name: str
    forecast: np.ndarray
    ci_lower: np.ndarray
    ci_upper: np.ndarray
    val_mape: float
    val_rmse: float
    feature_importance: dict = field(default_factory=dict)
    notes: str = ""


def build_features(df: pd.DataFrame, target: str = 'revenue_total') -> pd.DataFrame:
    d = df.copy()

    # Лаги рыночных переменных
    lag_cols = ['avg_price', 'avg_price_discount', 'avg_price_base',
                'stock', 'ads_budget', 'rating', 'reviews']
    for col in lag_cols:
        if col in d.columns:
            d[f'{col}_lag1'] = d[col].shift(1)
            d[f'{col}_lag2'] = d[col].shift(2)

    # Временные признаки
    d['month']     = d.index.month
    d['month_sin'] = np.sin(2 * np.pi * d['month'] / 12)
    d['month_cos'] = np.cos(2 * np.pi * d['month'] / 12)
    d['trend_idx'] = np.arange(len(d))
    d['is_q4']     = d['month'].isin([10, 11, 12]).astype(int)
    d['is_q1']     = d['month'].isin([1, 2, 3]).astype(int)

    # Лаги и скользящие целевой переменной
    if target in d.columns:
        d['target_lag1']  = d[target].shift(1)
        d['target_lag2']  = d[target].shift(2)
        d['target_lag3']  = d[target].shift(3)
        d['target_ma3']   = d[target].shift(1).rolling(3, min_periods=1).mean()
        d['target_ma6']   = d[target].shift(1).rolling(6, min_periods=1).mean()
        d['target_ma12']  = d[target].shift(1).rolling(12, min_periods=1).mean()
        d['target_yoy']   = d[target].shift(12)   # год назад

    return d


def select_features(df: pd.DataFrame, target: str, top_n: int = 12) -> tuple[list, dict]:
    from lightgbm import LGBMRegressor

    candidates = [c for c in df.columns
                  if c != target and not c.startswith('_')
                  and pd.api.types.is_numeric_dtype(df[c])
                  and df[c].notna().sum() > 5]

    if not candidates:
        return [], {}

    X = df[candidates].fillna(0)
    y = df[target]

    lgb = LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
    lgb.fit(X, y)

    imp = dict(zip(candidates, lgb.feature_importances_ / (lgb.feature_importances_.sum() + 1e-9)))
    selected = sorted(imp, key=lambda x: -imp[x])[:top_n]
    return selected, imp


def _train_val_split(y, X, val_months=3):
    n = len(y)
    n_val = min(val_months, max(1, n // 5))
    n_tr  = n - n_val
    return y.iloc[:n_tr], X.iloc[:n_tr], y.iloc[n_tr:], X.iloc[n_tr:]


# ── Модели ─────────────────────────────────────────────────────────────────

def fit_lgbm(y, X, horizon, future_X):
    from lightgbm import LGBMRegressor

    y_tr, X_tr, y_val, X_val = _train_val_split(y, X)

    model = LGBMRegressor(n_estimators=500, learning_rate=0.03,
                          num_leaves=15, min_child_samples=3,
                          subsample=0.8, colsample_bytree=0.8,
                          random_state=42, verbose=-1)
    model.fit(X_tr, np.log1p(y_tr))

    val_pred = np.expm1(model.predict(X_val))
    v_mape = mape(y_val.values, val_pred)
    v_rmse = rmse(y_val.values, val_pred)

    preds = []
    for i in range(horizon):
        row = future_X.iloc[[i]] if i < len(future_X) else future_X.iloc[[-1]]
        preds.append(max(0, float(np.expm1(model.predict(row.fillna(0))[0]))))

    # Quantile CI
    ci_l, ci_h = [], []
    for q, lst in [(0.1, ci_l), (0.9, ci_h)]:
        qm = LGBMRegressor(objective='quantile', alpha=q, n_estimators=300,
                           learning_rate=0.03, num_leaves=15, verbose=-1, random_state=0)
        qm.fit(X_tr, np.log1p(y_tr))
        for i in range(horizon):
            row = future_X.iloc[[i]] if i < len(future_X) else future_X.iloc[[-1]]
            lst.append(max(0, float(np.expm1(qm.predict(row.fillna(0))[0]))))

    imp = dict(zip(X_tr.columns,
                   model.feature_importances_ / (model.feature_importances_.sum() + 1e-9)))

    return ModelResult('LightGBM', np.array(preds), np.array(ci_l), np.array(ci_h),
                       v_mape, v_rmse, imp)


def fit_sarimax(y, X, horizon, future_X):
    if len(y) < 24:
        return None
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from pmdarima import auto_arima

    y_tr, X_tr, y_val, X_val = _train_val_split(y, X)
    y_tr_log = np.log1p(y_tr)

    try:
        auto = auto_arima(y_tr_log, exogenous=X_tr, seasonal=True, m=12,
                          max_p=2, max_q=2, max_P=1, max_Q=1, max_d=1,
                          stepwise=True, suppress_warnings=True,
                          error_action='ignore', information_criterion='bic')
        order, s_order = auto.order, auto.seasonal_order
    except Exception:
        order, s_order = (1, 1, 0), (0, 1, 0, 12)

    res = SARIMAX(y_tr_log, exog=X_tr, order=order, seasonal_order=s_order,
                  enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)

    val_pred = np.expm1(res.get_forecast(len(y_val), exog=X_val).predicted_mean.values)
    v_mape = mape(y_val.values, np.maximum(0, val_pred))

    X_fc = future_X.iloc[:horizon].fillna(0)
    fc = res.get_forecast(horizon, exog=X_fc)
    ci = fc.conf_int(alpha=0.2)

    return ModelResult('SARIMAX',
                       np.maximum(0, np.expm1(fc.predicted_mean.values)),
                       np.maximum(0, np.expm1(ci.iloc[:, 0].values)),
                       np.maximum(0, np.expm1(ci.iloc[:, 1].values)),
                       v_mape, rmse(y_val.values, np.maximum(0, val_pred)),
                       notes=f'order={order}')


def fit_ets(y, horizon):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    y_tr, _, y_val, _ = _train_val_split(y, pd.DataFrame(index=y.index))
    seasonal = 'add' if len(y_tr) >= 24 else None

    m = ExponentialSmoothing(y_tr, trend='add', seasonal=seasonal,
                             seasonal_periods=12 if seasonal else None).fit(optimized=True)
    val_pred = np.maximum(0, m.forecast(len(y_val)).values)
    v_mape = mape(y_val.values, val_pred)

    full = ExponentialSmoothing(y, trend='add', seasonal=seasonal,
                                seasonal_periods=12 if seasonal else None).fit(optimized=True)
    fc = full.forecast(horizon).values
    std = np.std(full.resid)
    return ModelResult('ETS', np.maximum(0, fc),
                       np.maximum(0, fc - 1.28 * std), fc + 1.28 * std,
                       v_mape, rmse(y_val.values, val_pred))


# ── Ансамбль ───────────────────────────────────────────────────────────────

@dataclass
class EnsembleResult:
    sku: str
    horizon: int
    forecast_dates: pd.DatetimeIndex
    forecast: np.ndarray
    ci_lower: np.ndarray
    ci_upper: np.ndarray
    ensemble_mape: float
    models: list
    weights: dict
    top_features: list
    feature_importance: dict
    history_y: pd.Series
    history_monthly: pd.DataFrame  # полная месячная таблица для графика


def ensemble_forecast(sku_df: pd.DataFrame,
                      target: str = 'revenue_total',
                      horizon: int = 3,
                      scenario: Optional[dict] = None) -> EnsembleResult:
    """
    sku_df: месячные данные по одному SKU с DatetimeIndex.
    scenario: dict с плановыми значениями {'avg_price': 350, 'ads_budget': 5000, ...}
    """
    sku_id = str(sku_df['sku_id'].iloc[0]) if 'sku_id' in sku_df.columns else 'SKU'

    feat_df = build_features(sku_df.set_index('date'), target).dropna(subset=[target])
    feat_df = feat_df[[c for c in feat_df.columns
                       if pd.api.types.is_numeric_dtype(feat_df[c])]]

    if len(feat_df) < 6:
        raise ValueError(f'Мало данных: {len(feat_df)} мес. Нужно минимум 6.')

    feature_cols, imp = select_features(feat_df, target)
    X = feat_df[feature_cols].fillna(0)
    y = feat_df[target]

    # Future X: скользящее среднее + сценарий пользователя
    last_date = y.index[-1]
    future_dates = pd.date_range(last_date, periods=horizon + 1, freq='MS')[1:]

    base_vals = X.tail(3).mean()
    future_X = pd.DataFrame([base_vals] * horizon, columns=feature_cols, index=future_dates)

    # Обновляем временные признаки
    future_X['month']     = future_dates.month
    future_X['month_sin'] = np.sin(2 * np.pi * future_dates.month / 12)
    future_X['month_cos'] = np.cos(2 * np.pi * future_dates.month / 12)
    future_X['trend_idx'] = np.arange(len(feat_df), len(feat_df) + horizon)
    future_X['is_q4']     = future_dates.month.isin([10, 11, 12]).astype(int)
    future_X['is_q1']     = future_dates.month.isin([1, 2, 3]).astype(int)
    if 'target_lag1' in feature_cols:
        future_X['target_lag1'] = float(y.iloc[-1])
    if 'target_ma3' in feature_cols:
        future_X['target_ma3']  = float(y.tail(3).mean())
    if 'target_ma6' in feature_cols:
        future_X['target_ma6']  = float(y.tail(6).mean())
    if 'target_ma12' in feature_cols:
        future_X['target_ma12'] = float(y.tail(12).mean())

    # Применяем сценарий
    if scenario:
        for k, v in scenario.items():
            lag_key = f'{k}_lag1'
            if lag_key in future_X.columns:
                future_X[lag_key] = v
            if k in future_X.columns:
                future_X[k] = v

    # Обучаем модели
    models = []
    for fn in [fit_lgbm, fit_ets]:
        try:
            models.append(fn(y, X, horizon, future_X))
        except Exception as e:
            print(f'  {fn.__name__} error: {e}')

    try:
        r = fit_sarimax(y, X, horizon, future_X)
        if r: models.append(r)
    except Exception as e:
        print(f'  SARIMAX error: {e}')

    if not models:
        raise RuntimeError('Все модели упали')

    # Веса
    raw_w = {m.name: (1.0 / m.val_mape if np.isfinite(m.val_mape) and m.val_mape > 0 else 0)
             for m in models}
    total = sum(raw_w.values())
    weights = {k: (v / total if total > 0 else 1/len(models)) for k, v in raw_w.items()}

    fc  = sum(weights[m.name] * m.forecast   for m in models)
    cl  = sum(weights[m.name] * m.ci_lower   for m in models)
    ch  = sum(weights[m.name] * m.ci_upper   for m in models)
    ens_mape = sum(weights[m.name] * m.val_mape for m in models
                   if np.isfinite(m.val_mape))

    return EnsembleResult(
        sku=sku_id, horizon=horizon,
        forecast_dates=future_dates,
        forecast=fc, ci_lower=cl, ci_upper=ch,
        ensemble_mape=ens_mape,
        models=models, weights=weights,
        top_features=feature_cols[:10],
        feature_importance=imp,
        history_y=y,
        history_monthly=sku_df.set_index('date'),
    )


# ── Обёртка с force_model ──────────────────────────────────────────────────
_orig_ensemble = ensemble_forecast

def ensemble_forecast(sku_df, target='revenue_total', horizon=3,
                      scenario=None, force_model=None):
    """
    force_model: 'LightGBM' | 'ETS' | 'SARIMAX' | None (ансамбль)
    """
    result = _orig_ensemble(sku_df, target=target, horizon=horizon, scenario=scenario)

    if force_model and force_model != "Ансамбль":
        # Находим нужную модель
        chosen = next((m for m in result.models if m.name == force_model), None)
        if chosen:
            result.forecast  = chosen.forecast
            result.ci_lower  = chosen.ci_lower
            result.ci_upper  = chosen.ci_upper
            result.ensemble_mape = chosen.val_mape
            result.weights = {force_model: 1.0}
    return result
