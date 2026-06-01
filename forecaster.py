"""
forecaster.py — Ансамблевый прогноз продаж
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass, field

COL_MAP = {
    'Дата':'date','SKU':'sku_id','Выручка/день':'revenue_total',
    'Продажи/день':'qty','Цена':'avg_price',
    'Цена со скидкой маркетплейса':'avg_price_discount',
    'Цена без скидки':'avg_price_base','На складе':'stock',
    'Рекламный бюджет':'ads_budget','Отзывы':'reviews',
    'Рейтинг':'rating','Упущенная прибыль':'lost_revenue',
}

def load_and_prepare(path_or_buf) -> pd.DataFrame:
    name = getattr(path_or_buf, 'name', str(path_or_buf))
    df = pd.read_csv(path_or_buf) if str(name).endswith('.csv') else pd.read_excel(path_or_buf)
    df = df.rename(columns={c: COL_MAP[c] for c in df.columns if c in COL_MAP})
    if 'date' not in df.columns:
        raise ValueError(f"Нет колонки даты. Колонки: {list(df.columns)}")
    if 'revenue_total' not in df.columns:
        raise ValueError(f"Нет колонки выручки. Колонки: {list(df.columns)}")
    df['date'] = pd.to_datetime(df['date'])
    if 'sku_id' not in df.columns:
        df['sku_id'] = 'SKU_1'
    # Агрегация в месячные если данные дневные
    med = df.sort_values('date')['date'].diff().dt.days.median()
    if med is not None and med <= 7:
        agg = {'revenue_total':'sum'}
        for c in ['qty','lost_revenue','ads_budget']:
            if c in df.columns: agg[c]='sum'
        for c in ['avg_price','avg_price_discount','avg_price_base','stock','rating']:
            if c in df.columns: agg[c]='mean'
        if 'reviews' in df.columns: agg['reviews']='last'
        df = df.groupby(['sku_id', pd.Grouper(key='date',freq='MS')]).agg(agg).reset_index()
    return df.sort_values(['sku_id','date']).reset_index(drop=True)

def _mape(yt, yp):
    m = yt > 0
    return float(np.mean(np.abs((yt[m]-yp[m])/yt[m]))*100) if m.sum() else float('nan')

def _rmse(yt, yp):
    return float(np.sqrt(np.mean((yt-yp)**2)))

@dataclass
class ModelResult:
    name: str
    forecast: np.ndarray
    ci_lower: np.ndarray
    ci_upper: np.ndarray
    val_mape: float
    val_rmse: float
    feature_importance: Dict = field(default_factory=dict)
    notes: str = ""

@dataclass
class EnsembleResult:
    sku: str
    horizon: int
    forecast_dates: pd.DatetimeIndex
    forecast: np.ndarray
    ci_lower: np.ndarray
    ci_upper: np.ndarray
    ensemble_mape: float
    models: List
    weights: Dict
    top_features: List
    feature_importance: Dict
    history_y: pd.Series
    history_monthly: pd.DataFrame

def _build_features(df: pd.DataFrame, target: str) -> pd.DataFrame:
    d = df.copy()
    for c in ['avg_price','avg_price_discount','avg_price_base','stock','ads_budget','rating','reviews']:
        if c in d.columns:
            d[f'{c}_lag1'] = d[c].shift(1)
            d[f'{c}_lag2'] = d[c].shift(2)
    d['month']     = d.index.month
    d['month_sin'] = np.sin(2*np.pi*d['month']/12)
    d['month_cos'] = np.cos(2*np.pi*d['month']/12)
    d['trend_idx'] = np.arange(len(d))
    d['is_q4']     = d['month'].isin([10,11,12]).astype(int)
    d['is_q1']     = d['month'].isin([1,2,3]).astype(int)
    if target in d.columns:
        d['t_lag1']  = d[target].shift(1)
        d['t_lag2']  = d[target].shift(2)
        d['t_lag3']  = d[target].shift(3)
        d['t_ma3']   = d[target].shift(1).rolling(3,  min_periods=1).mean()
        d['t_ma6']   = d[target].shift(1).rolling(6,  min_periods=1).mean()
        d['t_ma12']  = d[target].shift(1).rolling(12, min_periods=1).mean()
        d['t_yoy']   = d[target].shift(12)
    return d

def _select_features(df, target, top_n=12):
    from lightgbm import LGBMRegressor
    cands = [c for c in df.columns if c != target
             and pd.api.types.is_numeric_dtype(df[c])
             and df[c].notna().sum() > 5]
    if not cands: return [], {}
    X = df[cands].fillna(0); y = df[target]
    m = LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
    m.fit(X, y)
    imp = dict(zip(cands, m.feature_importances_/(m.feature_importances_.sum()+1e-9)))
    return sorted(imp, key=lambda x: -imp[x])[:top_n], imp

def _split(y, X, val=3):
    n = len(y); nv = min(val, max(1, n//5)); nt = n-nv
    return y.iloc[:nt], X.iloc[:nt], y.iloc[nt:], X.iloc[nt:]

def _make_future(X_train_cols, X_tail3, horizon, future_dates, y, feature_cols, scenario):
    """Строит future_X с ТОЧНО теми же колонками что X_train."""
    base = {c: float(X_tail3[c]) if c in X_tail3.index else 0.0 for c in X_train_cols}
    rows = []
    for i, dt in enumerate(future_dates):
        r = base.copy()
        # Временные признаки
        if 'month'     in r: r['month']     = dt.month
        if 'month_sin' in r: r['month_sin'] = float(np.sin(2*np.pi*dt.month/12))
        if 'month_cos' in r: r['month_cos'] = float(np.cos(2*np.pi*dt.month/12))
        if 'trend_idx' in r: r['trend_idx'] = float(len(y) + i)
        if 'is_q4'     in r: r['is_q4']     = float(dt.month in [10,11,12])
        if 'is_q1'     in r: r['is_q1']     = float(dt.month in [1,2,3])
        # Лаги цели
        if 't_lag1' in r: r['t_lag1'] = float(y.iloc[-1])
        if 't_ma3'  in r: r['t_ma3']  = float(y.tail(3).mean())
        if 't_ma6'  in r: r['t_ma6']  = float(y.tail(6).mean())
        if 't_ma12' in r: r['t_ma12'] = float(y.tail(12).mean())
        # Сценарий
        if scenario:
            for k, v in scenario.items():
                if f'{k}_lag1' in r: r[f'{k}_lag1'] = float(v)
                if k in r:           r[k]            = float(v)
        rows.append(r)
    df = pd.DataFrame(rows, index=future_dates, columns=X_train_cols)
    return df.fillna(0)

# ── Модели ────────────────────────────────────────────────────────────────

def _fit_lgbm(y, X, horizon, future_X) -> ModelResult:
    from lightgbm import LGBMRegressor
    y_tr, X_tr, y_val, X_val = _split(y, X)
    # future_X уже выровнен по колонкам X в ensemble_forecast
    m = LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=15,
                      min_child_samples=3, subsample=0.8, colsample_bytree=0.8,
                      random_state=42, verbose=-1)
    m.fit(X_tr, np.log1p(y_tr))
    vp = np.expm1(m.predict(X_val))
    preds = [max(0, float(np.expm1(m.predict(future_X.iloc[[i]])[0]))) for i in range(horizon)]
    ci_l, ci_h = [], []
    for q, lst in [(0.1, ci_l),(0.9, ci_h)]:
        qm = LGBMRegressor(objective='quantile', alpha=q, n_estimators=200,
                           learning_rate=0.03, num_leaves=15, verbose=-1, random_state=0)
        qm.fit(X_tr, np.log1p(y_tr))
        for i in range(horizon):
            lst.append(max(0, float(np.expm1(qm.predict(future_X.iloc[[i]])[0]))))
    imp = dict(zip(X_tr.columns, m.feature_importances_/(m.feature_importances_.sum()+1e-9)))
    return ModelResult('LightGBM', np.array(preds), np.array(ci_l), np.array(ci_h),
                       _mape(y_val.values, vp), _rmse(y_val.values, vp), imp)

def _fit_ets(y, X, horizon, future_X) -> ModelResult:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    y_tr, _, y_val, _ = _split(y, pd.DataFrame(index=y.index))
    seas = 'add' if len(y_tr) >= 24 else None
    m = ExponentialSmoothing(y_tr, trend='add', seasonal=seas,
                             seasonal_periods=12 if seas else None).fit(optimized=True)
    vp = np.maximum(0, m.forecast(len(y_val)).values)
    full = ExponentialSmoothing(y, trend='add', seasonal=seas,
                                seasonal_periods=12 if seas else None).fit(optimized=True)
    fc = full.forecast(horizon).values
    std = np.std(full.resid)
    return ModelResult('ETS', np.maximum(0, fc),
                       np.maximum(0, fc-1.28*std), fc+1.28*std,
                       _mape(y_val.values, vp), _rmse(y_val.values, vp))

def _fit_sarimax(y, X, horizon, future_X) -> Optional[ModelResult]:
    if len(y) < 24: return None
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from pmdarima import auto_arima
    y_tr, X_tr, y_val, X_val = _split(y, X)
    try:
        auto = auto_arima(np.log1p(y_tr), exogenous=X_tr, seasonal=True, m=12,
                          max_p=2,max_q=2,max_P=1,max_Q=1,max_d=1,
                          stepwise=True,suppress_warnings=True,
                          error_action='ignore',information_criterion='bic')
        order, s_order = auto.order, auto.seasonal_order
    except Exception:
        order, s_order = (1,1,0),(0,1,0,12)
    res = SARIMAX(np.log1p(y_tr), exog=X_tr, order=order, seasonal_order=s_order,
                  enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    vp = np.maximum(0, np.expm1(res.get_forecast(len(y_val), exog=X_val).predicted_mean.values))
    fc = res.get_forecast(horizon, exog=future_X)
    ci = fc.conf_int(alpha=0.2)
    return ModelResult('SARIMAX',
                       np.maximum(0, np.expm1(fc.predicted_mean.values)),
                       np.maximum(0, np.expm1(ci.iloc[:,0].values)),
                       np.maximum(0, np.expm1(ci.iloc[:,1].values)),
                       _mape(y_val.values, vp), _rmse(y_val.values, vp),
                       notes=f'order={order}')

# ── Ансамбль ─────────────────────────────────────────────────────────────

def ensemble_forecast(sku_df: pd.DataFrame, target: str = 'revenue_total',
                      horizon: int = 3, scenario: Optional[Dict] = None,
                      force_model: Optional[str] = None) -> EnsembleResult:
    sku_id = str(sku_df['sku_id'].iloc[0]) if 'sku_id' in sku_df.columns else 'SKU'

    # Берём только нужные колонки (без макро — они могут иметь разное число)
    base_cols = ['date','sku_id','revenue_total','qty','avg_price','avg_price_discount',
                 'avg_price_base','stock','ads_budget','reviews','rating','lost_revenue']
    use_cols = [c for c in base_cols if c in sku_df.columns]
    feat_df = _build_features(sku_df[use_cols].set_index('date'), target).dropna(subset=[target])
    feat_df = feat_df[[c for c in feat_df.columns if pd.api.types.is_numeric_dtype(feat_df[c])]]

    if len(feat_df) < 6:
        raise ValueError(f'Мало данных: {len(feat_df)} мес. Нужно минимум 6.')

    feature_cols, imp = _select_features(feat_df, target)
    X = feat_df[feature_cols].fillna(0)
    y = feat_df[target]

    last_date   = y.index[-1]
    future_dates = pd.date_range(last_date, periods=horizon+1, freq='MS')[1:]
    X_tail3     = X.tail(3).mean()

    # Строим future_X с ТЕМИ ЖЕ колонками что X (feature_cols)
    future_X = _make_future(feature_cols, X_tail3, horizon, future_dates, y, feature_cols, scenario)

    # Запускаем модели
    models = []
    for fn in [_fit_lgbm, _fit_ets]:
        try: models.append(fn(y, X, horizon, future_X))
        except Exception as e: print(f'{fn.__name__} error: {e}')
    try:
        r = _fit_sarimax(y, X, horizon, future_X)
        if r: models.append(r)
    except Exception as e:
        print(f'SARIMAX error: {e}')

    if not models:
        raise RuntimeError('Все модели упали')

    # Если запрошена конкретная модель
    if force_model and force_model != 'Ансамбль':
        chosen = next((m for m in models if m.name == force_model), None)
        if chosen:
            models = [chosen]

    # Веса 1/MAPE
    raw_w = {m.name: (1.0/m.val_mape if np.isfinite(m.val_mape) and m.val_mape > 0 else 0)
             for m in models}
    total = sum(raw_w.values())
    weights = {k: (v/total if total > 0 else 1/len(models)) for k,v in raw_w.items()}

    fc  = sum(weights[m.name]*m.forecast  for m in models)
    cl  = sum(weights[m.name]*m.ci_lower  for m in models)
    ch  = sum(weights[m.name]*m.ci_upper  for m in models)
    ens = sum(weights[m.name]*m.val_mape  for m in models if np.isfinite(m.val_mape))

    return EnsembleResult(
        sku=sku_id, horizon=horizon, forecast_dates=future_dates,
        forecast=fc, ci_lower=cl, ci_upper=ch, ensemble_mape=ens,
        models=models, weights=weights, top_features=feature_cols[:10],
        feature_importance=imp, history_y=y,
        history_monthly=sku_df.set_index('date'),
    )
