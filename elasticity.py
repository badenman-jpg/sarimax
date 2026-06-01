"""
elasticity.py — Расчёт ценовой эластичности спроса из исторических данных
Метод: log-log регрессия  ln(Q) = a + b*ln(P) + контрольные переменные
b = коэффициент эластичности
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass
class ElasticityResult:
    elasticity: float          # коэффициент (отрицательный = норма)
    elasticity_ci: tuple       # (нижняя, верхняя) 95% доверительный интервал
    r_squared: float           # качество подгонки
    n_obs: int                 # число наблюдений
    significant: bool          # p-value < 0.05
    p_value: float
    method: str                # 'ols', 'iv', 'insufficient_data'
    price_range: tuple         # (min_price, max_price) в данных
    interpretation: str        # текстовая интерпретация


def calc_elasticity(
    df: pd.DataFrame,
    price_col: str = "avg_price",
    qty_col: str   = "qty",
    revenue_col: str = "revenue_total",
    controls: list = None,
) -> ElasticityResult:
    """
    Считает ценовую эластичность по месячным данным одного SKU.

    Если qty недоступен — использует revenue / price как прокси.
    controls: список дополнительных регрессоров (напр. ['ads_budget', 'stock'])
    """
    d = df.copy()

    # Восстанавливаем qty если нет
    if qty_col not in d.columns or d[qty_col].isna().all():
        if price_col in d.columns and revenue_col in d.columns:
            d[qty_col] = d[revenue_col] / d[price_col].replace(0, np.nan)
        else:
            return _insufficient("qty и revenue недоступны")

    # Убираем нули и NaN
    mask = (d[price_col] > 0) & (d[qty_col] > 0) & d[price_col].notna() & d[qty_col].notna()
    d = d[mask].copy()

    if len(d) < 8:
        return _insufficient(f"мало точек: {len(d)} (нужно ≥ 8)")

    # Проверяем вариацию цены
    cv_price = d[price_col].std() / d[price_col].mean()
    if cv_price < 0.01:
        return _insufficient("цена не варьируется (CV < 1%) — эластичность не вычислима")

    # Log-log регрессия через statsmodels OLS
    try:
        import statsmodels.api as sm

        ln_p = np.log(d[price_col])
        ln_q = np.log(d[qty_col])

        X = pd.DataFrame({"ln_price": ln_p})

        # Добавляем тренд (автокорреляция временного ряда)
        X["trend"] = np.arange(len(d))
        X["month_sin"] = np.sin(2 * np.pi * np.arange(len(d)) / 12)
        X["month_cos"] = np.cos(2 * np.pi * np.arange(len(d)) / 12)

        # Контрольные переменные
        if controls:
            for c in controls:
                if c in d.columns and d[c].notna().sum() > len(d) * 0.5:
                    X[f"ln_{c}"] = np.log(d[c].clip(lower=0.01))

        X = sm.add_constant(X)
        model = sm.OLS(ln_q, X).fit(cov_type="HC3")

        elast    = float(model.params["ln_price"])
        se       = float(model.bse["ln_price"])
        p_val    = float(model.pvalues["ln_price"])
        ci_low   = elast - 1.96 * se
        ci_high  = elast + 1.96 * se
        r2       = float(model.rsquared)
        sig      = p_val < 0.05

        return ElasticityResult(
            elasticity=round(elast, 3),
            elasticity_ci=(round(ci_low, 3), round(ci_high, 3)),
            r_squared=round(r2, 3),
            n_obs=len(d),
            significant=sig,
            p_value=round(p_val, 4),
            method="ols",
            price_range=(round(d[price_col].min(), 2), round(d[price_col].max(), 2)),
            interpretation=_interpret(elast, sig, p_val),
        )

    except Exception as e:
        return _insufficient(f"OLS ошибка: {e}")


def apply_elasticity(
    base_qty: float,
    base_price: float,
    new_price: float,
    elasticity: float,
) -> float:
    """
    Прогноз кол-ва продаж при новой цене через эластичность.
    Q_new = Q_base * (P_new / P_base) ^ elasticity
    """
    if base_price <= 0 or new_price <= 0:
        return base_qty
    ratio = new_price / base_price
    return max(0.0, base_qty * (ratio ** elasticity))


def _insufficient(reason: str) -> ElasticityResult:
    return ElasticityResult(
        elasticity=0.0, elasticity_ci=(0.0, 0.0),
        r_squared=0.0, n_obs=0, significant=False,
        p_value=1.0, method="insufficient_data",
        price_range=(0.0, 0.0),
        interpretation=f"Недостаточно данных: {reason}",
    )


def _interpret(e: float, sig: bool, p: float) -> str:
    if not sig:
        return f"Незначимо (p={p:.3f}). Связь цена→спрос статистически не подтверждена."
    if e < -2:
        return f"Высокоэластичный спрос (e={e:.2f}). Снижение цены на 10% → +{abs(e)*10:.0f}% продаж."
    if e < -1:
        return f"Эластичный спрос (e={e:.2f}). Снижение цены выгодно для выручки."
    if e < 0:
        return f"Неэластичный спрос (e={e:.2f}). Цена слабо влияет на продажи."
    return f"Положительная эластичность (e={e:.2f}). Возможен эффект Гиффена или недостаточно данных."
