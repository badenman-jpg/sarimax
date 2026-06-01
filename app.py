"""
app.py — Streamlit UI для ансамблевого прогноза продаж
Запуск: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

from forecaster import ensemble_forecast, EnsembleResult

# ── Конфиг страницы ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sales Forecast",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card {
    background: #1C2030; border-radius: 12px;
    padding: 16px 20px; margin: 4px 0;
}
.metric-label { font-size: 11px; color: #5A6380; text-transform: uppercase; letter-spacing: .08em; }
.metric-value { font-size: 28px; font-weight: 800; color: #E8ECF4; }
.metric-delta { font-size: 13px; margin-top: 2px; }
.model-tag {
    display: inline-block; padding: 3px 10px; border-radius: 20px;
    font-size: 11px; font-weight: 700; margin: 2px;
}
.quality-good  { background: rgba(34,197,94,.15); color: #22C55E; }
.quality-ok    { background: rgba(234,179,8,.15);  color: #EAB308; }
.quality-bad   { background: rgba(239,68,68,.15);  color: #EF4444; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# САЙДБАР — загрузка и настройки
# ══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📈 Sales Forecast")
    st.markdown("---")

    uploaded = st.file_uploader(
        "Загрузите данные (Excel/CSV)",
        type=["xlsx", "xls", "csv"],
        help="Нужен файл с колонками: sku_id, date, revenue_total (+ опциональные факторы)"
    )

    st.markdown("#### Настройки прогноза")
    horizon = st.slider("Горизонт (месяцев)", 1, 12, 3)
    target_col = st.text_input("Целевая переменная", "revenue_total")

    st.markdown("---")
    st.markdown("#### Плановые значения факторов")
    st.caption("Если оставить 0 — используется среднее последних 3 мес.")

    plan_price    = st.number_input("Плановая цена (0 = авто)", min_value=0.0, value=0.0, step=10.0)
    plan_discount = st.number_input("Скидка % (0 = авто)",      min_value=0.0, max_value=100.0, value=0.0)
    plan_ads      = st.number_input("Рекл. бюджет ₽ (0 = авто)", min_value=0.0, value=0.0, step=1000.0)
    plan_stock    = st.number_input("Остаток на складе (0 = авто)", min_value=0, value=0, step=100)
    plan_promo    = st.selectbox("Промо-акция", ["Нет", "Стандартная", "Агрессивная"])

    promo_map = {"Нет": 0, "Стандартная": 50, "Агрессивная": 100}

    run_btn = st.button("🚀 Рассчитать прогноз", type="primary", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# ЗАГРУЗКА ДАННЫХ
# ══════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_data(file) -> pd.DataFrame:
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["sku_id", "date"])
    return df


def mape_quality(mape_val: float) -> tuple[str, str]:
    if mape_val < 15:   return "Высокая", "quality-good"
    if mape_val < 30:   return "Средняя", "quality-ok"
    return "Низкая", "quality-bad"


# ══════════════════════════════════════════════════════════════════════════
# ГРАФИКИ
# ══════════════════════════════════════════════════════════════════════════
def make_forecast_chart(result: EnsembleResult) -> go.Figure:
    fig = go.Figure()

    # История
    fig.add_trace(go.Scatter(
        x=result.history_y.index, y=result.history_y.values,
        name="Факт", line=dict(color="#4F8EF7", width=2),
        hovertemplate="%{x|%b %Y}: <b>%{y:,.0f}</b><extra></extra>",
    ))

    # CI
    fig.add_trace(go.Scatter(
        x=list(result.forecast_dates) + list(result.forecast_dates[::-1]),
        y=list(result.ci_upper) + list(result.ci_lower[::-1]),
        fill="toself",
        fillcolor="rgba(124,92,252,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name="80% доверит. интервал",
        hoverinfo="skip",
    ))

    # Прогноз
    # Соединяем с последней точкой истории
    x_fc = [result.history_y.index[-1]] + list(result.forecast_dates)
    y_fc = [result.history_y.iloc[-1]]  + list(result.forecast)

    fig.add_trace(go.Scatter(
        x=x_fc, y=y_fc,
        name="Прогноз (ансамбль)",
        line=dict(color="#EF4444", width=2.5, dash="dash"),
        mode="lines+markers",
        marker=dict(size=8, color="#EF4444", symbol="circle"),
        hovertemplate="%{x|%b %Y}: <b>%{y:,.0f}</b><extra></extra>",
    ))

    # Отдельные модели (полупрозрачные)
    model_colors = {"LightGBM": "#22C55E", "SARIMAX": "#F97316",
                    "Prophet": "#EAB308", "ETS": "#06B6D4"}
    for m in result.models:
        if m.name == "ETS" and len(result.models) > 1:
            continue  # не показываем ETS если есть лучшие
        x_m = [result.history_y.index[-1]] + list(result.forecast_dates)
        y_m = [result.history_y.iloc[-1]]  + list(m.forecast)
        fig.add_trace(go.Scatter(
            x=x_m, y=y_m,
            name=f"{m.name} (MAPE {m.val_mape:.1f}%)",
            line=dict(color=model_colors.get(m.name, "#999"), width=1, dash="dot"),
            opacity=0.6,
            visible="legendonly",
        ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="system-ui", color="#9AA3BE", size=12),
        legend=dict(
            bgcolor="rgba(28,32,48,0.8)", bordercolor="rgba(255,255,255,.07)",
            borderwidth=1, x=0, y=1.02, orientation="h",
        ),
        xaxis=dict(gridcolor="rgba(90,99,128,.12)", showline=False),
        yaxis=dict(gridcolor="rgba(90,99,128,.12)", showline=False,
                   title="Выручка, ₽"),
        hovermode="x unified",
        margin=dict(l=10, r=10, t=10, b=10),
        height=380,
    )
    return fig


def make_importance_chart(result: EnsembleResult) -> go.Figure:
    imp = result.feature_importance
    top = sorted(imp.items(), key=lambda x: -x[1])[:12]
    labels = [t[0] for t in top]
    values = [t[1] * 100 for t in top]

    colors = ["#4F8EF7" if v > 10 else "#7C5CFC" if v > 5 else "#5A6380" for v in values]

    fig = go.Figure(go.Bar(
        x=values[::-1], y=labels[::-1],
        orientation="h",
        marker_color=colors[::-1],
        hovertemplate="%{y}: <b>%{x:.1f}%</b><extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#9AA3BE", size=11),
        xaxis=dict(gridcolor="rgba(90,99,128,.12)", title="Важность, %"),
        yaxis=dict(showgrid=False),
        margin=dict(l=10, r=10, t=10, b=10),
        height=320,
    )
    return fig


def make_model_comparison(result: EnsembleResult) -> go.Figure:
    names  = [m.name for m in result.models] + ["Ансамбль"]
    mapes  = [m.val_mape for m in result.models] + [result.ensemble_mape]
    colors = ["#22C55E" if v < 15 else "#EAB308" if v < 30 else "#EF4444" for v in mapes]
    colors[-1] = "#4F8EF7"  # ансамбль всегда синий

    fig = go.Figure(go.Bar(
        x=names, y=mapes,
        marker_color=colors,
        text=[f"{v:.1f}%" for v in mapes],
        textposition="outside",
        hovertemplate="%{x}: MAPE <b>%{y:.1f}%</b><extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#9AA3BE", size=12),
        yaxis=dict(gridcolor="rgba(90,99,128,.12)", title="MAPE (чем меньше — лучше)"),
        xaxis=dict(showgrid=False),
        margin=dict(l=10, r=10, t=10, b=10),
        height=260,
        showlegend=False,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════
# ГЛАВНЫЙ ЭКРАН
# ══════════════════════════════════════════════════════════════════════════
if uploaded is None:
    st.markdown("""
    <div style="text-align:center; padding: 80px 20px;">
        <div style="font-size:64px; margin-bottom:20px;">📈</div>
        <h1 style="font-size:32px; font-weight:900;">Sales Forecast</h1>
        <p style="color:#5A6380; font-size:16px; max-width:500px; margin:0 auto 32px;">
        Ансамблевый прогноз продаж на 1–12 месяцев.<br>
        LightGBM + SARIMAX + Prophet + ETS.<br>
        Загрузите Excel или CSV с историей продаж.
        </p>
        <div style="color:#5A6380; font-size:13px;">
        Обязательные колонки: <code>sku_id</code>, <code>date</code>, <code>revenue_total</code>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📋 Формат входного файла"):
        st.dataframe(pd.DataFrame({
            "sku_id":             ["SKU001", "SKU001", "SKU001"],
            "date":               ["2024-01-01", "2024-02-01", "2024-03-01"],
            "revenue_total":      [150000, 180000, 165000],
            "avg_price_wb":       [388, 388, 350],
            "ads_budget_wb":      [5000, 7000, 5000],
            "stock_wb":           [500, 400, 600],
            "planned_discount_next_month": [10, 0, 15],
            "promo_plan_next_month":       [1, 0, 1],
            "seasonality_index":  [0.9, 1.0, 1.1],
        }), use_container_width=True)
    st.stop()

# Загружаем данные
with st.spinner("Загружаем данные..."):
    df = load_data(uploaded)

skus = df["sku_id"].unique().tolist()
st.markdown(f"**Загружено:** {len(df):,} строк · {len(skus)} SKU · "
            f"{df['date'].min().strftime('%b %Y')} — {df['date'].max().strftime('%b %Y')}")

# Выбор SKU
selected_sku = st.selectbox("Выберите SKU для анализа", skus)

if not run_btn:
    st.info("👈 Настройте параметры в боковой панели и нажмите **Рассчитать прогноз**")
    st.stop()

# ── Запуск прогноза ─────────────────────────────────────────────────────────
sku_df = df[df["sku_id"] == selected_sku].copy().set_index("date").sort_index()

# Строим плановые экзогенные если пользователь задал значения
plan_rows = []
last_date = sku_df.index[-1]
for i in range(horizon):
    future_date = pd.date_range(last_date, periods=i + 2, freq="MS")[-1]
    row = {"date": future_date, "sku_id": selected_sku}
    if plan_price    > 0: row["avg_price_wb"] = plan_price
    if plan_discount > 0: row["planned_discount_next_month"] = plan_discount
    if plan_ads      > 0: row["ads_budget_wb"] = plan_ads
    if plan_stock    > 0: row["stock_wb"] = plan_stock
    row["promo_plan_next_month"] = promo_map[plan_promo]
    plan_rows.append(row)

future_exog_df = pd.DataFrame(plan_rows).set_index("date") if any(
    [plan_price > 0, plan_discount > 0, plan_ads > 0, plan_stock > 0, plan_promo != "Нет"]
) else None

with st.spinner("Обучаем модели и строим прогноз..."):
    try:
        result = ensemble_forecast(
            sku_df,
            target=target_col,
            horizon=horizon,
            future_exog=future_exog_df,
        )
    except Exception as e:
        st.error(f"Ошибка при расчёте прогноза: {e}")
        st.stop()

# ══════════════════════════════════════════════════════════════════════════
# ВЫВОД РЕЗУЛЬТАТОВ
# ══════════════════════════════════════════════════════════════════════════
quality_label, quality_class = mape_quality(result.ensemble_mape)

# KPI-строка
c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    total_fc = result.forecast.sum()
    st.markdown(f"""<div class="metric-card">
    <div class="metric-label">Прогноз выручки</div>
    <div class="metric-value">{total_fc:,.0f} ₽</div>
    <div class="metric-delta" style="color:#5A6380;">за {horizon} мес.</div>
    </div>""", unsafe_allow_html=True)

with c2:
    history_same = result.history_y.tail(horizon).sum()
    delta_pct = (total_fc / history_same - 1) * 100 if history_same > 0 else 0
    delta_color = "#22C55E" if delta_pct >= 0 else "#EF4444"
    st.markdown(f"""<div class="metric-card">
    <div class="metric-label">Δ vs предыдущий период</div>
    <div class="metric-value" style="color:{delta_color};">{delta_pct:+.1f}%</div>
    <div class="metric-delta" style="color:#5A6380;">{history_same:,.0f} ₽ факт</div>
    </div>""", unsafe_allow_html=True)

with c3:
    st.markdown(f"""<div class="metric-card">
    <div class="metric-label">Точность модели</div>
    <div class="metric-value"><span class="model-tag {quality_class}">{quality_label}</span></div>
    <div class="metric-delta" style="color:#5A6380;">MAPE {result.ensemble_mape:.1f}%</div>
    </div>""", unsafe_allow_html=True)

with c4:
    best_model = min(result.models, key=lambda m: m.val_mape if np.isfinite(m.val_mape) else 999)
    st.markdown(f"""<div class="metric-card">
    <div class="metric-label">Лучшая модель</div>
    <div class="metric-value" style="font-size:20px;">{best_model.name}</div>
    <div class="metric-delta" style="color:#5A6380;">MAPE {best_model.val_mape:.1f}%</div>
    </div>""", unsafe_allow_html=True)

with c5:
    n_models = len(result.models)
    model_names = " · ".join([m.name for m in result.models])
    st.markdown(f"""<div class="metric-card">
    <div class="metric-label">Моделей в ансамбле</div>
    <div class="metric-value">{n_models}</div>
    <div class="metric-delta" style="color:#5A6380;font-size:10px;">{model_names}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("")

# Главный график
st.markdown("#### 📊 Прогноз продаж")
st.plotly_chart(make_forecast_chart(result), use_container_width=True)

# Таблица прогноза + скачать
fc_table = pd.DataFrame({
    "Месяц": result.forecast_dates.strftime("%Y-%m"),
    "Прогноз, ₽": result.forecast.round(0).astype(int),
    "Нижняя граница (80%)": result.ci_lower.round(0).astype(int),
    "Верхняя граница (80%)": result.ci_upper.round(0).astype(int),
})

col_tbl, col_dl = st.columns([3, 1])
with col_tbl:
    st.dataframe(fc_table, use_container_width=True, hide_index=True)
with col_dl:
    csv_buf = io.StringIO()
    fc_table.to_csv(csv_buf, index=False)
    st.download_button(
        "⬇️ Скачать прогноз CSV",
        csv_buf.getvalue().encode("utf-8-sig"),
        file_name=f"forecast_{selected_sku}.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.markdown("")

# Нижняя строка — веса моделей + важность признаков
col_l, col_m, col_r = st.columns([1, 1, 1])

with col_l:
    st.markdown("#### 🤖 Сравнение моделей")
    st.plotly_chart(make_model_comparison(result), use_container_width=True)

with col_m:
    st.markdown("#### ⚖️ Веса в ансамбле")
    weights_df = pd.DataFrame([
        {"Модель": k, "Вес": f"{v*100:.1f}%",
         "MAPE": f"{next(m.val_mape for m in result.models if m.name==k):.1f}%"}
        for k, v in sorted(result.weights.items(), key=lambda x: -x[1])
    ])
    st.dataframe(weights_df, use_container_width=True, hide_index=True)

    st.markdown("**Топ-признаки:**")
    for feat in result.top_features[:6]:
        imp_val = result.feature_importance.get(feat, 0) * 100
        st.markdown(f"- `{feat}` — {imp_val:.1f}%")

with col_r:
    st.markdown("#### 🔍 Важность признаков")
    st.plotly_chart(make_importance_chart(result), use_container_width=True)

# Подсказки по качеству
st.markdown("")
if result.ensemble_mape > 30:
    st.warning(
        "⚠️ MAPE > 30% — прогноз ненадёжен. Возможные причины: "
        "мало истории (< 12 мес), резкие изменения в продажах, "
        "нестационарный ряд. Попробуйте добавить больше факторов."
    )
elif result.ensemble_mape < 15:
    st.success(f"✅ Высокая точность прогноза (MAPE {result.ensemble_mape:.1f}%).")
