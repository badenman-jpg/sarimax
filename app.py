"""
app.py — Streamlit UI для прогноза продаж
Запуск: streamlit run app.py
"""
import warnings; warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io

from forecaster import load_and_prepare, ensemble_forecast, EnsembleResult

st.set_page_config(page_title="Sales Forecast", page_icon="📈", layout="wide")

st.markdown("""<style>
.block-container { padding-top: 1.5rem; }
.kpi { background:#1C2030; border-radius:12px; padding:16px 18px; margin:4px 0; }
.kpi-label { font-size:10px; color:#5A6380; text-transform:uppercase; letter-spacing:.08em; }
.kpi-value { font-size:26px; font-weight:900; color:#E8ECF4; line-height:1.1; }
.kpi-sub   { font-size:12px; color:#5A6380; margin-top:2px; }
.good { color:#22C55E !important; } .warn { color:#EAB308 !important; } .bad { color:#EF4444 !important; }
</style>""", unsafe_allow_html=True)


# ── Сайдбар ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 Sales Forecast")
    st.caption("Ансамблевый прогноз: LightGBM + ETS + SARIMAX")
    st.divider()

    uploaded = st.file_uploader("Загрузите файл (Excel/CSV)",
                                 type=["xlsx","xls","csv"])

    st.markdown("#### Горизонт прогноза")
    horizon = st.slider("Месяцев вперёд", 1, 12, 3)

    st.markdown("#### Сценарий (оставьте 0 = авто)")
    sc_price    = st.number_input("Цена продажи",    min_value=0.0, value=0.0, step=10.0)
    sc_discount = st.number_input("Цена со скидкой", min_value=0.0, value=0.0, step=10.0)
    sc_ads      = st.number_input("Рекл. бюджет ₽",  min_value=0.0, value=0.0, step=500.0)
    sc_stock    = st.number_input("Остаток на складе",min_value=0,   value=0,   step=50)

    run = st.button("🚀 Рассчитать прогноз", type="primary", use_container_width=True)


# ── Загрузка ───────────────────────────────────────────────────────────────
if uploaded is None:
    st.markdown("""
    <div style="text-align:center;padding:60px 20px;">
      <div style="font-size:56px;margin-bottom:16px;">📈</div>
      <h2>Sales Forecast</h2>
      <p style="color:#5A6380;max-width:480px;margin:0 auto 24px;">
        Загрузите Excel или CSV с историей продаж.<br>
        Поддерживаются дневные и месячные данные.
      </p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📋 Поддерживаемые колонки"):
        st.markdown("""
| Колонка | Описание |
|---|---|
| `Дата` / `date` | Дата (обязательно) |
| `SKU` / `sku_id` | Артикул |
| `Выручка/день` / `revenue_total` | Выручка (обязательно) |
| `Продажи/день` / `qty` | Количество продаж |
| `Цена` / `avg_price` | Цена продажи |
| `Цена со скидкой маркетплейса` | Цена со скидкой |
| `Цена без скидки` / `avg_price_base` | Базовая цена |
| `На складе` / `stock` | Остаток |
| `Рекламный бюджет` / `ads_budget` | Реклама |
| `Отзывы` / `reviews` | Кол-во отзывов |
| `Рейтинг` / `rating` | Рейтинг |
        """)
    st.stop()


@st.cache_data(show_spinner=False)
def load(file):
    return load_and_prepare(file)

with st.spinner("Читаем файл..."):
    try:
        df = load(uploaded)
    except Exception as e:
        st.error(f"Ошибка загрузки: {e}")
        st.stop()

skus = df['sku_id'].unique().tolist()

col_info1, col_info2, col_info3 = st.columns(3)
with col_info1:
    st.metric("Строк загружено", f"{len(df):,}")
with col_info2:
    st.metric("SKU", len(skus))
with col_info3:
    st.metric("Период", f"{df['date'].min().strftime('%b %Y')} — {df['date'].max().strftime('%b %Y')}")

sku = st.selectbox("SKU для анализа", skus,
                   format_func=lambda x: f"{x}")

# Предпросмотр данных
with st.expander("📊 Предпросмотр исходных данных"):
    sku_preview = df[df['sku_id'] == sku].tail(12)
    st.dataframe(sku_preview.set_index('date'), use_container_width=True)

if not run:
    st.info("👈 Нажмите **Рассчитать прогноз** в боковой панели")
    st.stop()


# ── Прогноз ────────────────────────────────────────────────────────────────
scenario = {}
if sc_price    > 0: scenario['avg_price']          = sc_price
if sc_discount > 0: scenario['avg_price_discount']  = sc_discount
if sc_ads      > 0: scenario['ads_budget']           = sc_ads
if sc_stock    > 0: scenario['stock']                = sc_stock

sku_df = df[df['sku_id'] == sku].copy()

with st.spinner("Обучаем модели..."):
    try:
        result = ensemble_forecast(sku_df, target='revenue_total',
                                   horizon=horizon,
                                   scenario=scenario or None)
    except Exception as e:
        st.error(f"Ошибка прогноза: {e}")
        st.stop()


# ── KPI ────────────────────────────────────────────────────────────────────
total_fc = result.forecast.sum()
history_same = result.history_y.tail(horizon).sum()
delta_pct = (total_fc / history_same - 1) * 100 if history_same > 0 else 0

mape_val = result.ensemble_mape
q_label  = "Высокая" if mape_val < 15 else ("Средняя" if mape_val < 30 else "Низкая")
q_class  = "good"    if mape_val < 15 else ("warn"    if mape_val < 30 else "bad")

best_m = min(result.models, key=lambda m: m.val_mape if np.isfinite(m.val_mape) else 999)
delta_color = "#22C55E" if delta_pct >= 0 else "#EF4444"

c1, c2, c3, c4 = st.columns(4)
for col, label, val, sub, color in [
    (c1, "Прогноз выручки",     f"{total_fc:,.0f} ₽",       f"на {horizon} мес.",        "#E8ECF4"),
    (c2, "Δ vs прошлый период", f"{delta_pct:+.1f}%",        f"{history_same:,.0f} ₽ факт", delta_color),
    (c3, "Точность",            q_label,                     f"MAPE {mape_val:.1f}%",       "#E8ECF4"),
    (c4, "Лучшая модель",       best_m.name,                 f"MAPE {best_m.val_mape:.1f}%","#E8ECF4"),
]:
    col.markdown(f"""<div class="kpi">
    <div class="kpi-label">{label}</div>
    <div class="kpi-value" style="color:{color};">{val}</div>
    <div class="kpi-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("")

# ── Главный график ─────────────────────────────────────────────────────────
st.markdown("#### 📊 Прогноз продаж")

fig = go.Figure()

# История
fig.add_trace(go.Scatter(
    x=result.history_y.index, y=result.history_y.values,
    name="Факт", line=dict(color="#4F8EF7", width=2),
    hovertemplate="%{x|%b %Y}: <b>%{y:,.0f} ₽</b><extra></extra>",
))

# CI
x_ci = list(result.forecast_dates) + list(result.forecast_dates[::-1])
y_ci = list(result.ci_upper) + list(result.ci_lower[::-1])
fig.add_trace(go.Scatter(x=x_ci, y=y_ci, fill="toself",
    fillcolor="rgba(239,68,68,0.1)", line=dict(color="rgba(0,0,0,0)"),
    name="80% интервал", hoverinfo="skip"))

# Прогноз (соединяем с последней точкой)
x_fc = [result.history_y.index[-1]] + list(result.forecast_dates)
y_fc = [float(result.history_y.iloc[-1])] + list(result.forecast)
fig.add_trace(go.Scatter(x=x_fc, y=y_fc, name="Прогноз",
    line=dict(color="#EF4444", width=2.5, dash="dash"),
    mode="lines+markers", marker=dict(size=8, color="#EF4444"),
    hovertemplate="%{x|%b %Y}: <b>%{y:,.0f} ₽</b><extra></extra>"))

# Отдельные модели
colors = {"LightGBM":"#22C55E","SARIMAX":"#F97316","ETS":"#06B6D4"}
for m in result.models:
    y_m = [float(result.history_y.iloc[-1])] + list(m.forecast)
    fig.add_trace(go.Scatter(
        x=x_fc, y=y_m,
        name=f"{m.name} (MAPE {m.val_mape:.1f}%)",
        line=dict(color=colors.get(m.name,"#999"), width=1, dash="dot"),
        opacity=0.6, visible="legendonly"))

fig.update_layout(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#9AA3BE", size=12),
    legend=dict(bgcolor="rgba(28,32,48,0.8)", bordercolor="rgba(255,255,255,.07)",
                borderwidth=1, orientation="h", x=0, y=1.02),
    xaxis=dict(gridcolor="rgba(90,99,128,.12)"),
    yaxis=dict(gridcolor="rgba(90,99,128,.12)", title="Выручка, ₽"),
    hovermode="x unified", margin=dict(l=10,r=10,t=10,b=10), height=380,
)
st.plotly_chart(fig, use_container_width=True)


# ── Таблица прогноза ───────────────────────────────────────────────────────
fc_df = pd.DataFrame({
    "Месяц":                   result.forecast_dates.strftime("%Y-%m"),
    "Прогноз, ₽":              result.forecast.round(0).astype(int),
    "Нижняя граница (80%), ₽": result.ci_lower.round(0).astype(int),
    "Верхняя граница (80%), ₽":result.ci_upper.round(0).astype(int),
})

col_t, col_d = st.columns([3,1])
with col_t:
    st.dataframe(fc_df, use_container_width=True, hide_index=True)
with col_d:
    buf = io.StringIO()
    fc_df.to_csv(buf, index=False)
    st.download_button("⬇️ Скачать CSV", buf.getvalue().encode("utf-8-sig"),
                       f"forecast_{sku}.csv", "text/csv", use_container_width=True)

st.markdown("")

# ── Нижние графики ─────────────────────────────────────────────────────────
col_l, col_r = st.columns(2)

with col_l:
    st.markdown("#### 🤖 Точность моделей (MAPE)")
    names  = [m.name for m in result.models] + ["Ансамбль"]
    mapes  = [m.val_mape for m in result.models] + [result.ensemble_mape]
    bar_colors = ["#22C55E" if v < 15 else "#EAB308" if v < 30 else "#EF4444" for v in mapes]
    bar_colors[-1] = "#4F8EF7"
    fig2 = go.Figure(go.Bar(x=names, y=mapes, marker_color=bar_colors,
                             text=[f"{v:.1f}%" for v in mapes], textposition="outside"))
    fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       font=dict(color="#9AA3BE"),
                       yaxis=dict(gridcolor="rgba(90,99,128,.12)", title="MAPE %"),
                       xaxis=dict(showgrid=False),
                       margin=dict(l=10,r=10,t=10,b=10), height=260, showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

with col_r:
    st.markdown("#### 🔍 Важность признаков")
    top_imp = sorted(result.feature_importance.items(), key=lambda x: -x[1])[:12]
    labels  = [t[0] for t in top_imp][::-1]
    values  = [t[1]*100 for t in top_imp][::-1]
    fig3 = go.Figure(go.Bar(x=values, y=labels, orientation="h",
                             marker_color=["#4F8EF7"]*len(labels),
                             hovertemplate="%{y}: <b>%{x:.1f}%</b><extra></extra>"))
    fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       font=dict(color="#9AA3BE", size=10),
                       xaxis=dict(gridcolor="rgba(90,99,128,.12)", title="Важность %"),
                       yaxis=dict(showgrid=False),
                       margin=dict(l=10,r=10,t=10,b=10), height=280)
    st.plotly_chart(fig3, use_container_width=True)

# Веса ансамбля
with st.expander("⚖️ Веса моделей в ансамбле"):
    wdf = pd.DataFrame([{"Модель": k, "Вес": f"{v*100:.1f}%",
                          "Вал. MAPE": f"{next(m.val_mape for m in result.models if m.name==k):.1f}%"}
                         for k, v in sorted(result.weights.items(), key=lambda x: -x[1])])
    st.dataframe(wdf, use_container_width=True, hide_index=True)

if mape_val > 30:
    st.warning("⚠️ MAPE > 30% — прогноз ненадёжен. Попробуйте увеличить период данных.")
elif mape_val < 15:
    st.success(f"✅ Точность прогноза высокая (MAPE {mape_val:.1f}%)")
