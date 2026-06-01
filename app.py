"""
app.py — Streamlit UI для прогноза продаж v2
Новое: маппинг колонок, выбор модели, эластичность, макро из API, сценарная таблица
"""
import warnings; warnings.filterwarnings("ignore")
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io

from forecaster import load_and_prepare, ensemble_forecast, EnsembleResult
from elasticity import calc_elasticity, apply_elasticity, ElasticityResult
from macro_data import get_macro_df

st.set_page_config(page_title="Sales Forecast", page_icon="📈", layout="wide")

st.markdown("""<style>
.block-container{padding-top:1.2rem;}
.kpi{background:#1C2030;border-radius:12px;padding:14px 18px;margin:4px 0;}
.kpi-label{font-size:10px;color:#5A6380;text-transform:uppercase;letter-spacing:.08em;}
.kpi-value{font-size:24px;font-weight:900;color:#E8ECF4;line-height:1.1;}
.kpi-sub{font-size:11px;color:#5A6380;margin-top:2px;}
.good{color:#22C55E!important;}.warn{color:#EAB308!important;}.bad{color:#EF4444!important;}
.elast-card{background:#13161E;border:1px solid rgba(79,142,247,.2);
            border-radius:12px;padding:16px 20px;margin:8px 0;}
</style>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# САЙДБАР
# ══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📈 Sales Forecast v2")
    st.caption("LightGBM · ETS · SARIMAX · Эластичность")
    st.divider()

    uploaded = st.file_uploader("Загрузите файл (Excel/CSV)",
                                type=["xlsx","xls","csv"])

    # Загружаем данные сразу для маппинга
    raw_df = None
    if uploaded:
        try:
            if uploaded.name.endswith('.csv'):
                raw_df = pd.read_csv(uploaded)
            else:
                raw_df = pd.read_excel(uploaded)
            uploaded.seek(0)
        except Exception:
            pass

    # Маппинг колонок
    if raw_df is not None:
        st.markdown("#### 🔗 Связать колонки")
        cols_available = ["— не использовать —"] + list(raw_df.columns)

        def col_select(label, defaults):
            default_col = next((c for c in defaults if c in raw_df.columns), cols_available[0])
            idx = cols_available.index(default_col) if default_col in cols_available else 0
            return st.selectbox(label, cols_available, index=idx, key=f"map_{label}")

        map_date    = col_select("📅 Дата",             ["Дата","date","Date"])
        map_sku     = col_select("🏷 SKU/Артикул",       ["SKU","sku_id","Артикул","артикул"])
        map_revenue = col_select("💰 Выручка",           ["Выручка/день","revenue_total","Выручка"])
        map_qty     = col_select("📦 Кол-во продаж",     ["Продажи/день","qty","Количество"])
        map_price   = col_select("🏷 Цена продажи",      ["Цена","avg_price","price"])
        map_price_b = col_select("🏷 Цена без скидки",   ["Цена без скидки","avg_price_base"])
        map_price_d = col_select("🏷 Цена со скидкой",   ["Цена со скидкой маркетплейса","avg_price_discount"])
        map_stock   = col_select("🏭 Остаток склада",    ["На складе","stock"])
        map_ads     = col_select("📣 Рекл. бюджет",      ["Рекламный бюджет","ads_budget"])
        map_reviews = col_select("⭐ Отзывы",            ["Отзывы","reviews"])
        map_rating  = col_select("⭐ Рейтинг",           ["Рейтинг","rating"])

        col_mapping = {
            map_date:    "date",    map_sku:     "sku_id",
            map_revenue: "revenue_total", map_qty: "qty",
            map_price:   "avg_price", map_price_b: "avg_price_base",
            map_price_d: "avg_price_discount", map_stock: "stock",
            map_ads:     "ads_budget", map_reviews: "reviews",
            map_rating:  "rating",
        }
        col_mapping = {k: v for k, v in col_mapping.items()
                       if k != "— не использовать —"}

    st.divider()
    st.markdown("#### ⚙️ Настройки прогноза")
    horizon = st.slider("Горизонт (месяцев)", 1, 12, 3)

    model_choice = st.selectbox(
        "Модель прогноза",
        ["🤖 Авто (лучшая по MAPE)", "LightGBM", "ETS", "SARIMAX", "Все на одном графике"],
        help="Авто — ансамбль с весами по точности. Ручной — только выбранная модель."
    )

    st.markdown("#### 🌍 Макроданные")
    use_macro = st.checkbox("Подтягивать USD и инфляцию из ЦБ/Росстат", value=True)
    if use_macro:
        st.caption("Данные кэшируются на 1 час")

    run = st.button("🚀 Рассчитать прогноз", type="primary", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# ЭКРАН БЕЗ ФАЙЛА
# ══════════════════════════════════════════════════════════════════════════
if uploaded is None:
    st.markdown("""
    <div style="text-align:center;padding:60px 20px;">
      <div style="font-size:56px;margin-bottom:16px;">📈</div>
      <h2>Sales Forecast v2</h2>
      <p style="color:#5A6380;max-width:520px;margin:0 auto 24px;">
        Ансамблевый прогноз продаж с ценовой эластичностью,<br>
        макроданными из ЦБ РФ и выбором модели.
      </p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📋 Поддерживаемые колонки (любые названия — маппинг в сайдбаре)"):
        st.markdown("""
| Колонка | Тип | Описание |
|---|---|---|
| Дата / date | дата | Дата записи (дневная или месячная) |
| SKU / sku_id | текст | Артикул товара |
| Выручка/день / revenue_total | число | Выручка — **обязательно** |
| Продажи/день / qty | число | Кол-во продаж (для эластичности) |
| Цена / avg_price | число | Цена продажи |
| Цена без скидки | число | Базовая цена |
| На складе / stock | число | Остаток |
| Рекламный бюджет / ads_budget | число | Бюджет рекламы |
| Отзывы / reviews | число | Число отзывов |
| Рейтинг / rating | число | Рейтинг товара |
        """)
    st.stop()


# ══════════════════════════════════════════════════════════════════════════
# ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ
# ══════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load(file_bytes, file_name, mapping):
    import io as _io
    buf = _io.BytesIO(file_bytes)
    if file_name.endswith('.csv'):
        df = pd.read_csv(buf)
    else:
        df = pd.read_excel(buf)
    # Применяем маппинг пользователя
    df = df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})
    return df


with st.spinner("Читаем файл..."):
    try:
        file_bytes = uploaded.read()
        raw = load(file_bytes, uploaded.name, col_mapping if raw_df is not None else {})
        # Передаём уже переименованный файл в load_and_prepare
        import io as _io
        buf2 = _io.BytesIO(file_bytes)
        buf2.name = uploaded.name
        df = load_and_prepare(buf2)
        # Если маппинг был, применяем его поверх
        if raw_df is not None and col_mapping:
            buf3 = _io.BytesIO(file_bytes)
            buf3.name = uploaded.name
            if uploaded.name.endswith('.csv'):
                tmp = pd.read_csv(buf3)
            else:
                tmp = pd.read_excel(buf3)
            tmp = tmp.rename(columns={k: v for k, v in col_mapping.items() if k in tmp.columns})
            # Перезапускаем prepare на переименованных данных
            from forecaster import load_and_prepare as lap
            import tempfile, os
            with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tf:
                tmp.to_excel(tf.name, index=False)
                df = lap(tf.name)
                os.unlink(tf.name)
    except Exception as e:
        st.error(f"Ошибка загрузки: {e}")
        st.stop()

# Подтягиваем макроданные
macro_df = None
if use_macro:
    with st.spinner("Загружаем макроданные (ЦБ РФ, Росстат)..."):
        try:
            macro_df = get_macro_df(str(df['date'].min())[:10])
            usd_latest = macro_df['usd_rate'].dropna().iloc[-1]
            inf_latest = macro_df['inflation_index'].dropna().iloc[-1]
            st.sidebar.success(f"✅ USD: {usd_latest:.1f} ₽ | ИПЦ: {inf_latest:.1f}%")
        except Exception as e:
            st.sidebar.warning(f"Макроданные недоступны: {e}")

skus = df['sku_id'].unique().tolist()

# Метрики файла
c1, c2, c3, c4 = st.columns(4)
c1.metric("Строк", f"{len(df):,}")
c2.metric("SKU", len(skus))
c3.metric("Период",
          f"{df['date'].min().strftime('%b %Y')} — {df['date'].max().strftime('%b %Y')}")
c4.metric("Макро", "✅ Загружено" if macro_df is not None else "❌ Нет")

sku = st.selectbox("SKU для анализа", skus)
sku_df = df[df['sku_id'] == sku].copy()

# Добавляем макроданные в sku_df если есть
if macro_df is not None and 'date' in sku_df.columns:
    sku_df = sku_df.merge(
        macro_df.reset_index().rename(columns={'index': 'date'}),
        on='date', how='left'
    )

with st.expander("📊 Предпросмотр данных"):
    st.dataframe(sku_df.set_index('date').tail(12), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# ВКЛАДКИ
# ══════════════════════════════════════════════════════════════════════════
tab_forecast, tab_elast, tab_scenario, tab_macro = st.tabs(
    ["📈 Прогноз", "📉 Эластичность", "🎯 Сценарий", "🌍 Макроданные"]
)


# ── ВКЛАДКА: ЭЛАСТИЧНОСТЬ ──────────────────────────────────────────────────
with tab_elast:
    st.markdown("#### 📉 Ценовая эластичность спроса")
    st.caption("Метод: log-log OLS регрессия ln(Q) = a + b·ln(P) + тренд + сезонность")

    price_col_e = 'avg_price' if 'avg_price' in sku_df.columns else None
    qty_col_e   = 'qty' if 'qty' in sku_df.columns else None

    if price_col_e is None:
        st.warning("Нет колонки с ценой — эластичность не вычислима.")
    else:
        controls_e = [c for c in ['ads_budget','stock','reviews'] if c in sku_df.columns]
        elast_result: ElasticityResult = calc_elasticity(
            sku_df, price_col=price_col_e, qty_col=qty_col_e or 'qty',
            revenue_col='revenue_total', controls=controls_e,
        )

        col_e1, col_e2 = st.columns([1, 2])
        with col_e1:
            e_val = elast_result.elasticity
            e_color = "#22C55E" if e_val < -1 else ("#EAB308" if e_val < 0 else "#EF4444")
            sig_label = "✅ Значимо" if elast_result.significant else "⚠️ Незначимо"

            st.markdown(f"""<div class="elast-card">
            <div class="kpi-label">Коэффициент эластичности</div>
            <div style="font-size:40px;font-weight:900;color:{e_color};">{e_val}</div>
            <div style="font-size:12px;color:#5A6380;margin-top:4px;">
              {sig_label} · p={elast_result.p_value} · R²={elast_result.r_squared}<br>
              Наблюдений: {elast_result.n_obs} · Цена: {elast_result.price_range[0]}–{elast_result.price_range[1]} ₽
            </div>
            </div>""", unsafe_allow_html=True)

            st.info(elast_result.interpretation)

            # Ручная правка
            st.markdown("**Скорректировать вручную:**")
            manual_e = st.number_input("Коэффициент эластичности",
                                       value=float(elast_result.elasticity),
                                       step=0.1, format="%.2f",
                                       key="manual_elasticity",
                                       help="Авто-рассчитанный коэффициент. Измените если знаете лучше.")
            use_manual_e = st.checkbox("Использовать ручное значение", value=False)

            effective_elasticity = manual_e if use_manual_e else elast_result.elasticity

        with col_e2:
            if price_col_e in sku_df.columns and 'revenue_total' in sku_df.columns:
                # График цена vs продажи
                plot_df = sku_df[[price_col_e, 'revenue_total']].dropna()
                if qty_col_e and qty_col_e in sku_df.columns:
                    plot_df['qty'] = sku_df[qty_col_e]
                else:
                    plot_df['qty'] = plot_df['revenue_total'] / plot_df[price_col_e].replace(0, np.nan)

                plot_df = plot_df[plot_df[price_col_e] > 0].dropna()

                fig_e = go.Figure()
                fig_e.add_trace(go.Scatter(
                    x=plot_df[price_col_e], y=plot_df['qty'],
                    mode='markers',
                    marker=dict(color='#4F8EF7', size=6, opacity=0.7),
                    name='Факт',
                    hovertemplate="Цена: %{x:.0f} ₽<br>Продажи: %{y:.0f}<extra></extra>",
                ))

                # Линия тренда
                if len(plot_df) > 3:
                    x_line = np.linspace(plot_df[price_col_e].min(),
                                         plot_df[price_col_e].max(), 50)
                    base_qty = plot_df['qty'].mean()
                    base_price = plot_df[price_col_e].mean()
                    y_line = [apply_elasticity(base_qty, base_price, p, effective_elasticity)
                              for p in x_line]
                    fig_e.add_trace(go.Scatter(
                        x=x_line, y=y_line,
                        mode='lines', name=f'Эластичность e={effective_elasticity:.2f}',
                        line=dict(color='#EF4444', width=2, dash='dash'),
                    ))

                fig_e.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9AA3BE"),
                    xaxis=dict(title="Цена, ₽", gridcolor="rgba(90,99,128,.12)"),
                    yaxis=dict(title="Продажи, шт", gridcolor="rgba(90,99,128,.12)"),
                    height=320, margin=dict(l=10,r=10,t=10,b=10),
                    legend=dict(bgcolor="rgba(28,32,48,.8)"),
                )
                st.plotly_chart(fig_e, use_container_width=True)

            # Симулятор эластичности
            st.markdown("**Симулятор: что будет при изменении цены?**")
            base_p = float(sku_df[price_col_e].dropna().iloc[-3:].mean()) if price_col_e in sku_df.columns else 100
            base_q = float(sku_df['qty'].dropna().iloc[-3:].mean()) if qty_col_e in sku_df.columns else \
                     float(sku_df['revenue_total'].dropna().iloc[-3:].mean() / base_p)

            sim_price = st.slider("Новая цена, ₽",
                                  min_value=int(base_p * 0.5),
                                  max_value=int(base_p * 2.0),
                                  value=int(base_p), step=10)
            sim_qty  = apply_elasticity(base_q, base_p, sim_price, effective_elasticity)
            sim_rev  = sim_price * sim_qty
            base_rev = base_p * base_q
            rev_delta_pct = (sim_rev / base_rev - 1) * 100 if base_rev > 0 else 0

            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Прогноз продаж", f"{sim_qty:,.0f} шт",
                       f"{(sim_qty/base_q-1)*100:+.1f}%" if base_q > 0 else "")
            sc2.metric("Прогноз выручки", f"{sim_rev:,.0f} ₽",
                       f"{rev_delta_pct:+.1f}%")
            sc3.metric("Текущая выручка", f"{base_rev:,.0f} ₽")


# ── ВКЛАДКА: СЦЕНАРИЙ ─────────────────────────────────────────────────────
with tab_scenario:
    st.markdown("#### 🎯 Настройка сценария прогноза")
    st.caption("Отредактируйте плановые значения по месяцам. Тренд вычисляется автоматически.")

    last_date = pd.to_datetime(sku_df['date'].max())
    future_dates = pd.date_range(last_date, periods=horizon + 1, freq='MS')[1:]

    # Базовые значения — среднее последних 3 мес
    def last_mean(col, default=0):
        if col in sku_df.columns:
            return float(sku_df[col].dropna().tail(3).mean())
        return default

    base_price = last_mean('avg_price', 0)
    base_ads   = last_mean('ads_budget', 0)
    base_stock = last_mean('stock', 100000)

    # Таблица сценария
    scenario_data = pd.DataFrame({
        "Месяц":         future_dates.strftime("%Y-%m"),
        "Цена, ₽":       [round(base_price, 0)] * horizon,
        "Промо-дни (%)": [0] * horizon,
        "Склад, шт":     [int(base_stock)] * horizon,
        "Рекл. бюджет":  [int(base_ads)] * horizon,
    })

    # Добавляем USD если есть
    if macro_df is not None:
        macro_future = macro_df.reindex(future_dates)
        scenario_data["USD курс"] = macro_future["usd_rate"].values.round(1)
        scenario_data["ИПЦ (инфляция)"] = macro_future["inflation_index"].values.round(1)

    st.info("✏️ Отредактируйте цену и % промо-дней (тренд вычисляется автоматически):")
    edited = st.data_editor(scenario_data, use_container_width=True,
                            hide_index=True, key="scenario_table")

    # Сохраняем в session_state для прогноза
    st.session_state['scenario_edited'] = edited
    st.session_state['effective_elasticity'] = st.session_state.get('manual_elasticity',
                                                                       elast_result.elasticity if price_col_e else 0)


# ── ВКЛАДКА: МАКРОДАННЫЕ ──────────────────────────────────────────────────
with tab_macro:
    st.markdown("#### 🌍 Макроэкономические данные")
    if macro_df is None:
        st.warning("Макроданные не загружены. Включите опцию в сайдбаре.")
    else:
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("**Курс USD/RUB (ЦБ РФ)**")
            usd_plot = macro_df['usd_rate'].dropna()
            fig_usd = go.Figure(go.Scatter(
                x=usd_plot.index, y=usd_plot.values,
                line=dict(color='#4F8EF7', width=2),
                hovertemplate="%{x|%b %Y}: <b>%{y:.1f} ₽</b><extra></extra>",
            ))
            fig_usd.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                                   plot_bgcolor="rgba(0,0,0,0)",
                                   font=dict(color="#9AA3BE"),
                                   yaxis=dict(gridcolor="rgba(90,99,128,.12)", title="₽/$"),
                                   xaxis=dict(gridcolor="rgba(90,99,128,.12)"),
                                   height=220, margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig_usd, use_container_width=True)
        with col_m2:
            st.markdown("**ИПЦ (инфляция, Росстат)**")
            inf_plot = macro_df['inflation_index'].dropna()
            fig_inf = go.Figure(go.Scatter(
                x=inf_plot.index, y=inf_plot.values,
                line=dict(color='#F97316', width=2),
                hovertemplate="%{x|%b %Y}: <b>%{y:.1f}%</b><extra></extra>",
            ))
            fig_inf.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                                   plot_bgcolor="rgba(0,0,0,0)",
                                   font=dict(color="#9AA3BE"),
                                   yaxis=dict(gridcolor="rgba(90,99,128,.12)", title="ИПЦ %"),
                                   xaxis=dict(gridcolor="rgba(90,99,128,.12)"),
                                   height=220, margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig_inf, use_container_width=True)

        st.dataframe(macro_df.tail(12).round(2), use_container_width=True)


# ── ВКЛАДКА: ПРОГНОЗ ──────────────────────────────────────────────────────
with tab_forecast:
    if not run:
        st.info("👈 Нажмите **Рассчитать прогноз** в боковой панели")
        st.stop()

    # Строим сценарий из таблицы
    scenario = {}
    if 'scenario_edited' in st.session_state:
        ed = st.session_state['scenario_edited']
        if 'Цена, ₽' in ed.columns:
            scenario['avg_price'] = float(ed['Цена, ₽'].mean())
        if 'Склад, шт' in ed.columns:
            scenario['stock'] = float(ed['Склад, шт'].mean())
        if 'Рекл. бюджет' in ed.columns:
            scenario['ads_budget'] = float(ed['Рекл. бюджет'].mean())

    # Определяем forced_model
    force_model = None
    if model_choice == "LightGBM": force_model = "LightGBM"
    elif model_choice == "ETS":    force_model = "ETS"
    elif model_choice == "SARIMAX":force_model = "SARIMAX"

    with st.spinner("Обучаем модели..."):
        try:
            result = ensemble_forecast(
                sku_df, target='revenue_total',
                horizon=horizon,
                scenario=scenario or None,
                force_model=force_model,
            )
        except Exception as e:
            st.error(f"Ошибка прогноза: {e}")
            st.stop()

    # Корректировка прогноза через эластичность
    elast_correction = None
    if (price_col_e and 'scenario_edited' in st.session_state
            and 'Цена, ₽' in st.session_state['scenario_edited'].columns):
        new_price = float(st.session_state['scenario_edited']['Цена, ₽'].mean())
        eff_e = st.session_state.get('effective_elasticity', 0)
        if eff_e != 0 and base_price > 0:
            elast_correction = new_price / base_price

    # KPI
    total_fc = result.forecast.sum()
    history_same = result.history_y.tail(horizon).sum()
    delta_pct = (total_fc/history_same - 1)*100 if history_same > 0 else 0
    mape_v = result.ensemble_mape
    q_label = "Высокая" if mape_v < 15 else ("Средняя" if mape_v < 30 else "Низкая")
    best_m = min(result.models, key=lambda m: m.val_mape if np.isfinite(m.val_mape) else 999)
    d_color = "#22C55E" if delta_pct >= 0 else "#EF4444"

    col1, col2, col3, col4, col5 = st.columns(5)
    for col, lbl, val, sub, clr in [
        (col1, "Прогноз выручки",    f"{total_fc:,.0f} ₽",    f"за {horizon} мес.", "#E8ECF4"),
        (col2, "Δ vs прошлый период",f"{delta_pct:+.1f}%",    f"{history_same:,.0f} ₽ факт", d_color),
        (col3, "Точность",           q_label,                  f"MAPE {mape_v:.1f}%", "#E8ECF4"),
        (col4, "Лучшая модель",      best_m.name,              f"MAPE {best_m.val_mape:.1f}%", "#E8ECF4"),
        (col5, "Модель выбора",      model_choice.split()[0] if model_choice != "🤖 Авто (лучшая по MAPE)" else "Ансамбль",
               "", "#E8ECF4"),
    ]:
        col.markdown(f"""<div class="kpi"><div class="kpi-label">{lbl}</div>
        <div class="kpi-value" style="color:{clr};">{val}</div>
        <div class="kpi-sub">{sub}</div></div>""", unsafe_allow_html=True)

    st.markdown("")

    # График
    model_colors = {"LightGBM":"#22C55E","SARIMAX":"#F97316","ETS":"#06B6D4"}
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=result.history_y.index, y=result.history_y.values,
        name="Факт", line=dict(color="#4F8EF7", width=2),
        hovertemplate="%{x|%b %Y}: <b>%{y:,.0f} ₽</b><extra></extra>",
    ))
    x_ci = list(result.forecast_dates) + list(result.forecast_dates[::-1])
    y_ci = list(result.ci_upper) + list(result.ci_lower[::-1])
    fig.add_trace(go.Scatter(x=x_ci, y=y_ci, fill="toself",
        fillcolor="rgba(239,68,68,.1)", line=dict(color="rgba(0,0,0,0)"),
        name="80% CI", hoverinfo="skip"))
    x_fc = [result.history_y.index[-1]] + list(result.forecast_dates)
    y_fc = [float(result.history_y.iloc[-1])] + list(result.forecast)
    fig.add_trace(go.Scatter(x=x_fc, y=y_fc, name="Прогноз (ансамбль)",
        line=dict(color="#EF4444", width=2.5, dash="dash"),
        mode="lines+markers", marker=dict(size=8),
        hovertemplate="%{x|%b %Y}: <b>%{y:,.0f} ₽</b><extra></extra>"))

    show_all = model_choice == "Все на одном графике"
    for m in result.models:
        y_m = [float(result.history_y.iloc[-1])] + list(m.forecast)
        fig.add_trace(go.Scatter(
            x=x_fc, y=y_m,
            name=f"{m.name} (MAPE {m.val_mape:.1f}%)",
            line=dict(color=model_colors.get(m.name,"#999"), width=1.5, dash="dot"),
            opacity=0.7,
            visible=True if show_all else "legendonly",
        ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#9AA3BE", size=12),
        legend=dict(bgcolor="rgba(28,32,48,.8)", bordercolor="rgba(255,255,255,.07)",
                    borderwidth=1, orientation="h", x=0, y=1.02),
        xaxis=dict(gridcolor="rgba(90,99,128,.12)"),
        yaxis=dict(gridcolor="rgba(90,99,128,.12)", title="Выручка, ₽"),
        hovermode="x unified", margin=dict(l=10,r=10,t=10,b=10), height=380,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Таблица прогноза
    fc_df = pd.DataFrame({
        "Месяц":           result.forecast_dates.strftime("%Y-%m"),
        "Прогноз, ₽":      result.forecast.round(0).astype(int),
        "Нижняя (80%), ₽": result.ci_lower.round(0).astype(int),
        "Верхняя (80%), ₽":result.ci_upper.round(0).astype(int),
    })
    # Добавляем прогнозы по каждой модели
    for m in result.models:
        fc_df[f"{m.name}, ₽"] = m.forecast.round(0).astype(int)

    col_t, col_d = st.columns([3,1])
    with col_t:
        st.dataframe(fc_df, use_container_width=True, hide_index=True)
    with col_d:
        buf = io.StringIO()
        fc_df.to_csv(buf, index=False)
        st.download_button("⬇️ Скачать CSV", buf.getvalue().encode("utf-8-sig"),
                           f"forecast_{sku}.csv", "text/csv", use_container_width=True)

    # Сравнение моделей
    with st.expander("🤖 Сравнение моделей и веса"):
        col_l, col_r = st.columns(2)
        with col_l:
            names  = [m.name for m in result.models] + ["Ансамбль"]
            mapes  = [m.val_mape for m in result.models] + [result.ensemble_mape]
            bcolors= ["#22C55E" if v<15 else "#EAB308" if v<30 else "#EF4444" for v in mapes]
            bcolors[-1] = "#4F8EF7"
            fig2 = go.Figure(go.Bar(x=names, y=mapes, marker_color=bcolors,
                text=[f"{v:.1f}%" for v in mapes], textposition="outside"))
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9AA3BE"),
                yaxis=dict(gridcolor="rgba(90,99,128,.12)",title="MAPE %"),
                xaxis=dict(showgrid=False),
                margin=dict(l=10,r=10,t=10,b=10),height=240,showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)
        with col_r:
            wdf = pd.DataFrame([{
                "Модель": k,
                "Вес": f"{v*100:.1f}%",
                "Вал. MAPE": f"{next(m.val_mape for m in result.models if m.name==k):.1f}%",
                "Описание": {"LightGBM":"Градиентный бустинг, учитывает все факторы",
                             "ETS":"Экспоненциальное сглаживание, тренд+сезонность",
                             "SARIMAX":"ARIMA с сезонностью и внешними факторами"}.get(k,"")
            } for k, v in sorted(result.weights.items(), key=lambda x: -x[1])])
            st.dataframe(wdf, use_container_width=True, hide_index=True)

    if mape_v > 30:
        st.warning("⚠️ MAPE > 30% — добавьте больше исторических данных или проверьте факторы.")
    elif mape_v < 15:
        st.success(f"✅ Точность высокая (MAPE {mape_v:.1f}%)")
