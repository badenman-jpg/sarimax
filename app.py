"""app.py — Sales Forecast Streamlit UI"""
import warnings; warnings.filterwarnings("ignore")
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io, tempfile, os

from forecaster import load_and_prepare, ensemble_forecast
from elasticity  import calc_elasticity, apply_elasticity
from macro_data  import get_macro_df

st.set_page_config(page_title="Sales Forecast", page_icon="📈", layout="wide")
st.markdown("""<style>
.block-container{padding-top:1.2rem;}
.kpi{background:#1C2030;border-radius:12px;padding:14px 18px;margin:4px 0;}
.kpi-label{font-size:10px;color:#5A6380;text-transform:uppercase;letter-spacing:.08em;}
.kpi-value{font-size:24px;font-weight:900;color:#E8ECF4;line-height:1.1;}
.kpi-sub{font-size:11px;color:#5A6380;margin-top:2px;}
</style>""", unsafe_allow_html=True)

COL_MAP = {
    'Дата':'date','SKU':'sku_id','Выручка/день':'revenue_total',
    'Продажи/день':'qty','Цена':'avg_price',
    'Цена со скидкой маркетплейса':'avg_price_discount',
    'Цена без скидки':'avg_price_base','На складе':'stock',
    'Рекламный бюджет':'ads_budget','Отзывы':'reviews','Рейтинг':'rating',
}

# ── Сайдбар ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 Sales Forecast")
    st.divider()
    uploaded = st.file_uploader("Загрузите файл (Excel/CSV)", type=["xlsx","xls","csv"])

    raw_cols = []
    if uploaded:
        try:
            raw_cols = list(pd.read_csv(uploaded, nrows=0).columns
                           if uploaded.name.endswith('.csv')
                           else pd.read_excel(uploaded, nrows=0).columns)
            uploaded.seek(0)
        except Exception:
            pass

    if raw_cols:
        st.markdown("#### 🔗 Связать колонки")
        opts = ["— пропустить —"] + raw_cols
        def pick(label, hints):
            d = next((c for c in hints if c in raw_cols), opts[0])
            return st.selectbox(label, opts, index=opts.index(d) if d in opts else 0, key=f"m_{label}")
        m_date  = pick("📅 Дата",           ["Дата","date"])
        m_sku   = pick("🏷 SKU",             ["SKU","sku_id"])
        m_rev   = pick("💰 Выручка",         ["Выручка/день","revenue_total"])
        m_qty   = pick("📦 Продажи шт",      ["Продажи/день","qty"])
        m_price = pick("💲 Цена",            ["Цена","avg_price"])
        m_pbase = pick("💲 Цена без скидки", ["Цена без скидки","avg_price_base"])
        m_pdisc = pick("💲 Цена со скидкой", ["Цена со скидкой маркетплейса","avg_price_discount"])
        m_stock = pick("🏭 Склад",           ["На складе","stock"])
        m_ads   = pick("📣 Реклама",         ["Рекламный бюджет","ads_budget"])
        m_rev2  = pick("⭐ Отзывы",          ["Отзывы","reviews"])
        m_rat   = pick("⭐ Рейтинг",         ["Рейтинг","rating"])
        user_map = {k:v for k,v in [
            (m_date,'date'),(m_sku,'sku_id'),(m_rev,'revenue_total'),(m_qty,'qty'),
            (m_price,'avg_price'),(m_pbase,'avg_price_base'),(m_pdisc,'avg_price_discount'),
            (m_stock,'stock'),(m_ads,'ads_budget'),(m_rev2,'reviews'),(m_rat,'rating'),
        ] if k != "— пропустить —"}
    else:
        user_map = COL_MAP

    st.divider()
    st.markdown("#### ⚙️ Настройки")
    horizon = st.slider("Горизонт (мес)", 1, 12, 3)
    model_choice = st.selectbox("Модель",
        ["🤖 Авто (ансамбль)","LightGBM","ETS","SARIMAX","Все на графике"])
    use_macro = st.checkbox("Макроданные из ЦБ/Росстат", value=True)
    run = st.button("🚀 Рассчитать прогноз", type="primary", use_container_width=True)


# ── Без файла ──────────────────────────────────────────────────────────────
if uploaded is None:
    st.markdown("""<div style="text-align:center;padding:60px 20px;">
    <div style="font-size:56px;margin-bottom:16px;">📈</div>
    <h2>Sales Forecast</h2>
    <p style="color:#5A6380;max-width:500px;margin:0 auto;">
    Загрузите Excel или CSV. Прогноз штук + выручки с учётом цены, инфляции и эластичности.
    </p></div>""", unsafe_allow_html=True)
    st.stop()


# ── Загрузка ───────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_df(file_bytes, file_name, mapping):
    buf = io.BytesIO(file_bytes)
    df = pd.read_csv(buf) if file_name.endswith('.csv') else pd.read_excel(buf)
    df = df.rename(columns={k:v for k,v in mapping.items() if k in df.columns})
    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tf:
        df.to_excel(tf.name, index=False)
        result = load_and_prepare(tf.name)
        os.unlink(tf.name)
    return result

with st.spinner("Читаем файл..."):
    try:
        fbytes = uploaded.read()
        df = load_df(fbytes, uploaded.name, user_map)
    except Exception as e:
        st.error(f"Ошибка загрузки: {e}")
        st.stop()

# Макроданные
macro_df = None
if use_macro:
    with st.spinner("Макроданные (ЦБ РФ)..."):
        try:
            macro_df = get_macro_df(str(df['date'].min())[:10])
            usd = macro_df['usd_rate'].dropna().iloc[-1]
            inf = macro_df['inflation_index'].dropna().iloc[-1]
            st.sidebar.success(f"✅ USD: {usd:.1f} ₽ | ИПЦ: {inf:.1f}%")
        except Exception as e:
            st.sidebar.warning(f"Макро: {e}")

skus = df['sku_id'].unique().tolist()
c1,c2,c3 = st.columns(3)
c1.metric("Строк", f"{len(df):,}")
c2.metric("SKU", len(skus))
c3.metric("Период", f"{df['date'].min().strftime('%b %Y')} — {df['date'].max().strftime('%b %Y')}")

sku     = st.selectbox("SKU", skus)
sku_df  = df[df['sku_id'] == sku].copy()
has_qty = 'qty' in sku_df.columns and sku_df['qty'].sum() > 0

with st.expander("📊 Данные"):
    st.dataframe(sku_df.set_index('date').tail(12))


# ── Вкладки ────────────────────────────────────────────────────────────────
tab_fc, tab_el, tab_sc, tab_mc, tab_bt = st.tabs(["📈 Прогноз","📉 Эластичность","🎯 Сценарий","🌍 Макро","🔬 Бэктест"])


# ── Эластичность ──────────────────────────────────────────────────────────
with tab_el:
    st.markdown("#### 📉 Ценовая эластичность спроса")
    if 'avg_price' not in sku_df.columns:
        st.warning("Нет колонки с ценой.")
        er = None
    else:
        er = calc_elasticity(sku_df, price_col='avg_price', qty_col='qty',
                             revenue_col='revenue_total',
                             controls=[c for c in ['ads_budget','stock'] if c in sku_df.columns])
        ec = "#22C55E" if er.elasticity < -1 else ("#EAB308" if er.elasticity < 0 else "#EF4444")
        col1, col2 = st.columns([1,2])
        with col1:
            st.markdown(f"""<div style="background:#13161E;border:1px solid rgba(79,142,247,.2);
            border-radius:12px;padding:16px 20px;">
            <div style="font-size:10px;color:#5A6380;text-transform:uppercase;">Эластичность</div>
            <div style="font-size:44px;font-weight:900;color:{ec};">{er.elasticity}</div>
            <div style="font-size:12px;color:#5A6380;">
              {"✅ Значимо" if er.significant else "⚠️ Незначимо"} · p={er.p_value} · R²={er.r_squared}<br>
              Наблюдений: {er.n_obs} · Цена: {er.price_range[0]}–{er.price_range[1]} ₽
            </div></div>""", unsafe_allow_html=True)
            st.info(er.interpretation)
            st.markdown("**Скорректировать вручную:**")
            manual_e = st.number_input("Коэффициент", value=float(er.elasticity),
                                       step=0.1, format="%.2f", key="manual_e")
            use_manual = st.checkbox("Использовать ручное значение")
            eff_e = manual_e if use_manual else er.elasticity
            st.session_state['eff_e'] = eff_e
        with col2:
            bp = float(sku_df['avg_price'].dropna().tail(3).mean())
            bq = float(sku_df['qty'].dropna().tail(3).mean()) if has_qty \
                 else float(sku_df['revenue_total'].dropna().tail(3).mean() / max(bp,1))
            sp = st.slider("Новая цена, ₽", int(bp*0.5), int(bp*2.0), int(bp), step=10)
            sq = apply_elasticity(bq, bp, sp, eff_e)
            sr = sp*sq; br = bp*bq
            s1,s2,s3 = st.columns(3)
            s1.metric("Прогноз продаж",  f"{sq:,.0f} шт",  f"{(sq/bq-1)*100:+.1f}%" if bq>0 else "")
            s2.metric("Прогноз выручки", f"{sr:,.0f} ₽",   f"{(sr/br-1)*100:+.1f}%" if br>0 else "")
            s3.metric("Текущая выручка", f"{br:,.0f} ₽")
            if 'avg_price' in sku_df.columns:
                pdf = sku_df[['avg_price','revenue_total']].dropna().copy()
                pdf['qty_e'] = pdf['revenue_total'] / pdf['avg_price'].replace(0,np.nan)
                pdf = pdf[pdf['avg_price']>0].dropna()
                fig_e = go.Figure()
                fig_e.add_trace(go.Scatter(x=pdf['avg_price'], y=pdf['qty_e'], mode='markers',
                    marker=dict(color='#4F8EF7',size=6,opacity=0.7), name='Факт'))
                xr = np.linspace(pdf['avg_price'].min(), pdf['avg_price'].max(), 50)
                yr = [apply_elasticity(float(pdf['qty_e'].mean()), float(pdf['avg_price'].mean()), p, eff_e) for p in xr]
                fig_e.add_trace(go.Scatter(x=xr, y=yr, mode='lines',
                    line=dict(color='#EF4444',width=2,dash='dash'), name=f'e={eff_e:.2f}'))
                fig_e.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9AA3BE"),height=280,
                    xaxis=dict(gridcolor="rgba(90,99,128,.12)",title="Цена, ₽"),
                    yaxis=dict(gridcolor="rgba(90,99,128,.12)",title="Продажи"),
                    margin=dict(l=10,r=10,t=10,b=10))
                st.plotly_chart(fig_e, use_container_width=True)


# ── Сценарий ───────────────────────────────────────────────────────────────
with tab_sc:
    st.markdown("#### 🎯 Настройка сценария")
    st.caption("Тренд вычисляется автоматически. Цена и ИПЦ влияют на прогноз выручки.")
    last_date = pd.to_datetime(sku_df['date'].max())
    fdates    = pd.date_range(last_date, periods=horizon+1, freq='MS')[1:]
    bp2  = float(sku_df['avg_price'].dropna().tail(3).mean())  if 'avg_price'   in sku_df.columns else 0
    ba   = float(sku_df['ads_budget'].dropna().tail(3).mean()) if 'ads_budget'  in sku_df.columns else 0
    bs   = float(sku_df['stock'].dropna().tail(3).mean())      if 'stock'       in sku_df.columns else 100000
    sc_data = pd.DataFrame({
        "Месяц":          fdates.strftime("%Y-%m"),
        "Цена, ₽":        [round(bp2)] * horizon,
        "Промо-дни (%)":  [0]          * horizon,
        "Склад, шт":      [int(bs)]    * horizon,
        "Реклама, ₽":     [int(ba)]    * horizon,
    })
    if macro_df is not None:
        mfut = macro_df.reindex(fdates)
        sc_data["USD, ₽"] = mfut["usd_rate"].values.round(1)
        sc_data["ИПЦ, %"] = mfut["inflation_index"].values.round(1)
    edited = st.data_editor(sc_data, use_container_width=True, hide_index=True)
    st.session_state['scenario_ed'] = edited
    st.info("💡 ИПЦ влияет на выручку: при инфляции 10% в год выручка растёт ~+0.8% в месяц.\n"
            "Эластичность влияет: при изменении цены прогноз штук корректируется по формуле Q×(P_new/P_base)^e.")


# ── Макро ──────────────────────────────────────────────────────────────────
with tab_mc:
    st.markdown("#### 🌍 Макроэкономические данные")
    if macro_df is None:
        st.warning("Включите «Макроданные» в сайдбаре.")
    else:
        c1m, c2m = st.columns(2)
        for col, series, title, color, ytitle in [
            (c1m, macro_df['usd_rate'].dropna(),        "USD/RUB (ЦБ РФ)",  "#4F8EF7", "₽/$"),
            (c2m, macro_df['inflation_index'].dropna(), "ИПЦ (встроенные)", "#F97316", "ИПЦ %"),
        ]:
            fig_m = go.Figure(go.Scatter(x=series.index, y=series.values,
                line=dict(color=color,width=2)))
            fig_m.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9AA3BE"),height=200,
                yaxis=dict(gridcolor="rgba(90,99,128,.12)",title=ytitle),
                xaxis=dict(gridcolor="rgba(90,99,128,.12)"),
                margin=dict(l=10,r=10,t=20,b=10),
                title=dict(text=title,font=dict(size=13)))
            col.plotly_chart(fig_m, use_container_width=True)
        st.dataframe(macro_df.tail(12).round(2))


# ── Прогноз ────────────────────────────────────────────────────────────────
with tab_fc:
    if not run:
        st.info("👈 Нажмите **Рассчитать прогноз** в сайдбаре")
        st.stop()

    # ── Читаем сценарий ────────────────────────────────────────────────────
    scenario = {}
    if 'scenario_ed' in st.session_state:
        ed = st.session_state['scenario_ed']
        if 'Цена, ₽'   in ed.columns: scenario['avg_price']       = float(ed['Цена, ₽'].mean())
        if 'Склад, шт' in ed.columns: scenario['stock']            = float(ed['Склад, шт'].mean())
        if 'Реклама, ₽'in ed.columns: scenario['ads_budget']       = float(ed['Реклама, ₽'].mean())
        if 'ИПЦ, %'    in ed.columns: scenario['inflation_index']  = float(ed['ИПЦ, %'].mean())

    # Плановая цена для расчёта выручки из штук
    plan_price = scenario.get('avg_price',
        float(sku_df['avg_price'].dropna().tail(3).mean()) if 'avg_price' in sku_df.columns else 1.0)
    base_price = float(sku_df['avg_price'].dropna().tail(3).mean()) if 'avg_price' in sku_df.columns else plan_price

    # Эластичность
    eff_e = st.session_state.get('eff_e', 0.0)
    if eff_e != 0 and base_price > 0 and plan_price != base_price:
        scenario['elasticity']  = eff_e
        scenario['base_price']  = base_price

    # Инфляционная поправка (месячная ставка)
    annual_inf  = (scenario.get('inflation_index', 100) / 100) - 1.0
    monthly_inf = (1 + annual_inf) ** (1/12) - 1

    force = None
    if   model_choice == "LightGBM": force = "LightGBM"
    elif model_choice == "ETS":      force = "ETS"
    elif model_choice == "SARIMAX":  force = "SARIMAX"

    # ── Прогноз штук ──────────────────────────────────────────────────────
    with st.spinner("Прогнозируем штуки..."):
        result_qty = None
        if has_qty:
            try:
                result_qty = ensemble_forecast(sku_df, target='qty',
                                               horizon=horizon, scenario=scenario or None,
                                               force_model=force)
            except Exception as e:
                st.warning(f"Прогноз штук: {e}")

    # ── Прогноз выручки ───────────────────────────────────────────────────
    with st.spinner("Прогнозируем выручку..."):
        result_rev = None
        try:
            result_rev = ensemble_forecast(sku_df, target='revenue_total',
                                           horizon=horizon, scenario=scenario or None,
                                           force_model=force)
        except Exception as e:
            st.error(f"Прогноз выручки: {e}")
            st.stop()

    # ── Если есть прогноз штук — выручка = штуки × цена × инфляция ───────
    if result_qty is not None:
        inf_factors = np.array([(1 + monthly_inf)**(i+1) for i in range(horizon)])
        # Эластичность уже учтена в forecaster через scenario['elasticity']
        # Здесь применяем ценовую и инфляционную поправку поверх штук
        price_factor = plan_price / base_price if base_price > 0 else 1.0
        rev_from_qty         = result_qty.forecast  * plan_price * inf_factors
        rev_from_qty_lo      = result_qty.ci_lower   * plan_price * inf_factors
        rev_from_qty_hi      = result_qty.ci_upper   * plan_price * inf_factors
        # Показываем оба варианта
        has_derived_rev = True
    else:
        has_derived_rev = False

    # ── KPI ───────────────────────────────────────────────────────────────
    main_rev = rev_from_qty if has_derived_rev else result_rev.forecast
    total_fc = main_rev.sum()
    hist_s   = result_rev.history_y.tail(horizon).sum()
    dpct     = (total_fc/hist_s - 1)*100 if hist_s > 0 else 0
    mv       = result_rev.ensemble_mape
    ql       = "Высокая" if mv<15 else ("Средняя" if mv<30 else "Низкая")
    bm       = min(result_rev.models, key=lambda m: m.val_mape if np.isfinite(m.val_mape) else 999)
    dc       = "#22C55E" if dpct>=0 else "#EF4444"

    k1,k2,k3,k4,k5 = st.columns(5)
    for col,lbl,val,sub,clr in [
        (k1,"Прогноз выручки",  f"{total_fc:,.0f} ₽",     f"за {horizon} мес.", "#E8ECF4"),
        (k2,"Δ vs прошлый",     f"{dpct:+.1f}%",           f"{hist_s:,.0f} ₽",  dc),
        (k3,"Точность (MAPE)",  ql,                        f"{mv:.1f}%",         "#E8ECF4"),
        (k4,"Лучшая модель",    bm.name,                   f"MAPE {bm.val_mape:.1f}%","#E8ECF4"),
        (k5,"Плановая цена",    f"{plan_price:,.0f} ₽",    f"база {base_price:,.0f} ₽","#E8ECF4"),
    ]:
        col.markdown(f"""<div class="kpi"><div class="kpi-label">{lbl}</div>
        <div class="kpi-value" style="color:{clr};">{val}</div>
        <div class="kpi-sub">{sub}</div></div>""", unsafe_allow_html=True)

    st.markdown("")

    MC = {"LightGBM":"#22C55E","SARIMAX":"#F97316","ETS":"#06B6D4"}

    # ── ГРАФИК 1: Штуки ───────────────────────────────────────────────────
    if result_qty is not None:
        st.markdown("#### 📦 Прогноз продаж (штуки)")
        st.caption("Модели работают на стабильном ряду штук — это даёт лучшую точность SARIMAX")
        fig_q = go.Figure()
        fig_q.add_trace(go.Scatter(x=result_qty.history_y.index, y=result_qty.history_y.values,
            name="Факт (шт)", line=dict(color="#4F8EF7",width=2),
            hovertemplate="%{x|%b %Y}: <b>%{y:,.0f} шт</b><extra></extra>"))
        xci = list(result_qty.forecast_dates)+list(result_qty.forecast_dates[::-1])
        yci = list(result_qty.ci_upper)+list(result_qty.ci_lower[::-1])
        fig_q.add_trace(go.Scatter(x=xci,y=yci,fill="toself",
            fillcolor="rgba(239,68,68,.1)",line=dict(color="rgba(0,0,0,0)"),
            name="80% CI",hoverinfo="skip"))
        xfc = [result_qty.history_y.index[-1]]+list(result_qty.forecast_dates)
        yfc = [float(result_qty.history_y.iloc[-1])]+list(result_qty.forecast)
        fig_q.add_trace(go.Scatter(x=xfc,y=yfc,name="Прогноз (шт)",
            line=dict(color="#EF4444",width=2.5,dash="dash"),mode="lines+markers",
            marker=dict(size=8),hovertemplate="%{x|%b %Y}: <b>%{y:,.0f} шт</b><extra></extra>"))
        show_all = model_choice == "Все на графике"
        for m in result_qty.models:
            ym = [float(result_qty.history_y.iloc[-1])]+list(m.forecast)
            fig_q.add_trace(go.Scatter(x=xfc,y=ym,
                name=f"{m.name} MAPE {m.val_mape:.1f}%",
                line=dict(color=MC.get(m.name,"#999"),width=1.5,dash="dot"),opacity=0.7,
                visible=True if show_all else "legendonly"))
        fig_q.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#9AA3BE",size=12),height=340,hovermode="x unified",
            legend=dict(bgcolor="rgba(28,32,48,.8)",orientation="h",x=0,y=1.02),
            xaxis=dict(gridcolor="rgba(90,99,128,.12)"),
            yaxis=dict(gridcolor="rgba(90,99,128,.12)",title="Штуки"),
            margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig_q, use_container_width=True)

    # ── ГРАФИК 2: Выручка ─────────────────────────────────────────────────
    st.markdown("#### 💰 Прогноз выручки (₽)")
    if has_derived_rev:
        st.caption(f"Выручка = штуки × {plan_price:,.0f} ₽ × инфляция. "
                   f"Инфляция: {annual_inf*100:.1f}%/год → +{monthly_inf*100:.2f}%/мес")
    fig_r = go.Figure()
    # История
    fig_r.add_trace(go.Scatter(x=result_rev.history_y.index, y=result_rev.history_y.values,
        name="Факт (₽)", line=dict(color="#4F8EF7",width=2),
        hovertemplate="%{x|%b %Y}: <b>%{y:,.0f} ₽</b><extra></extra>"))
    # CI
    if has_derived_rev:
        xci2 = list(result_qty.forecast_dates)+list(result_qty.forecast_dates[::-1])
        yci2 = list(rev_from_qty_hi)+list(rev_from_qty_lo[::-1])
        prognoz_y = [float(result_rev.history_y.iloc[-1])]+list(rev_from_qty)
        prognoz_x = [result_rev.history_y.index[-1]]+list(result_qty.forecast_dates)
        label_rev = "Прогноз (шт×цена×ИПЦ)"
    else:
        xci2 = list(result_rev.forecast_dates)+list(result_rev.forecast_dates[::-1])
        yci2 = list(result_rev.ci_upper)+list(result_rev.ci_lower[::-1])
        prognoz_y = [float(result_rev.history_y.iloc[-1])]+list(result_rev.forecast)
        prognoz_x = [result_rev.history_y.index[-1]]+list(result_rev.forecast_dates)
        label_rev = "Прогноз (выручка)"

    fig_r.add_trace(go.Scatter(x=xci2,y=yci2,fill="toself",
        fillcolor="rgba(239,68,68,.1)",line=dict(color="rgba(0,0,0,0)"),
        name="80% CI",hoverinfo="skip"))
    fig_r.add_trace(go.Scatter(x=prognoz_x,y=prognoz_y,name=label_rev,
        line=dict(color="#EF4444",width=2.5,dash="dash"),mode="lines+markers",
        marker=dict(size=8),hovertemplate="%{x|%b %Y}: <b>%{y:,.0f} ₽</b><extra></extra>"))

    # Прямой прогноз выручки как пунктир для сравнения
    if has_derived_rev:
        xfc2 = [result_rev.history_y.index[-1]]+list(result_rev.forecast_dates)
        yfc2 = [float(result_rev.history_y.iloc[-1])]+list(result_rev.forecast)
        fig_r.add_trace(go.Scatter(x=xfc2,y=yfc2,name="Прогноз (прямой, без цены/ИПЦ)",
            line=dict(color="#9AA3BE",width=1,dash="dot"),opacity=0.6,
            hovertemplate="%{x|%b %Y}: <b>%{y:,.0f} ₽</b><extra></extra>"))

    show_all = model_choice == "Все на графике"
    for m in result_rev.models:
        ym2 = [float(result_rev.history_y.iloc[-1])]+list(m.forecast)
        fig_r.add_trace(go.Scatter(x=xfc2 if has_derived_rev else prognoz_x, y=ym2,
            name=f"{m.name} MAPE {m.val_mape:.1f}%",
            line=dict(color=MC.get(m.name,"#999"),width=1.5,dash="dot"),opacity=0.7,
            visible=True if show_all else "legendonly"))

    fig_r.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#9AA3BE",size=12),height=340,hovermode="x unified",
        legend=dict(bgcolor="rgba(28,32,48,.8)",orientation="h",x=0,y=1.02),
        xaxis=dict(gridcolor="rgba(90,99,128,.12)"),
        yaxis=dict(gridcolor="rgba(90,99,128,.12)",title="Выручка, ₽"),
        margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(fig_r, use_container_width=True)

    # ── Таблица ────────────────────────────────────────────────────────────
    st.markdown("#### 📋 Таблица прогноза")
    fdates_str = result_rev.forecast_dates.strftime("%Y-%m")
    fc_data    = {"Месяц": fdates_str}
    if result_qty is not None:
        fc_data["Прогноз шт"]    = result_qty.forecast.round(0).astype(int)
        fc_data["Шт нижняя"]     = result_qty.ci_lower.round(0).astype(int)
        fc_data["Шт верхняя"]    = result_qty.ci_upper.round(0).astype(int)
    if has_derived_rev:
        fc_data["Выручка (шт×цена×ИПЦ), ₽"] = rev_from_qty.round(0).astype(int)
    fc_data["Выручка (прямой прогноз), ₽"]   = result_rev.forecast.round(0).astype(int)
    fc_data["Нижняя, ₽"]                      = result_rev.ci_lower.round(0).astype(int)
    fc_data["Верхняя, ₽"]                     = result_rev.ci_upper.round(0).astype(int)
    fc_df = pd.DataFrame(fc_data)

    ct, cd = st.columns([3,1])
    with ct: st.dataframe(fc_df, use_container_width=True, hide_index=True)
    with cd:
        buf = io.StringIO(); fc_df.to_csv(buf, index=False)
        st.download_button("⬇️ CSV", buf.getvalue().encode("utf-8-sig"),
                           f"forecast_{sku}.csv","text/csv",use_container_width=True)

    # ── Сравнение моделей ──────────────────────────────────────────────────
    with st.expander("🤖 Сравнение моделей"):
        tabs_m = st.tabs(["Штуки","Выручка"]) if result_qty else [None, st.container()]
        for idx_t, (res, title) in enumerate([(result_qty,"Штуки"),(result_rev,"Выручка")]):
            if res is None: continue
            with (tabs_m[idx_t] if result_qty else st.container()):
                names  = [m.name for m in res.models]+["Ансамбль"]
                mapes  = [m.val_mape for m in res.models]+[res.ensemble_mape]
                bc     = ["#22C55E" if v<15 else "#EAB308" if v<30 else "#EF4444" for v in mapes]
                bc[-1] = "#4F8EF7"
                f2 = go.Figure(go.Bar(x=names,y=mapes,marker_color=bc,
                    text=[f"{v:.1f}%" for v in mapes],textposition="outside"))
                f2.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9AA3BE"),height=220,showlegend=False,
                    yaxis=dict(gridcolor="rgba(90,99,128,.12)",title="MAPE %"),
                    xaxis=dict(showgrid=False),margin=dict(l=10,r=10,t=10,b=10))
                cl2,cr2 = st.columns(2)
                cl2.plotly_chart(f2,use_container_width=True)
                wdf = pd.DataFrame([{
                    "Модель": k,"Вес": f"{v*100:.1f}%",
                    "MAPE": f"{next(m.val_mape for m in res.models if m.name==k):.1f}%",
                    "Описание": {"LightGBM":"Градиентный бустинг, все факторы",
                                 "ETS":"Тренд + сезонность, стабильная",
                                 "SARIMAX":"ARIMA+сезонность, нужно 3+ года"}.get(k,"")
                } for k,v in sorted(res.weights.items(),key=lambda x:-x[1])])
                cr2.dataframe(wdf,use_container_width=True,hide_index=True)

    with st.expander("📖 Как работают модели"):
        st.markdown("""
**Архитектура прогноза:**
1. **Штуки** прогнозируются моделями (стабильный ряд → SARIMAX работает лучше)
2. **Выручка** = Прогноз штук × Плановая цена × (1 + инфляция)^месяц

**LightGBM** — учитывает все факторы (цена, склад, реклама, лаги). Хорошо на любом горизонте.

**ETS** — тренд + сезонность. Стабильна, не зависит от факторов.

**SARIMAX** — ARIMA с сезонностью. Требует 3+ года истории. При малой истории переобучается.
Включается сезонность `m=12` только при истории ≥ 36 мес.

**Инфляция:** применяется к прогнозу выручки как `(1+r_месячная)^шаг`.
ИПЦ 109.5% = +9.5%/год = +0.76%/мес → за 3 мес выручка вырастет ~+2.3% только от инфляции.

**Эластичность:** если цена изменится с 388 ₽ до 420 ₽ при e=-1.5,
то штуки снизятся на `(420/388)^-1.5 - 1 ≈ -12%`, но выручка: `+8% цена × -12% штуки = -4%`.
        """)

    if mv > 30:
        st.warning("⚠️ MAPE > 30% — выберите LightGBM вручную или добавьте данных.")
    elif mv < 15:
        st.success(f"✅ Точность высокая (MAPE {mv:.1f}%)")


# ── Бэктест ────────────────────────────────────────────────────────────────
with tab_bt:
    st.markdown("#### 🔬 Бэктестирование — калибровка точности")
    st.caption("Обучаем модель на части данных, прогнозируем вперёд, сравниваем с реальным фактом из файла.")

    n_months = len(sku_df)
    max_test  = min(18, n_months // 3)
    min_train = 12

    if n_months < min_train + 3:
        st.warning(f"Мало данных для бэктеста. Нужно минимум {min_train + 3} месяцев, есть {n_months}.")
    else:
        bt_horizon = st.slider(
            "Горизонт бэктеста (мес) — сколько последних месяцев использовать как тест",
            min_value=3, max_value=max_test, value=min(12, max_test), key="bt_h"
        )
        bt_force = st.selectbox("Модель для бэктеста",
            ["🤖 Авто (ансамбль)","LightGBM","ETS","SARIMAX"], key="bt_m")
        bt_run = st.button("▶️ Запустить бэктест", key="bt_run")

        if bt_run:
            # Разбивка: обучение на первых N-bt_horizon, тест на последних bt_horizon
            train_df = sku_df.iloc[:-bt_horizon].copy()
            test_df  = sku_df.iloc[-bt_horizon:].copy()
            test_dates = pd.to_datetime(test_df['date'].values)

            bt_force_map = {"🤖 Авто (ансамбль)": None, "LightGBM":"LightGBM",
                            "ETS":"ETS", "SARIMAX":"SARIMAX"}
            bt_fm = bt_force_map[bt_force]

            with st.spinner(f"Обучаем на {len(train_df)} мес, прогнозируем на {bt_horizon} мес..."):
                bt_qty_res = bt_rev_res = None
                try:
                    if has_qty:
                        bt_qty_res = ensemble_forecast(train_df, target='qty',
                                                       horizon=bt_horizon, force_model=bt_fm)
                except Exception as e:
                    st.warning(f"Бэктест штук: {e}")
                try:
                    bt_rev_res = ensemble_forecast(train_df, target='revenue_total',
                                                   horizon=bt_horizon, force_model=bt_fm)
                except Exception as e:
                    st.error(f"Бэктест выручки: {e}"); st.stop()

            # ── Метрики точности ──────────────────────────────────────────
            def bt_metrics(pred, actual, label):
                mask = actual > 0
                if mask.sum() == 0: return {}
                mape_v = float(np.mean(np.abs((actual[mask]-pred[mask])/actual[mask]))*100)
                rmse_v = float(np.sqrt(np.mean((actual-pred)**2)))
                bias_v = float(np.mean((pred-actual)/actual[mask])*100)
                return {"Метрика": label, "MAPE %": round(mape_v,1),
                        "RMSE": f"{rmse_v:,.0f}", "Bias %": f"{bias_v:+.1f}%",
                        "Оценка": "✅ Отлично" if mape_v<15 else "🟡 Средне" if mape_v<30 else "🔴 Плохо"}

            metrics_rows = []
            if bt_qty_res:
                actual_qty = test_df['qty'].values if 'qty' in test_df.columns else None
                if actual_qty is not None:
                    metrics_rows.append(bt_metrics(bt_qty_res.forecast, actual_qty, "Штуки"))
            actual_rev = test_df['revenue_total'].values
            metrics_rows.append(bt_metrics(bt_rev_res.forecast, actual_rev, "Выручка (прямой)"))
            if bt_qty_res and 'avg_price' in train_df.columns:
                bp_bt = float(train_df['avg_price'].dropna().tail(3).mean())
                rev_from_q = bt_qty_res.forecast * bp_bt
                metrics_rows.append(bt_metrics(rev_from_q, actual_rev, "Выручка (шт×цена)"))

            if metrics_rows:
                st.markdown("##### 📊 Точность прогноза на историческом тесте")
                mdf = pd.DataFrame(metrics_rows)
                st.dataframe(mdf, use_container_width=True, hide_index=True)

            # ── График 1: Штуки бэктест ───────────────────────────────────
            if bt_qty_res and 'qty' in test_df.columns:
                st.markdown("##### 📦 Штуки: прогноз vs факт")
                actual_qty = test_df['qty'].values
                fig_bt_q = go.Figure()
                # Полная история
                fig_bt_q.add_trace(go.Scatter(
                    x=sku_df['date'], y=sku_df['qty'] if 'qty' in sku_df.columns else sku_df['revenue_total'],
                    name="Факт (всё)", line=dict(color="#4F8EF7", width=2, dash="dot"),
                    opacity=0.5,
                    hovertemplate="%{x|%b %Y}: <b>%{y:,.0f}</b><extra></extra>"))
                # Тренировочная часть
                fig_bt_q.add_trace(go.Scatter(
                    x=train_df['date'], y=train_df['qty'],
                    name=f"Обучение ({len(train_df)} мес)", line=dict(color="#4F8EF7", width=2.5),
                    hovertemplate="%{x|%b %Y}: <b>%{y:,.0f} шт</b><extra></extra>"))
                # Факт тестового периода
                fig_bt_q.add_trace(go.Scatter(
                    x=test_dates, y=actual_qty,
                    name="Факт (тест)", line=dict(color="#22C55E", width=2.5),
                    mode="lines+markers", marker=dict(size=8, symbol="circle"),
                    hovertemplate="%{x|%b %Y}: <b>%{y:,.0f} шт (факт)</b><extra></extra>"))
                # CI
                xci_bt = list(bt_qty_res.forecast_dates) + list(bt_qty_res.forecast_dates[::-1])
                yci_bt = list(bt_qty_res.ci_upper) + list(bt_qty_res.ci_lower[::-1])
                fig_bt_q.add_trace(go.Scatter(x=xci_bt, y=yci_bt, fill="toself",
                    fillcolor="rgba(239,68,68,.12)", line=dict(color="rgba(0,0,0,0)"),
                    name="80% CI", hoverinfo="skip"))
                # Прогноз
                xfc_bt = [pd.Timestamp(train_df['date'].iloc[-1])] + list(bt_qty_res.forecast_dates)
                yfc_bt = [float(train_df['qty'].iloc[-1])] + list(bt_qty_res.forecast)
                fig_bt_q.add_trace(go.Scatter(x=xfc_bt, y=yfc_bt,
                    name="Прогноз", line=dict(color="#EF4444", width=2.5, dash="dash"),
                    mode="lines+markers", marker=dict(size=8),
                    hovertemplate="%{x|%b %Y}: <b>%{y:,.0f} шт (прогноз)</b><extra></extra>"))
                # Вертикальная линия разбивки
                split_date = train_df['date'].iloc[-1]
                fig_bt_q.add_vline(x=split_date, line_width=1.5, line_dash="dash",
                                   line_color="#EAB308",
                                   annotation_text="← обучение | тест →",
                                   annotation_position="top")
                fig_bt_q.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#9AA3BE", size=12), height=380, hovermode="x unified",
                    legend=dict(bgcolor="rgba(28,32,48,.8)", orientation="h", x=0, y=1.02),
                    xaxis=dict(gridcolor="rgba(90,99,128,.12)"),
                    yaxis=dict(gridcolor="rgba(90,99,128,.12)", title="Штуки"),
                    margin=dict(l=10,r=10,t=30,b=10))
                st.plotly_chart(fig_bt_q, use_container_width=True)

            # ── График 2: Выручка бэктест ──────────────────────────────────
            st.markdown("##### 💰 Выручка: прогноз vs факт")
            fig_bt_r = go.Figure()
            fig_bt_r.add_trace(go.Scatter(
                x=sku_df['date'], y=sku_df['revenue_total'],
                name="Факт (всё)", line=dict(color="#4F8EF7", width=2, dash="dot"),
                opacity=0.4,
                hovertemplate="%{x|%b %Y}: <b>%{y:,.0f} ₽</b><extra></extra>"))
            fig_bt_r.add_trace(go.Scatter(
                x=train_df['date'], y=train_df['revenue_total'],
                name=f"Обучение ({len(train_df)} мес)", line=dict(color="#4F8EF7", width=2.5),
                hovertemplate="%{x|%b %Y}: <b>%{y:,.0f} ₽</b><extra></extra>"))
            fig_bt_r.add_trace(go.Scatter(
                x=test_dates, y=actual_rev,
                name="Факт (тест)", line=dict(color="#22C55E", width=2.5),
                mode="lines+markers", marker=dict(size=8),
                hovertemplate="%{x|%b %Y}: <b>%{y:,.0f} ₽ (факт)</b><extra></extra>"))
            # CI выручки
            xci_r = list(bt_rev_res.forecast_dates) + list(bt_rev_res.forecast_dates[::-1])
            yci_r = list(bt_rev_res.ci_upper) + list(bt_rev_res.ci_lower[::-1])
            fig_bt_r.add_trace(go.Scatter(x=xci_r, y=yci_r, fill="toself",
                fillcolor="rgba(239,68,68,.12)", line=dict(color="rgba(0,0,0,0)"),
                name="80% CI", hoverinfo="skip"))
            # Прогноз выручки прямой
            xfc_r = [pd.Timestamp(train_df['date'].iloc[-1])] + list(bt_rev_res.forecast_dates)
            yfc_r = [float(train_df['revenue_total'].iloc[-1])] + list(bt_rev_res.forecast)
            fig_bt_r.add_trace(go.Scatter(x=xfc_r, y=yfc_r,
                name="Прогноз выручки", line=dict(color="#EF4444", width=2.5, dash="dash"),
                mode="lines+markers", marker=dict(size=8),
                hovertemplate="%{x|%b %Y}: <b>%{y:,.0f} ₽</b><extra></extra>"))
            # Прогноз через штуки
            if bt_qty_res and 'avg_price' in train_df.columns:
                bp_bt2 = float(train_df['avg_price'].dropna().tail(3).mean())
                rev_q2 = bt_qty_res.forecast * bp_bt2
                xfc_q2 = [pd.Timestamp(train_df['date'].iloc[-1])] + list(bt_qty_res.forecast_dates)
                yfc_q2 = [float(train_df['revenue_total'].iloc[-1])] + list(rev_q2)
                fig_bt_r.add_trace(go.Scatter(x=xfc_q2, y=yfc_q2,
                    name=f"Прогноз (шт×{bp_bt2:,.0f}₽)",
                    line=dict(color="#F97316", width=2, dash="dash"),
                    mode="lines+markers", marker=dict(size=6),
                    hovertemplate="%{x|%b %Y}: <b>%{y:,.0f} ₽</b><extra></extra>"))
            fig_bt_r.add_vline(x=train_df['date'].iloc[-1], line_width=1.5, line_dash="dash",
                               line_color="#EAB308",
                               annotation_text="← обучение | тест →",
                               annotation_position="top")
            fig_bt_r.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9AA3BE", size=12), height=380, hovermode="x unified",
                legend=dict(bgcolor="rgba(28,32,48,.8)", orientation="h", x=0, y=1.02),
                xaxis=dict(gridcolor="rgba(90,99,128,.12)"),
                yaxis=dict(gridcolor="rgba(90,99,128,.12)", title="Выручка, ₽"),
                margin=dict(l=10,r=10,t=30,b=10))
            st.plotly_chart(fig_bt_r, use_container_width=True)

            # ── Детальная таблица ──────────────────────────────────────────
            with st.expander("📋 Детальная таблица бэктеста"):
                bt_data = {"Месяц": [d.strftime("%Y-%m") for d in test_dates]}
                if bt_qty_res and 'qty' in test_df.columns:
                    bt_data["Факт шт"]    = test_df['qty'].values.astype(int)
                    bt_data["Прогноз шт"] = bt_qty_res.forecast.round(0).astype(int)
                    bt_data["Ошибка шт"]  = (bt_qty_res.forecast - test_df['qty'].values).round(0).astype(int)
                bt_data["Факт выручка, ₽"]    = actual_rev.astype(int)
                bt_data["Прогноз выручка, ₽"] = bt_rev_res.forecast.round(0).astype(int)
                bt_data["Ошибка, ₽"]          = (bt_rev_res.forecast - actual_rev).round(0).astype(int)
                bt_data["Ошибка %"]           = [
                    f"{((p/a-1)*100):+.1f}%" if a>0 else "—"
                    for p,a in zip(bt_rev_res.forecast, actual_rev)
                ]
                st.dataframe(pd.DataFrame(bt_data), use_container_width=True, hide_index=True)

            st.info("""💡 **Интерпретация:**
- **MAPE < 15%** — модель хорошо калиброван, прогнозу можно доверять
- **MAPE 15–30%** — допустимо для волатильных товаров
- **MAPE > 30%** — модель не улавливает паттерн, попробуйте другую модель или больший горизонт обучения
- Смотрите на **Bias**: если +20% — модель систематически завышает прогноз""")
