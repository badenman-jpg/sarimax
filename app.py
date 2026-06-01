"""app.py — Sales Forecast Streamlit UI"""
import warnings; warnings.filterwarnings("ignore")
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io, tempfile, os

from forecaster import load_and_prepare, ensemble_forecast, EnsembleResult
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
    'Рекламный бюджет':'ads_budget','Отзывы':'reviews',
    'Рейтинг':'rating',
}

# ── Сайдбар ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 Sales Forecast")
    st.divider()
    uploaded = st.file_uploader("Загрузите файл (Excel/CSV)", type=["xlsx","xls","csv"])

    # Читаем сырые колонки для маппинга
    raw_cols = []
    if uploaded:
        try:
            if uploaded.name.endswith('.csv'):
                raw_cols = list(pd.read_csv(uploaded, nrows=0).columns)
            else:
                raw_cols = list(pd.read_excel(uploaded, nrows=0).columns)
            uploaded.seek(0)
        except Exception:
            pass

    if raw_cols:
        st.markdown("#### 🔗 Связать колонки")
        opts = ["— пропустить —"] + raw_cols
        def pick(label, hints):
            d = next((c for c in hints if c in raw_cols), opts[0])
            return st.selectbox(label, opts, index=opts.index(d) if d in opts else 0,
                                key=f"m_{label}")
        m_date    = pick("📅 Дата",          ["Дата","date"])
        m_sku     = pick("🏷 SKU",            ["SKU","sku_id"])
        m_rev     = pick("💰 Выручка",        ["Выручка/день","revenue_total"])
        m_qty     = pick("📦 Продажи шт",     ["Продажи/день","qty"])
        m_price   = pick("💲 Цена",           ["Цена","avg_price"])
        m_pbase   = pick("💲 Цена без скидки",["Цена без скидки","avg_price_base"])
        m_pdisc   = pick("💲 Цена со скидкой",["Цена со скидкой маркетплейса","avg_price_discount"])
        m_stock   = pick("🏭 Склад",          ["На складе","stock"])
        m_ads     = pick("📣 Реклама",        ["Рекламный бюджет","ads_budget"])
        m_reviews = pick("⭐ Отзывы",         ["Отзывы","reviews"])
        m_rating  = pick("⭐ Рейтинг",        ["Рейтинг","rating"])
        user_map  = {k:v for k,v in [
            (m_date,'date'),(m_sku,'sku_id'),(m_rev,'revenue_total'),(m_qty,'qty'),
            (m_price,'avg_price'),(m_pbase,'avg_price_base'),(m_pdisc,'avg_price_discount'),
            (m_stock,'stock'),(m_ads,'ads_budget'),(m_reviews,'reviews'),(m_rating,'rating'),
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


# ── Без файла ─────────────────────────────────────────────────────────────
if uploaded is None:
    st.markdown("""<div style="text-align:center;padding:60px 20px;">
    <div style="font-size:56px;margin-bottom:16px;">📈</div>
    <h2>Sales Forecast</h2>
    <p style="color:#5A6380;max-width:500px;margin:0 auto;">
    Загрузите Excel или CSV с историей продаж.<br>
    Свяжите колонки в сайдбаре и нажмите Рассчитать.
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
    with st.spinner("Макроданные (ЦБ РФ, Росстат)..."):
        try:
            macro_df = get_macro_df(str(df['date'].min())[:10])
            usd = macro_df['usd_rate'].dropna().iloc[-1]
            st.sidebar.success(f"✅ USD: {usd:.1f} ₽")
        except Exception as e:
            st.sidebar.warning(f"Макро: {e}")

skus = df['sku_id'].unique().tolist()
c1,c2,c3 = st.columns(3)
c1.metric("Строк", f"{len(df):,}")
c2.metric("SKU", len(skus))
c3.metric("Период", f"{df['date'].min().strftime('%b %Y')} — {df['date'].max().strftime('%b %Y')}")

sku = st.selectbox("SKU", skus)
sku_df = df[df['sku_id'] == sku].copy()

with st.expander("📊 Данные"):
    st.dataframe(sku_df.set_index('date').tail(12))

# ── Вкладки ────────────────────────────────────────────────────────────────
tab_fc, tab_el, tab_sc, tab_mc = st.tabs(["📈 Прогноз","📉 Эластичность","🎯 Сценарий","🌍 Макро"])

# ── Эластичность ──────────────────────────────────────────────────────────
with tab_el:
    st.markdown("#### 📉 Ценовая эластичность спроса")
    if 'avg_price' not in sku_df.columns:
        st.warning("Нет колонки с ценой.")
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
            # Симулятор
            bp = float(sku_df['avg_price'].dropna().tail(3).mean())
            bq = float(sku_df['qty'].dropna().tail(3).mean()) if 'qty' in sku_df.columns \
                 else float(sku_df['revenue_total'].dropna().tail(3).mean()/bp)
            sp = st.slider("Новая цена, ₽", int(bp*0.5), int(bp*2.0), int(bp), step=10)
            sq = apply_elasticity(bq, bp, sp, eff_e)
            sr = sp * sq; br = bp * bq
            s1,s2,s3 = st.columns(3)
            s1.metric("Прогноз продаж", f"{sq:,.0f} шт", f"{(sq/bq-1)*100:+.1f}%" if bq>0 else "")
            s2.metric("Прогноз выручки",f"{sr:,.0f} ₽",  f"{(sr/br-1)*100:+.1f}%" if br>0 else "")
            s3.metric("Текущая выручка",f"{br:,.0f} ₽")
            # Scatter цена vs продажи
            if 'avg_price' in sku_df.columns:
                pdf = sku_df[['avg_price','revenue_total']].dropna().copy()
                pdf['qty_est'] = pdf['revenue_total'] / pdf['avg_price'].replace(0,np.nan)
                pdf = pdf[pdf['avg_price']>0].dropna()
                fig_e = go.Figure()
                fig_e.add_trace(go.Scatter(x=pdf['avg_price'], y=pdf['qty_est'],
                    mode='markers', marker=dict(color='#4F8EF7',size=6,opacity=0.7), name='Факт'))
                xr = np.linspace(pdf['avg_price'].min(), pdf['avg_price'].max(), 50)
                yr = [apply_elasticity(float(pdf['qty_est'].mean()), float(pdf['avg_price'].mean()), p, eff_e) for p in xr]
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
    st.caption("Отредактируйте плановые значения. Тренд вычисляется автоматически.")
    last_date = pd.to_datetime(sku_df['date'].max())
    fdates = pd.date_range(last_date, periods=horizon+1, freq='MS')[1:]
    bp2 = float(sku_df['avg_price'].dropna().tail(3).mean()) if 'avg_price' in sku_df.columns else 0
    ba  = float(sku_df['ads_budget'].dropna().tail(3).mean()) if 'ads_budget' in sku_df.columns else 0
    bs  = float(sku_df['stock'].dropna().tail(3).mean()) if 'stock' in sku_df.columns else 100000
    sc_data = pd.DataFrame({
        "Месяц": fdates.strftime("%Y-%m"),
        "Цена, ₽": [round(bp2)] * horizon,
        "Промо-дни (%)": [0] * horizon,
        "Склад, шт": [int(bs)] * horizon,
        "Реклама, ₽": [int(ba)] * horizon,
    })
    if macro_df is not None:
        mfut = macro_df.reindex(fdates)
        sc_data["USD, ₽"]    = mfut["usd_rate"].values.round(1)
        sc_data["ИПЦ, %"]    = mfut["inflation_index"].values.round(1)
    edited = st.data_editor(sc_data, use_container_width=True, hide_index=True)
    st.session_state['scenario_ed'] = edited

# ── Макро ──────────────────────────────────────────────────────────────────
with tab_mc:
    st.markdown("#### 🌍 Макроэкономические данные")
    if macro_df is None:
        st.warning("Включите «Макроданные» в сайдбаре.")
    else:
        c1m, c2m = st.columns(2)
        for col, series, title, color, ytitle in [
            (c1m, macro_df['usd_rate'].dropna(),       "USD/RUB (ЦБ РФ)",   "#4F8EF7", "₽/$"),
            (c2m, macro_df['inflation_index'].dropna(),"ИПЦ (Росстат)",     "#F97316", "ИПЦ %"),
        ]:
            fig_m = go.Figure(go.Scatter(x=series.index, y=series.values,
                line=dict(color=color, width=2)))
            fig_m.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9AA3BE"),height=200,
                yaxis=dict(gridcolor="rgba(90,99,128,.12)",title=ytitle),
                xaxis=dict(gridcolor="rgba(90,99,128,.12)"),
                margin=dict(l=10,r=10,t=20,b=10),title=dict(text=title,font=dict(size=13)))
            col.plotly_chart(fig_m, use_container_width=True)
        st.dataframe(macro_df.tail(12).round(2))

# ── Прогноз ────────────────────────────────────────────────────────────────
with tab_fc:
    if not run:
        st.info("👈 Нажмите **Рассчитать прогноз** в сайдбаре")
        st.stop()

    # Сценарий из таблицы
    scenario = {}
    if 'scenario_ed' in st.session_state:
        ed = st.session_state['scenario_ed']
        if 'Цена, ₽'    in ed.columns: scenario['avg_price']  = float(ed['Цена, ₽'].mean())
        if 'Склад, шт'  in ed.columns: scenario['stock']       = float(ed['Склад, шт'].mean())
        if 'Реклама, ₽' in ed.columns: scenario['ads_budget']  = float(ed['Реклама, ₽'].mean())

    force = None
    if   model_choice == "LightGBM": force = "LightGBM"
    elif model_choice == "ETS":      force = "ETS"
    elif model_choice == "SARIMAX":  force = "SARIMAX"

    with st.spinner("Обучаем модели..."):
        try:
            result = ensemble_forecast(sku_df, target='revenue_total',
                                       horizon=horizon, scenario=scenario or None,
                                       force_model=force)
        except Exception as e:
            st.error(f"Ошибка: {e}")
            st.stop()

    # KPI
    total_fc = result.forecast.sum()
    hist_s   = result.history_y.tail(horizon).sum()
    dpct     = (total_fc/hist_s-1)*100 if hist_s > 0 else 0
    mv       = result.ensemble_mape
    ql       = "Высокая" if mv<15 else ("Средняя" if mv<30 else "Низкая")
    bm       = min(result.models, key=lambda m: m.val_mape if np.isfinite(m.val_mape) else 999)
    dc       = "#22C55E" if dpct>=0 else "#EF4444"

    k1,k2,k3,k4 = st.columns(4)
    for col,lbl,val,sub,clr in [
        (k1,"Прогноз выручки", f"{total_fc:,.0f} ₽", f"за {horizon} мес.",  "#E8ECF4"),
        (k2,"Δ vs прошлый",   f"{dpct:+.1f}%",       f"{hist_s:,.0f} ₽ факт", dc),
        (k3,"Точность",       ql,                     f"MAPE {mv:.1f}%",     "#E8ECF4"),
        (k4,"Лучшая модель",  bm.name,                f"MAPE {bm.val_mape:.1f}%","#E8ECF4"),
    ]:
        col.markdown(f"""<div class="kpi"><div class="kpi-label">{lbl}</div>
        <div class="kpi-value" style="color:{clr};">{val}</div>
        <div class="kpi-sub">{sub}</div></div>""", unsafe_allow_html=True)

    st.markdown("")

    # График
    MC = {"LightGBM":"#22C55E","SARIMAX":"#F97316","ETS":"#06B6D4"}
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=result.history_y.index, y=result.history_y.values,
        name="Факт", line=dict(color="#4F8EF7",width=2),
        hovertemplate="%{x|%b %Y}: <b>%{y:,.0f} ₽</b><extra></extra>"))
    xci = list(result.forecast_dates)+list(result.forecast_dates[::-1])
    yci = list(result.ci_upper)+list(result.ci_lower[::-1])
    fig.add_trace(go.Scatter(x=xci,y=yci,fill="toself",
        fillcolor="rgba(239,68,68,.1)",line=dict(color="rgba(0,0,0,0)"),
        name="80% CI",hoverinfo="skip"))
    xfc = [result.history_y.index[-1]]+list(result.forecast_dates)
    yfc = [float(result.history_y.iloc[-1])]+list(result.forecast)
    fig.add_trace(go.Scatter(x=xfc,y=yfc,name="Прогноз",
        line=dict(color="#EF4444",width=2.5,dash="dash"),mode="lines+markers",
        marker=dict(size=8),hovertemplate="%{x|%b %Y}: <b>%{y:,.0f} ₽</b><extra></extra>"))
    show_all = model_choice == "Все на графике"
    for m in result.models:
        ym = [float(result.history_y.iloc[-1])]+list(m.forecast)
        fig.add_trace(go.Scatter(x=xfc,y=ym,
            name=f"{m.name} MAPE {m.val_mape:.1f}%",
            line=dict(color=MC.get(m.name,"#999"),width=1.5,dash="dot"),opacity=0.7,
            visible=True if show_all else "legendonly"))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#9AA3BE",size=12),height=380,hovermode="x unified",
        legend=dict(bgcolor="rgba(28,32,48,.8)",orientation="h",x=0,y=1.02),
        xaxis=dict(gridcolor="rgba(90,99,128,.12)"),
        yaxis=dict(gridcolor="rgba(90,99,128,.12)",title="Выручка, ₽"),
        margin=dict(l=10,r=10,t=10,b=10))
    st.plotly_chart(fig, use_container_width=True)

    # Таблица
    fc_df = pd.DataFrame({
        "Месяц":       result.forecast_dates.strftime("%Y-%m"),
        "Прогноз, ₽":  result.forecast.round(0).astype(int),
        "Нижняя, ₽":   result.ci_lower.round(0).astype(int),
        "Верхняя, ₽":  result.ci_upper.round(0).astype(int),
    })
    for m in result.models:
        fc_df[f"{m.name}, ₽"] = m.forecast.round(0).astype(int)
    ct, cd = st.columns([3,1])
    with ct: st.dataframe(fc_df, use_container_width=True, hide_index=True)
    with cd:
        buf = io.StringIO(); fc_df.to_csv(buf, index=False)
        st.download_button("⬇️ CSV", buf.getvalue().encode("utf-8-sig"),
                           f"forecast_{sku}.csv","text/csv",use_container_width=True)

    # Сравнение моделей
    with st.expander("🤖 Сравнение моделей"):
        names  = [m.name for m in result.models]+["Ансамбль"]
        mapes  = [m.val_mape for m in result.models]+[result.ensemble_mape]
        bc     = ["#22C55E" if v<15 else "#EAB308" if v<30 else "#EF4444" for v in mapes]
        bc[-1] = "#4F8EF7"
        f2 = go.Figure(go.Bar(x=names,y=mapes,marker_color=bc,
            text=[f"{v:.1f}%" for v in mapes],textposition="outside"))
        f2.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#9AA3BE"),height=240,showlegend=False,
            yaxis=dict(gridcolor="rgba(90,99,128,.12)",title="MAPE %"),
            xaxis=dict(showgrid=False),margin=dict(l=10,r=10,t=10,b=10))
        cl2, cr2 = st.columns(2)
        cl2.plotly_chart(f2, use_container_width=True)
        wdf = pd.DataFrame([{
            "Модель": k, "Вес": f"{v*100:.1f}%",
            "MAPE": f"{next(m.val_mape for m in result.models if m.name==k):.1f}%",
            "Описание": {"LightGBM":"Градиентный бустинг",
                         "ETS":"Экспоненциальное сглаживание",
                         "SARIMAX":"ARIMA с сезонностью"}.get(k,"")
        } for k,v in sorted(result.weights.items(), key=lambda x: -x[1])])
        cr2.dataframe(wdf, use_container_width=True, hide_index=True)

    if mv > 30:
        st.warning("⚠️ MAPE > 30% — добавьте больше данных или уточните факторы.")
    elif mv < 15:
        st.success(f"✅ Точность высокая (MAPE {mv:.1f}%)")
