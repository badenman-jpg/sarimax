"""
macro_data.py — Подтягивает макроданные из открытых API
ЦБ РФ: курс USD/RUB
Росстат (через EMISS API): индекс инфляции
"""
import warnings; warnings.filterwarnings("ignore")
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
import streamlit as st


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_usd_rate(date_from: str = "2020-01-01") -> pd.Series:
    """
    Курс USD/RUB из ЦБ РФ XML API.
    Возвращает месячный ряд (среднее за месяц).
    """
    try:
        date_to = datetime.today().strftime("%d/%m/%Y")
        date_from_fmt = pd.to_datetime(date_from).strftime("%d/%m/%Y")
        url = (f"https://www.cbr.ru/scripts/XML_dynamic.asp"
               f"?date_req1={date_from_fmt}&date_req2={date_to}&VAL_NM_RQ=R01235")
        resp = requests.get(url, timeout=10)
        resp.encoding = "windows-1251"
        from xml.etree import ElementTree as ET
        root = ET.fromstring(resp.text)
        records = []
        for rec in root.findall("Record"):
            date = pd.to_datetime(rec.attrib["Date"], format="%d.%m.%Y")
            val = float(rec.find("Value").text.replace(",", "."))
            records.append({"date": date, "usd_rate": val})
        df = pd.DataFrame(records).set_index("date")
        # Агрегируем в месячные (среднее)
        monthly = df.resample("MS").mean()
        return monthly["usd_rate"]
    except Exception as e:
        print(f"ЦБ РФ API error: {e}")
        return pd.Series(dtype=float)


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_inflation() -> pd.Series:
    """
    Индекс потребительских цен (инфляция) — Росстат через EMISS.
    Fallback: статические данные если API недоступен.
    """
    try:
        # EMISS API Росстата — ИПЦ помесячный
        url = ("https://showdata.gks.ru/api/en/data/"
               "?id=57032&periodicity=m&from=2020-01-01")
        resp = requests.get(url, timeout=10)
        data = resp.json()
        records = []
        for item in data.get("data", []):
            try:
                date = pd.to_datetime(item["date"][:10])
                val = float(item["value"])
                records.append({"date": date, "inflation_index": val})
            except Exception:
                continue
        if records:
            df = pd.DataFrame(records).set_index("date")
            monthly = df.resample("MS").last()
            return monthly["inflation_index"]
    except Exception as e:
        print(f"Росстат API error: {e}")

    # Fallback — известные данные + линейная экстраполяция
    known = {
        "2020-01": 102.4, "2020-06": 103.0, "2020-12": 104.9,
        "2021-06": 106.5, "2021-12": 108.4,
        "2022-06": 117.1, "2022-12": 111.9,
        "2023-06": 103.2, "2023-12": 107.4,
        "2024-06": 108.6, "2024-12": 109.5,
        "2025-06": 109.8,
    }
    dates = pd.to_datetime(list(known.keys()))
    vals  = list(known.values())
    idx   = pd.date_range("2020-01-01", datetime.today(), freq="MS")
    s     = pd.Series(vals, index=dates).reindex(idx).interpolate("linear")
    s.name = "inflation_index"
    return s


def get_macro_df(date_from: str = "2020-01-01") -> pd.DataFrame:
    """
    Возвращает DataFrame с макропеременными: usd_rate, inflation_index.
    Индекс — DatetimeIndex с частотой MS.
    """
    usd = fetch_usd_rate(date_from)
    inf = fetch_inflation()

    idx = pd.date_range(date_from, datetime.today() + timedelta(days=365), freq="MS")
    df  = pd.DataFrame(index=idx)

    if len(usd) > 0:
        df["usd_rate"] = usd.reindex(idx).interpolate("linear").ffill().bfill()
    else:
        df["usd_rate"] = np.nan

    if len(inf) > 0:
        df["inflation_index"] = inf.reindex(idx).interpolate("linear").ffill().bfill()
    else:
        df["inflation_index"] = np.nan

    return df
