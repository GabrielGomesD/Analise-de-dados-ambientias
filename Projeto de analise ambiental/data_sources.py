"""
data_sources.py

Funções para integrar APIs externas:
- CETESB ArcGIS MapServer (QUALAR)  -> fetch_cetesb_layer
- AQICN (WAQI) -> fetch_aqicn_station (requires token)
- IEMA / Plataforma da Qualidade do Ar -> fetch_iema_placeholder (placeholder that downloads CSV if URL provided)

Usage: see README or main.py integration examples.
"""
import requests
import pandas as pd
from datetime import datetime, timedelta
import os
from urllib.parse import urlencode

def fetch_cetesb_layer(layer_id=None, where="1=1", out_fields="*", f="json", server=None, timeout=30):
    """
    Buscar dados do MapServer ArcGIS da CETESB (QUALAR).
    Parameters:
      layer_id: int or None. If None, fetches MapServer root (not typical). Prefer specific layer.
      where: SQL where clause, e.g. "DATAHORA >= TIMESTAMP '2024-01-01 00:00:00'"
      out_fields: fields to return, default "*"
      f: output format, default json
      server: full base server URL (if None, default CETESB QUALAR endpoint used)
    Retorna: pandas.DataFrame
    Example layer URL:
      https://servicos.cetesb.sp.gov.br/arcgis/rest/services/QUALAR/CETESB_QUALAR/MapServer/3/query
    """
    if server is None:
        base = "https://servicos.cetesb.sp.gov.br/arcgis/rest/services/QUALAR/CETESB_QUALAR/MapServer"
    else:
        base = server.rstrip("/")
    if layer_id is None:
        # default to layer 3 (NO2 hourly 48h layer) as an example
        layer_id = 3
    url = f"{base}/{layer_id}/query"
    params = {
        "where": where,
        "outFields": out_fields,
        "f": f,
        "resultRecordCount": 10000,
        "orderByFields": "DATAHORA DESC"
    }
    resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    js = resp.json()
    # O JSON do ArcGIS possui lista 'features' com 'attributes'
    features = js.get("features", [])
    records = [feat.get("attributes", {}) for feat in features]
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame.from_records(records)
    # Tenta normalizar nomes comuns de data/hora
    for col in ["DATAHORA", "datahora", "DataHora", "date", "datetime"]:
        if col in df.columns:
            try:
                df["datetime"] = pd.to_datetime(df[col])
                break
            except Exception:
                pass
    return df

def fetch_aqicn_station(station_id, token, timeout=20):
    """
    Buscar dados em tempo real / mais recentes do AQICN (WAQI) (WAQI) API for a given station.
    station_id may be numeric id like '362' or station code like 'A405352' or '@362' depending on platform.
    Requer token WAQI (AQICN) (free registration).
    Returns pandas.DataFrame with a single-row latest measurement (plus nested history if available).
    Example URL:
      https://api.waqi.info/feed/@362/?token=__TOKEN__
      or https://api.waqi.info/feed/geo:lat;lon/?token=__TOKEN__
    """
    base = "https://api.waqi.info/feed"
    # allow passing full id or numeric
    if station_id.startswith("http") or station_id.startswith("https"):
        url = station_id
    else:
        # normalize: if startswith '@' keep, else try direct
        if station_id.startswith("@"):
            ident = station_id
        else:
            ident = station_id
        url = f"{base}/{ident}/"
    params = {"token": token}
    resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    js = resp.json()
    if js.get("status") != "ok":
        # return empty df with message
        return pd.DataFrame({"error":[js.get("data")]})
    data = js.get("data", {})
    # Flatten useful fields
    record = {}
    record["idx"] = data.get("idx")
    record["aqi"] = data.get("aqi")
    record["time"] = data.get("time", {}).get("s")
    record["city_name"] = data.get("city", {}).get("name")
    record["geo"] = ",".join(map(str, data.get("city", {}).get("geo", [])))
    iaqi = data.get("iaqi", {})
    for k,v in iaqi.items():
        record[f"iaqi_{k}"] = v.get("v")
    return pd.DataFrame([record])

def fetch_iema_placeholder(csv_url, timeout=30):
    """
    Função reservada para baixar um CSV a partir de uma URL (IEMA ou outro) from a given URL (IEMA or other),
    returning a pandas.DataFrame. If the resource is not CSV, user should adapt.
    """
    resp = requests.get(csv_url, timeout=timeout)
    resp.raise_for_status()
    # save to temp
    tmp = os.path.basename(csv_url.split("?")[0])
    with open(tmp, "wb") as f:
        f.write(resp.content)
    try:
        df = pd.read_csv(tmp, parse_dates=True, infer_datetime_format=True)
    finally:
        try:
            os.remove(tmp)
        except:
            pass
    return df

if __name__ == "__main__":
    print("módulo data_sources: funções auxiliares para CETESB, AQICN e IEMA (placeholder).")
