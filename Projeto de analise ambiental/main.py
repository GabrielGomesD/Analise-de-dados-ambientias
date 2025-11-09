"""
Análise de Dados Ambientais para Soluções Sustentáveis nas Cidades
Projeto: Data Science Fundamentals
Autor: Gabriel Dino Gomes
Uso:
    python main.py --input data/environmental_data.csv --output outputs
Caso o arquivo CSV de entrada não seja encontrado, o script gera uma amostra de conjunto de dados sintéticos.
Outputs:
 - dados_limpos.csv
 - EDA plots (png)
 - modelo_resultados.txt
 - relatorio.pdf
"""
import os
import argparse

# Integrações de API opcionais

try:
    from data_sources import fetch_cetesb_layer, fetch_aqicn_station, fetch_iema_placeholder
except Exception:

    # As fontes de dados podem não estar disponíveis em ambientes onde a rede está desativada.

    fetch_cetesb_layer = None
    fetch_aqicn_station = None
    fetch_iema_placeholder = None


import os, sys, argparse
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def generate_synthetic(path):
    """Gera um conjunto de dados ambientais sintéticos e salva eles no path."""
    rng = np.random.RandomState(42)
    n = 180 #Numero de dias
    dates = pd.date_range(start="2025-08-01", periods=n, freq="D")
    city_zone = rng.choice(["Centro","Norte","Sul","Leste","Oeste"], size=n)
    temp = 20 + 10 * np.sin(np.linspace(0, 20, n)) + rng.normal(0,2,n)
    humidity = 40 + 20 * np.cos(np.linspace(0, 10, n)) + rng.normal(0,5,n)
    pm25 = np.abs(15 + 0.5*temp + rng.normal(0,5,n) + (city_zone=="Centro")*10)
    energy_consumption = 50 + 2*temp + 0.5*humidity + 0.3*pm25 + rng.normal(0,10,n)
    waste_kg = np.abs(0.5*energy_consumption + rng.normal(0,5,n))
    water_usage = 200 + 5*humidity + rng.normal(0,20,n)
    df = pd.DataFrame({
        "datetime": dates,
        "zone": city_zone,
        "temperature_C": np.round(temp,2),
        "humidity_pct": np.round(humidity,2),
        "pm25_ug_m3": np.round(pm25,2),
        "energy_kwh": np.round(energy_consumption,2),
        "waste_kg": np.round(waste_kg,2),
        "water_l": np.round(water_usage,2)
    })
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Synthetic dataset saved to {path}")
    return df

def load_data(path, args=None):

    # Se --use-apis estiver definido e data_sources disponível, tentar buscar dados via APIs e mesclar

    if args is not None and getattr(args, "use_apis", False):
        print("--use-apis ativado: tentando obter dados via APIs...")
        frames = []

        # CETESB

        if fetch_cetesb_layer is not None and args.cetesb_layer is not None:
            try:
                print(f"Fetching CETESB layer {args.cetesb_layer}...")
                df_c = fetch_cetesb_layer(layer_id=int(args.cetesb_layer))
                if not df_c.empty:
                    frames.append(df_c)
            except Exception as e:
                print("Falha ao buscar CETESB:", e)

        # AQICN

        if fetch_aqicn_station is not None and args.aqicn_token and args.aqicn_station:
            try:
                print(f"Fetching AQICN station {args.aqicn_station}...")
                df_a = fetch_aqicn_station(args.aqicn_station, args.aqicn_token)
                if not df_a.empty:
                    frames.append(df_a)
            except Exception as e:
                print("Falha ao buscar AQICN:", e)

        # IEMA placeholder CSV

        if fetch_iema_placeholder is not None and args.iema_csv:
            try:
                print(f"Fetching IEMA CSV from {args.iema_csv}...")
                df_i = fetch_iema_placeholder(args.iema_csv)
                if not df_i.empty:
                    frames.append(df_i)
            except Exception as e:
                print("Falha ao buscar IEMA \(CSV\):", e)
        if frames:

            # Tente concatenar horizontalmente com junção externa (alinhar por data e hora ou criar um índice sintético).

            try:
                merged = pd.concat(frames, axis=0, ignore_index=True, sort=False)
                #  Analisa a coluna de data e hora, se presente
                if "datetime" in merged.columns:
                    merged["datetime"] = pd.to_datetime(merged["datetime"], errors="coerce")
                merged.to_csv(path, index=False)
                print(f"API data saved to {path}")
                return merged
            except Exception as e:
                print("Failed to merge API frames:", e)

        # Se disponível, utilize o arquivo local como alternativa.

    if not os.path.exists(path):
        print("CSV de entrada não encontrado. Gerando dataset sintético para demonstração.")
        return generate_synthetic(path)
    df = pd.read_csv(path, parse_dates=["datetime"]) if os.path.exists(path) else generate_synthetic(path)
    return df

def clean_data(df):
    df = df.copy()

    # Remove duplicados

    df = df.drop_duplicates()

    # Lidar com valores ausentes: - preencher os valores numéricos com a mediana e os valores categóricos com a moda.

    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(exclude=[np.number, "datetime"]).columns
    for c in num_cols:
        med = df[c].median()
        df[c] = df[c].fillna(med)
    for c in cat_cols:
        mode = df[c].mode().iloc[0] if not df[c].mode().empty else "unknown"
        df[c] = df[c].fillna(mode)

    # Criar funcionalidades por hora/dia

    if "datetime" in df.columns:
        df["day"] = df["datetime"].dt.day
        df["dayofweek"] = df["datetime"].dt.dayofweek
    return df

def eda_and_plots(df, outdir):
    os.makedirs(outdir, exist_ok=True)
    plots = []

    # séries temporais de PM2,5 e energia

    plt.figure(figsize=(10,4))
    df.set_index("datetime")["pm25_ug_m3"][:200].plot(title="PM2.5 (amostra) - Série temporal")
    plt.tight_layout()
    p1 = os.path.join(outdir, "pm25_timeseries.png")        
    plt.savefig(p1); plt.close(); plots.append(p1)
    plt.figure(figsize=(10,4))
    df.set_index("datetime")["energy_kwh"][:200].plot(title="Consumo de energia (amostra) - Série temporal")
    plt.tight_layout()
    p2 = os.path.join(outdir, "energia_timeseries.png")
    plt.savefig(p2); plt.close(); plots.append(p2)

    # dispersão pm25 vs energia

    plt.figure(figsize=(6,4))
    plt.scatter(df["pm25_ug_m3"], df["energy_kwh"], alpha=0.4)
    plt.xlabel("PM2.5 (µg/m³)"); plt.ylabel("Energia (kWh)"); plt.title("PM2.5 vs Consumo de energia")
    plt.tight_layout()
    p3 = os.path.join(outdir, "pm25_vs_energia.png")
    plt.savefig(p3); plt.close(); plots.append(p3)

    # Mapa de calor da matriz de correlação (usando matplotlib)

    corr = df.select_dtypes(include=[np.number]).corr()
    plt.figure(figsize=(6,5))
    plt.imshow(corr, vmin=-1, vmax=1)
    plt.colorbar()
    plt.xticks(range(len(corr)), corr.columns, rotation=90)
    plt.yticks(range(len(corr)), corr.columns)
    plt.title("Matriz de correlação (features numericas)")
    plt.tight_layout()
    p4 = os.path.join(outdir, "Matriz de correlação.png")
    plt.savefig(p4); plt.close(); plots.append(p4)
    return plots, corr

def detect_anomalies(df, outdir):
    os.makedirs(outdir, exist_ok=True)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    iso = IsolationForest(contamination=0.01, random_state=42)
    X = df[num_cols].fillna(0)
    iso.fit(X)
    scores = iso.decision_function(X)
    preds = iso.predict(X)
    df["anomaly_score"] = scores
    df["anomaly"] = preds == -1

    # salvar gráfico de amostra de anomalia (energia)

    plt.figure(figsize=(10,4))
    plt.plot(df["datetime"], df["energy_kwh"], label="energy_kwh")
    plt.scatter(df.loc[df["anomaly"], "datetime"], df.loc[df["anomaly"], "energy_kwh"], color="red", label="anomaly")
    plt.legend()
    plt.title("Anomalias no consumo de energia (pontos em vermelho)")
    plt.tight_layout()
    p = os.path.join(outdir, "anomalias_energia.png")
    plt.savefig(p); plt.close()
    return df, p

def simple_model(df, outdir):

    # Prever energia_kWh usando outros recursos

    df_model = df.dropna().copy()
    features = ["temperature_C", "humidity_pct", "pm25_ug_m3", "day", "dayofweek", "water_l"]
    for f in features:
        if f not in df_model.columns:
            df_model[f] = 0
    X = df_model[features]
    y = df_model["energy_kwh"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # importâncias do recurso

    importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)

    # salvar resultados

    res_path = os.path.join(outdir, "modelo_resultados.txt")
    with open(res_path, "w") as f:
        f.write(f"RandomForest regression to predict energy_kwh\\nMSE: {mse:.3f}\\nR2: {r2:.3f}\\n\\nFeature importances:\\n")
        f.write(importances.to_string())
    return model, res_path, mse, r2, importances

def build_pdf_report(df, plots, corr, model_info, out_pdf):
    with PdfPages(out_pdf) as pdf:

        # página 1 - título

        plt.figure(figsize=(8.27,11.69))
        plt.axis("off")
        text = [
            "Análise de Dados Ambientais para Soluções Sustentáveis nas Cidades",
            "",
            "Resumo do projeto e principais etapas do ciclo de vida da ciência de dados:",
            "- Entendimento do problema; Coleta; Preparação; Análise; Modelagem; Comunicação",
            "",
            f"Gerado em: {datetime.now().isoformat()}"
        ]
        plt.text(0.1, 0.8, "\\n".join(text), fontsize=12)
        pdf.savefig(); plt.close()

        # adicionar gráficos

        for p in plots:
            img = plt.imread(p)
            plt.figure(figsize=(8.27,11.69))
            plt.imshow(img)
            plt.axis("off")
            pdf.savefig(); plt.close()

        # página de correlação com tabela

        plt.figure(figsize=(8.27,11.69))
        plt.axis("off")
        plt.text(0.05,0.95,"Matriz de correlação (features numericas):", fontsize=12)
        plt.table(cellText=np.round(corr.values,2).tolist(),
                  colLabels=corr.columns, rowLabels=corr.columns, loc="center", cellLoc="center")
        pdf.savefig(); plt.close()

        # resumo do modelo

        mse, r2, importances = model_info
        plt.figure(figsize=(8.27,11.69))
        plt.axis("off")
        txt = f"Modelo: RandomForestRegressor\\nMSE: {mse:.3f}\\nR2: {r2:.3f}\\n\\nImportância de features:\\n" + importances.to_string()
        plt.text(0.05, 0.9, txt, fontsize=10)
        pdf.savefig(); plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/environmental_data.csv", help="Path to input CSV")
    parser.add_argument("--output", default="outputs", help="Output directory")
    parser.add_argument("--use-apis", action="store_true", help="If set, try to fetch data from CETESB/AQICN/IEMA (network required)")
    parser.add_argument("--aqicn-token", default=None, help="WAQI / AQICN API token (optional)")
    parser.add_argument("--aqicn-station", default=None, help="Station id for AQICN (optional), e.g. @362 or A405352 or city name")
    parser.add_argument("--cetesb-layer", default=None, help="CETESB layer id to fetch (e.g. 3)")
    parser.add_argument("--iema-csv", default=None, help="Optional CSV URL from IEMA or other source")
    args = parser.parse_args()
    df = load_data(args.input)
    df_clean = clean_data(df)
    os.makedirs(args.output, exist_ok=True)
    df_clean.to_csv(os.path.join(args.output, "dados_limpos.csv"), index=False)
    plots, corr = eda_and_plots(df_clean, args.output)
    df_anom, anom_plot = detect_anomalies(df_clean, args.output)
    model, res_path, mse, r2, importances = simple_model(df_anom, args.output)

    # gerar PDF do relatório

    model_info = (mse, r2, importances)
    report_path = os.path.join(args.output, "relatorio.pdf")
    build_pdf_report(df_clean, plots+[anom_plot], corr, model_info, report_path)
    print("Resultados gravados em:", args.output)
    print("Resultados do modelo:", res_path)
    print("Relatório:", report_path)

if __name__ == "__main__":
    main()
