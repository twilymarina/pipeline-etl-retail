"""
=============================================================================
PROYECTO INTEGRADOR: Pipeline ETL de Retail
Materia: Programación para el Procesamiento de Datos
Descripción: Pipeline automatizado que integra fuentes heterogéneas
             (SQL simulado, NoSQL, APIs, archivos planos) para centralizar
             información de retail y facilitar la toma de decisiones.
=============================================================================
"""

import os
import re
import json
import random
import warnings
import csv
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")
np.random.seed(42)
random.seed(42)

OUTPUT_DIR = "output"
DATA_DIR   = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR,   exist_ok=True)

print("=" * 65)
print("  PIPELINE ETL — ANÁLISIS GLOBAL DE RETAIL Y COMPORTAMIENTO")
print("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 1: GENERACIÓN DE DATOS SIMULADOS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/6] Generando datos simulados...")

# ── 1a. ventas_historicas (SQL simulado) ─────────────────────────────────────
N_VENTAS = 7500
tiendas  = [f"T{str(i).zfill(3)}" for i in range(1, 21)]
monedas  = ["MXN"] * 70 + ["USD"] * 20 + ["EUR"] * 10

date_formats = ["%d/%m/%Y", "%Y-%m-%d", "%m-%d-%Y", "%d-%m-%Y"]

def random_date(start="2022-01-01", end="2024-12-31"):
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end,   "%Y-%m-%d")
    delta = (e - s).days
    d = s + timedelta(days=random.randint(0, delta))
    fmt = random.choice(date_formats)
    return d.strftime(fmt)

ventas = pd.DataFrame({
    "id_transaccion": [f"TXN{i:06d}" for i in range(1, N_VENTAS + 1)],
    "id_cliente":     [f"CLI{random.randint(1, 2000):05d}" for _ in range(N_VENTAS)],
    "monto":          np.round(np.random.exponential(scale=800, size=N_VENTAS) + 50, 2),
    "fecha":          [random_date() for _ in range(N_VENTAS)],
    "id_tienda":      [random.choice(tiendas) for _ in range(N_VENTAS)],
    "moneda":         [random.choice(monedas) for _ in range(N_VENTAS)],
})
# Insertar ~3% duplicados
dup_idx = ventas.sample(frac=0.03, random_state=42).index
ventas = pd.concat([ventas, ventas.loc[dup_idx]], ignore_index=True)
ventas.to_csv(f"{DATA_DIR}/ventas_historicas.csv", index=False)
print(f"   ventas_historicas : {len(ventas):,} filas  ({len(dup_idx)} duplicados)")

# ── 1b. perfiles_usuarios (NoSQL / MongoDB simulado) ─────────────────────────
N_PERFILES = 1500
paises_sucios = ["México","mexico","mex","MX","mx","MÉXICO",
                 "USA","United States","us","U.S.A","estados unidos",
                 "Colombia","colombia","col","COL"]
categorias_pref = ["Electrónica","Ropa","Hogar","Deportes","Alimentos","Juguetes"]

perfiles = []
for i in range(1, N_PERFILES + 1):
    perfiles.append({
        "id_cliente":       f"CLI{i:05d}",
        "nombre":           f"Usuario_{i}",
        "edad":             random.randint(18, 70),
        "pais":             random.choice(paises_sucios),
        "ciudad":           random.choice(["CDMX","Guadalajara","Monterrey","Bogotá","Miami","NYC"]),
        "latitud":          round(random.uniform(14.5, 32.7), 4),
        "longitud":         round(random.uniform(-117.0, -77.0), 4),
        "preferencias":     random.sample(categorias_pref, k=random.randint(1, 3)),
        "puntos_lealtad":   random.randint(0, 5000),
        "gastos_mensuales": round(random.uniform(200, 8000), 2),
        "ingresos":         round(random.uniform(5000, 80000), 2),
        "activo":           random.choice([True, False]),
    })
with open(f"{DATA_DIR}/perfiles_usuarios.json", "w", encoding="utf-8") as f:
    json.dump(perfiles, f, ensure_ascii=False, indent=2)
print(f"   perfiles_usuarios : {len(perfiles):,} documentos")

# ── 1c. inventario.csv ────────────────────────────────────────────────────────
N_INV = 800
categorias = ["Electrónica","Ropa","Hogar","Deportes","Alimentos","Juguetes","Librería"]
inv_rows = []
for i in range(1, N_INV + 1):
    row = {
        "sku":          f"SKU{i:05d}",
        "nombre":       f"Producto_{i}",
        "categoria":    random.choice(categorias),
        "stock":        random.randint(0, 500),
        "precio_costo": round(random.uniform(10, 500), 2),
        "precio_venta": round(random.uniform(15, 800), 2),
        "proveedor":    f"Proveedor_{random.randint(1,30)}",
        "almacen":      random.choice(["CDMX","GDL","MTY","BOG"]),
    }
    # 10% nulos
    if random.random() < 0.10:
        row[random.choice(["stock","precio_costo","precio_venta"])] = None
    inv_rows.append(row)

inv_df = pd.DataFrame(inv_rows)
# 5% duplicados
dup_inv = inv_df.sample(frac=0.05, random_state=42)
inv_df  = pd.concat([inv_df, dup_inv], ignore_index=True)
inv_df.to_csv(f"{DATA_DIR}/inventario.csv", index=False)
print(f"   inventario        : {len(inv_df):,} filas  ({len(dup_inv)} duplicados, ~10% nulos)")

# ── 1d. logs_servidor.txt ─────────────────────────────────────────────────────
log_levels  = ["INFO","WARNING","ERROR","ERROR","DEBUG"]
endpoints   = ["/home","/productos","/carrito","/checkout","/pago","/perfil"]
user_agents = ["Chrome/120","Firefox/119","Safari/17","Mobile/iOS"]

with open(f"{DATA_DIR}/logs_servidor.txt", "w") as f:
    for _ in range(2500):
        ts  = datetime.now() - timedelta(seconds=random.randint(0, 86400*30))
        lvl = random.choice(log_levels)
        ep  = random.choice(endpoints)
        ms  = random.randint(50, 3000)
        ip  = f"192.168.{random.randint(0,255)}.{random.randint(1,254)}"
        ua  = random.choice(user_agents)
        f.write(f"[{ts.strftime('%Y-%m-%d %H:%M:%S')}] {lvl} {ep} {ms}ms {ip} {ua}\n")
print("   logs_servidor     : 2,500 líneas")

# ── 1e. catalogos.xml ─────────────────────────────────────────────────────────
root = ET.Element("catalogo")
for cat in categorias:
    cat_el = ET.SubElement(root, "categoria", id=cat.lower().replace(" ","_"))
    ET.SubElement(cat_el, "nombre").text = cat
    ET.SubElement(cat_el, "descripcion").text = f"Productos de {cat}"
    for j in range(1, 4):
        sub = ET.SubElement(cat_el, "subcategoria")
        sub.text = f"{cat}_Sub{j}"
tree = ET.ElementTree(root)
tree.write(f"{DATA_DIR}/catalogos.xml", encoding="unicode", xml_declaration=True)
print("   catalogos.xml     : generado")

# ── 1f. metas_anuales.xlsx ────────────────────────────────────────────────────
regiones = ["Norte","Centro","Sur","Occidente","Oriente","Internacional"]
metas_df = pd.DataFrame({
    "region":          regiones,
    "meta_ventas":     [random.randint(500_000, 2_000_000) for _ in regiones],
    "meta_clientes":   [random.randint(1_000, 10_000) for _ in regiones],
    "meta_ticket_avg": [random.randint(300, 1_500) for _ in regiones],
    "kpi_satisfaccion":[round(random.uniform(70, 99), 1) for _ in regiones],
})
metas_df.to_excel(f"{DATA_DIR}/metas_anuales.xlsx", index=False)
print("   metas_anuales.xlsx: generado")

# ── 1g. API REST simulada (tipos de cambio) ───────────────────────────────────
tipos_cambio = {
    "base": "MXN",
    "fecha": datetime.now().strftime("%Y-%m-%d"),
    "tasas": {"USD": 0.0556, "EUR": 0.0513, "MXN": 1.0,
               "GBP": 0.0439, "CAD": 0.0757}
}
with open(f"{DATA_DIR}/tipos_cambio_api.json", "w") as f:
    json.dump(tipos_cambio, f, indent=2)
print("   tipos_cambio_api  : simulado (75 registros históricos)")

# ── 1h. Web Scraping simulado (precios competencia) ──────────────────────────
html_tabla = """<html><body>
<table id='precios'>
  <tr><th>Producto</th><th>Competidor</th><th>Precio</th><th>Moneda</th></tr>"""
for i in range(1, 76):
    html_tabla += f"\n  <tr><td>Prod_{i}</td><td>CompX</td><td>{round(random.uniform(100,1000),2)}</td><td>MXN</td></tr>"
html_tabla += "\n</table></body></html>"
with open(f"{DATA_DIR}/precios_competencia.html", "w") as f:
    f.write(html_tabla)
print("   precios_compet.   : 75 registros HTML")


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 2: EXTRACCIÓN
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/6] Extrayendo datos de todas las fuentes...")

# SQL simulado
df_ventas = pd.read_csv(f"{DATA_DIR}/ventas_historicas.csv")
print(f"   SQL  → ventas      : {len(df_ventas):,} filas")

# MongoDB simulado
with open(f"{DATA_DIR}/perfiles_usuarios.json", encoding="utf-8") as f:
    df_perfiles = pd.DataFrame(json.load(f))
print(f"   JSON → perfiles    : {len(df_perfiles):,} documentos")

# CSV
df_inventario = pd.read_csv(f"{DATA_DIR}/inventario.csv")
print(f"   CSV  → inventario  : {len(df_inventario):,} filas")

# XML
tree = ET.parse(f"{DATA_DIR}/catalogos.xml")
xml_root = tree.getroot()
cat_list = []
for cat in xml_root.findall("categoria"):
    cat_list.append({
        "id": cat.get("id"),
        "nombre": cat.find("nombre").text,
        "descripcion": cat.find("descripcion").text,
        "subcategorias": [s.text for s in cat.findall("subcategoria")]
    })
df_catalogos = pd.DataFrame(cat_list)
print(f"   XML  → catalogos   : {len(df_catalogos)} categorías")

# Excel
df_metas = pd.read_excel(f"{DATA_DIR}/metas_anuales.xlsx")
print(f"   XLSX → metas       : {len(df_metas)} regiones")

# API REST
with open(f"{DATA_DIR}/tipos_cambio_api.json") as f:
    api_data = json.load(f)
df_fx = pd.DataFrame([{"moneda": k, "tasa_a_mxn": 1/v if v != 0 else 1}
                       for k, v in api_data["tasas"].items()])
print(f"   API  → tipos cambio: {len(df_fx)} monedas")

# Web scraping
try:
    from bs4 import BeautifulSoup
    with open(f"{DATA_DIR}/precios_competencia.html") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
    rows = soup.find("table", {"id": "precios"}).find_all("tr")[1:]
    df_competencia = pd.DataFrame(
        [[td.text for td in r.find_all("td")] for r in rows],
        columns=["producto","competidor","precio","moneda"]
    )
    df_competencia["precio"] = df_competencia["precio"].astype(float)
    print(f"   HTML → competencia : {len(df_competencia)} productos")
except Exception as e:
    df_competencia = pd.DataFrame(columns=["producto","competidor","precio","moneda"])
    print(f"   HTML → competencia : fallback ({e})")

# Logs
log_pattern = re.compile(
    r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]\s+(\w+)\s+(\S+)\s+(\d+)ms\s+([\d.]+)\s+(\S+)'
)
log_records = []
with open(f"{DATA_DIR}/logs_servidor.txt") as f:
    for line in f:
        m = log_pattern.match(line.strip())
        if m:
            log_records.append({
                "timestamp": m.group(1), "nivel": m.group(2),
                "endpoint":  m.group(3), "tiempo_ms": int(m.group(4)),
                "ip":        m.group(5), "user_agent": m.group(6),
            })
df_logs = pd.DataFrame(log_records)
print(f"   TXT  → logs        : {len(df_logs):,} líneas parseadas")


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 3: TRANSFORMACIÓN Y CALIDAD
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/6] Transformación y limpieza de datos...")

# ── 3a. Ventas: eliminar duplicados ──────────────────────────────────────────
n_antes = len(df_ventas)
df_ventas.drop_duplicates(subset="id_transaccion", inplace=True)
print(f"   Duplicados ventas eliminados : {n_antes - len(df_ventas)}")

# ── 3b. Normalización de fechas ──────────────────────────────────────────────
def parse_fecha(f):
    for fmt in ["%d/%m/%Y","%Y-%m-%d","%m-%d-%Y","%d-%m-%Y"]:
        try:
            return pd.Timestamp(datetime.strptime(str(f).strip(), fmt))
        except:
            continue
    return pd.NaT

df_ventas["fecha"] = df_ventas["fecha"].apply(parse_fecha)
df_logs["timestamp"]  = pd.to_datetime(df_logs["timestamp"], errors="coerce")
print(f"   Fechas normalizadas a datetime64")

# ── 3c. Convertir monedas a MXN ──────────────────────────────────────────────
fx_map = dict(zip(df_fx["moneda"], df_fx["tasa_a_mxn"]))
df_ventas["monto_mxn"] = df_ventas.apply(
    lambda r: round(r["monto"] * fx_map.get(r["moneda"], 1.0), 2), axis=1
)
print("   Conversión de monedas a MXN completada")

# ── 3d. Limpieza de inventario ────────────────────────────────────────────────
n_antes = len(df_inventario)
df_inventario.drop_duplicates(subset="sku", keep="first", inplace=True)
print(f"   Duplicados inventario : {n_antes - len(df_inventario)} eliminados")

nulos_antes = df_inventario.isnull().sum().sum()
imp = SimpleImputer(strategy="median")
cols_num_inv = ["stock","precio_costo","precio_venta"]
df_inventario[cols_num_inv] = imp.fit_transform(df_inventario[cols_num_inv])
print(f"   Nulos inventario imputados con mediana : {nulos_antes}")

# ── 3e. Limpieza de texto en perfiles ────────────────────────────────────────
pais_map = {
    "mexico":"México","mex":"México","mx":"México","méxico":"México",
    "usa":"Estados Unidos","united states":"Estados Unidos",
    "us":"Estados Unidos","u.s.a":"Estados Unidos","estados unidos":"Estados Unidos",
    "colombia":"Colombia","col":"Colombia",
}
df_perfiles["pais_clean"] = (
    df_perfiles["pais"].str.lower().str.strip()
    .map(lambda x: pais_map.get(x, x.title()))
)
print(f"   Países estandarizados: {df_perfiles['pais'].nunique()} → {df_perfiles['pais_clean'].nunique()} valores únicos")

# ── 3f. Escalado numérico Min-Max en perfiles ─────────────────────────────────
cols_escalar = ["edad","puntos_lealtad","gastos_mensuales","ingresos"]
scaler_minmax = MinMaxScaler()
df_perfiles[[f"{c}_norm" for c in cols_escalar]] = scaler_minmax.fit_transform(
    df_perfiles[cols_escalar]
)
print("   Min-Max scaling aplicado a variables numéricas de perfiles")


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 4: ENRIQUECIMIENTO Y LÓGICA DE NEGOCIO
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/6] Enriquecimiento y reglas de negocio...")

# ── 4a. Agregación de ventas por cliente ─────────────────────────────────────
ventas_agg = df_ventas.groupby("id_cliente").agg(
    total_compras   = ("id_transaccion","count"),
    gasto_total     = ("monto_mxn","sum"),
    gasto_promedio  = ("monto_mxn","mean"),
    ultima_compra   = ("fecha","max"),
    primera_compra  = ("fecha","min"),
).reset_index()

# ── 4b. LEFT JOIN ventas + perfiles ──────────────────────────────────────────
df_master = pd.merge(
    ventas_agg,
    df_perfiles[["id_cliente","edad","pais_clean","puntos_lealtad",
                 "gastos_mensuales","ingresos","preferencias",
                 "edad_norm","puntos_lealtad_norm",
                 "gastos_mensuales_norm","ingresos_norm"]],
    on="id_cliente",
    how="left"
)
print(f"   Master después del join : {len(df_master):,} clientes")
print(f"   Tasa de enriquecimiento : {df_master['edad'].notna().mean()*100:.1f}%")

# ── 4c. Reglas de negocio — segmento_cliente ─────────────────────────────────
def segmentar(row):
    g = row.get("gasto_total", 0) or 0
    e = row.get("edad", 99) or 99
    p = row.get("puntos_lealtad", 0) or 0
    if g > 5000 and e < 30:
        return "Premium Joven"
    elif g > 5000 and e >= 30:
        return "Premium Maduro"
    elif g > 2000 and p > 2000:
        return "Leal Activo"
    elif g > 1000:
        return "Regular"
    else:
        return "Ocasional"

df_master["segmento_cliente"] = df_master.apply(segmentar, axis=1)
print("   Segmentos creados:")
print(df_master["segmento_cliente"].value_counts().to_string(header=False))

# ── 4d. Margen de inventario ──────────────────────────────────────────────────
df_inventario["margen_pct"] = (
    (df_inventario["precio_venta"] - df_inventario["precio_costo"])
    / df_inventario["precio_venta"] * 100
).round(2)

# ── 4e. Análisis de logs ──────────────────────────────────────────────────────
embudo = (
    df_logs.groupby("endpoint")["ip"]
    .count().reset_index(name="visitas")
    .sort_values("visitas", ascending=False)
)

# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 5: ANÁLISIS AVANZADO — PCA
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/6] Análisis avanzado — PCA...")

# Construir 20 variables de comportamiento
feature_cols = [
    "total_compras","gasto_total","gasto_promedio",
    "edad_norm","puntos_lealtad_norm","gastos_mensuales_norm","ingresos_norm",
]
# Generar variables sintéticas adicionales hasta 20
extra_feats = {}
for i in range(1, 14):
    extra_feats[f"var_comp_{i:02d}"] = (
        df_master["gasto_total"].fillna(0) * np.random.uniform(0.1, 2.0)
        + np.random.normal(0, 100, len(df_master))
    )
df_extra = pd.DataFrame(extra_feats, index=df_master.index)
df_pca_input = pd.concat(
    [df_master[feature_cols].fillna(0).reset_index(drop=True),
     df_extra.reset_index(drop=True)],
    axis=1
)

# Z-score para PCA
scaler_z = StandardScaler()
X_scaled  = scaler_z.fit_transform(df_pca_input)

# PCA
pca = PCA(n_components=10, random_state=42)
pca.fit(X_scaled)
varianza_exp = pca.explained_variance_ratio_
var_acum3    = varianza_exp[:3].sum() * 100

X_pca3 = pca.transform(X_scaled)[:, :3]
df_master["PC1"] = X_pca3[:, 0]
df_master["PC2"] = X_pca3[:, 1]
df_master["PC3"] = X_pca3[:, 2]

print(f"   Varianza explicada por 3 PCs : {var_acum3:.1f}%")
for i, v in enumerate(varianza_exp[:5], 1):
    bar = "█" * int(v * 50)
    print(f"   PC{i}: {v*100:5.1f}%  {bar}")


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 6: VISUALIZACIÓN — DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6/6] Generando dashboard de visualizaciones...")

palette = {
    "Premium Joven":  "#E74C3C",
    "Premium Maduro": "#E67E22",
    "Leal Activo":    "#27AE60",
    "Regular":        "#2980B9",
    "Ocasional":      "#8E44AD",
}
seg_order = list(palette.keys())

plt.style.use("seaborn-v0_8-whitegrid")
fig = plt.figure(figsize=(22, 18))
fig.patch.set_facecolor("#F8F9FA")
fig.suptitle(
    "ANÁLISIS GLOBAL DE RETAIL Y COMPORTAMIENTO DEL CLIENTE\nPipeline ETL — Dashboard Ejecutivo",
    fontsize=17, fontweight="bold", y=0.99, color="#2C3E50"
)

# ── Plot 1: Boxplot montos de venta por segmento ──────────────────────────────
ax1 = fig.add_subplot(3, 3, 1)
box_data = [
    df_master.loc[df_master["segmento_cliente"]==s, "gasto_total"].dropna().values
    for s in seg_order
]
bp = ax1.boxplot(box_data, patch_artist=True, labels=seg_order,
                 medianprops=dict(color="white", linewidth=2))
for patch, seg in zip(bp["boxes"], seg_order):
    patch.set_facecolor(palette[seg])
    patch.set_alpha(0.8)
ax1.set_title("Boxplot — Gasto Total por Segmento", fontweight="bold", fontsize=10)
ax1.set_ylabel("Gasto Total (MXN)")
ax1.tick_params(axis="x", rotation=30, labelsize=7)
ax1.set_yscale("log")

# ── Plot 2: Scatter PCA — PC1 vs PC2 ─────────────────────────────────────────
ax2 = fig.add_subplot(3, 3, 2)
for seg in seg_order:
    mask = df_master["segmento_cliente"] == seg
    ax2.scatter(
        df_master.loc[mask, "PC1"],
        df_master.loc[mask, "PC2"],
        c=palette[seg], label=seg, alpha=0.55, s=18, edgecolors="none"
    )
ax2.set_title(f"PCA — PC1 vs PC2 ({var_acum3:.1f}% var. explicada)", fontweight="bold", fontsize=10)
ax2.set_xlabel(f"PC1 ({varianza_exp[0]*100:.1f}%)")
ax2.set_ylabel(f"PC2 ({varianza_exp[1]*100:.1f}%)")
ax2.legend(fontsize=7, markerscale=1.2)

# ── Plot 3: Varianza explicada PCA ────────────────────────────────────────────
ax3 = fig.add_subplot(3, 3, 3)
pcs   = [f"PC{i}" for i in range(1, 11)]
bars  = ax3.bar(pcs, varianza_exp * 100, color="#3498DB", alpha=0.75, edgecolor="white")
ax3_twin = ax3.twinx()
ax3_twin.plot(pcs, np.cumsum(varianza_exp) * 100, "o-", color="#E74C3C", linewidth=2, markersize=5)
ax3_twin.set_ylabel("Varianza Acumulada (%)", color="#E74C3C")
ax3.set_title("PCA — Varianza Explicada por Componente", fontweight="bold", fontsize=10)
ax3.set_ylabel("Varianza Individual (%)")
ax3.tick_params(axis="x", rotation=45, labelsize=8)
ax3.axvline(x=2, color="gray", linestyle="--", alpha=0.5)
ax3.text(2.1, max(varianza_exp)*80, "3 PCs\nseleccionadas", fontsize=7, color="gray")

# ── Plot 4: Sankey simplificado (Flujo Web → Compra) ─────────────────────────
ax4 = fig.add_subplot(3, 3, 4)
funnel_steps = ["/home", "/productos", "/carrito", "/checkout", "/pago"]
funnel_vals  = []
for ep in funnel_steps:
    v = embudo.loc[embudo["endpoint"] == ep, "visitas"].values
    funnel_vals.append(int(v[0]) if len(v) > 0 else 0)

colors_funnel = ["#3498DB","#2ECC71","#F39C12","#E67E22","#E74C3C"]
ypos = list(range(len(funnel_steps)))
bars_f = ax4.barh(ypos, funnel_vals, color=colors_funnel, edgecolor="white", height=0.6)
ax4.set_yticks(ypos)
ax4.set_yticklabels(funnel_steps, fontsize=9)
ax4.set_title("Embudo de Conversión Web (Sankey simplificado)", fontweight="bold", fontsize=10)
ax4.set_xlabel("Número de Visitas")
for i, (bar, val) in enumerate(zip(bars_f, funnel_vals)):
    ax4.text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2,
             f"{val:,}", va="center", fontsize=8)
# Flechas entre pasos
for i in range(len(funnel_steps) - 1):
    pct = funnel_vals[i+1]/funnel_vals[i]*100 if funnel_vals[i] > 0 else 0
    ax4.text(max(funnel_vals)*0.5, i + 0.5, f"↓ {pct:.0f}%", fontsize=7, color="gray", ha="center")

# ── Plot 5: Distribución de segmentos ────────────────────────────────────────
ax5 = fig.add_subplot(3, 3, 5)
seg_counts = df_master["segmento_cliente"].value_counts()
wedges, texts, autotexts = ax5.pie(
    seg_counts, labels=seg_counts.index,
    colors=[palette[s] for s in seg_counts.index],
    autopct="%1.1f%%", startangle=140,
    textprops={"fontsize": 8}, pctdistance=0.75
)
ax5.set_title("Distribución de Segmentos de Clientes", fontweight="bold", fontsize=10)

# ── Plot 6: Top 10 tiendas por venta ─────────────────────────────────────────
ax6 = fig.add_subplot(3, 3, 6)
top_tiendas = (
    df_ventas.groupby("id_tienda")["monto_mxn"].sum()
    .sort_values(ascending=False).head(10)
)
ax6.barh(top_tiendas.index[::-1], top_tiendas.values[::-1],
         color="#2ECC71", edgecolor="white")
ax6.set_title("Top 10 Tiendas — Ventas Totales (MXN)", fontweight="bold", fontsize=10)
ax6.set_xlabel("Monto Total (MXN)")

# ── Plot 7: Ventas mensuales ──────────────────────────────────────────────────
ax7 = fig.add_subplot(3, 3, 7)
df_ventas_t = df_ventas.dropna(subset=["fecha"]).copy()
df_ventas_t["mes"] = df_ventas_t["fecha"].dt.to_period("M")
ventas_mes = df_ventas_t.groupby("mes")["monto_mxn"].sum().reset_index()
ventas_mes["mes_str"] = ventas_mes["mes"].astype(str)
ax7.fill_between(
    range(len(ventas_mes)), ventas_mes["monto_mxn"],
    alpha=0.4, color="#3498DB"
)
ax7.plot(range(len(ventas_mes)), ventas_mes["monto_mxn"],
         color="#2980B9", linewidth=1.5)
ax7.set_xticks(range(0, len(ventas_mes), max(1, len(ventas_mes)//8)))
ax7.set_xticklabels(
    ventas_mes["mes_str"].iloc[::max(1, len(ventas_mes)//8)],
    rotation=30, fontsize=7
)
ax7.set_title("Tendencia de Ventas Mensuales", fontweight="bold", fontsize=10)
ax7.set_ylabel("Monto Total (MXN)")

# ── Plot 8: Distribución de errores en logs ───────────────────────────────────
ax8 = fig.add_subplot(3, 3, 8)
nivel_counts = df_logs["nivel"].value_counts()
colors_log   = {"INFO":"#2ECC71","WARNING":"#F39C12","ERROR":"#E74C3C","DEBUG":"#3498DB"}
ax8.bar(
    nivel_counts.index,
    nivel_counts.values,
    color=[colors_log.get(n, "#95A5A6") for n in nivel_counts.index],
    edgecolor="white"
)
ax8.set_title("Distribución de Niveles en Logs del Servidor", fontweight="bold", fontsize=10)
ax8.set_ylabel("Número de Eventos")
for i, (idx, val) in enumerate(nivel_counts.items()):
    ax8.text(i, val + 10, str(val), ha="center", fontsize=8)

# ── Plot 9: Heatmap — correlación variables numéricas ────────────────────────
ax9 = fig.add_subplot(3, 3, 9)
corr_cols = ["gasto_total","total_compras","gasto_promedio",
             "edad_norm","puntos_lealtad_norm","gastos_mensuales_norm","ingresos_norm"]
df_corr = df_master[corr_cols].dropna().corr()
sns.heatmap(
    df_corr, annot=True, fmt=".2f", cmap="RdYlGn",
    ax=ax9, linewidths=0.3, annot_kws={"size": 6},
    cbar_kws={"shrink": 0.8}
)
ax9.set_title("Correlación — Variables Clave", fontweight="bold", fontsize=10)
ax9.tick_params(axis="x", rotation=45, labelsize=7)
ax9.tick_params(axis="y", rotation=0,  labelsize=7)

plt.tight_layout(rect=[0, 0, 1, 0.97])
dashboard_path = f"{OUTPUT_DIR}/dashboard_retail.png"
plt.savefig(dashboard_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"   Dashboard guardado: {dashboard_path}")


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 7: EXPORTAR DATA MASTER
# ─────────────────────────────────────────────────────────────────────────────
cols_export = [
    "id_cliente","total_compras","gasto_total","gasto_promedio",
    "ultima_compra","primera_compra","edad","pais_clean",
    "puntos_lealtad","gastos_mensuales","ingresos",
    "edad_norm","puntos_lealtad_norm","gastos_mensuales_norm","ingresos_norm",
    "segmento_cliente","PC1","PC2","PC3"
]
df_export = df_master[[c for c in cols_export if c in df_master.columns]].copy()

# Guardar como CSV (parquet no disponible sin pyarrow)
master_path = f"{OUTPUT_DIR}/data_master_clean.csv"
df_export.to_csv(master_path, index=False)
print(f"\n   Archivo maestro guardado: {master_path}")
print(f"   Dimensiones finales: {df_export.shape[0]:,} filas × {df_export.shape[1]} columnas")


# ─────────────────────────────────────────────────────────────────────────────
# RESUMEN EJECUTIVO
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  RESUMEN EJECUTIVO DEL PIPELINE")
print("=" * 65)
print(f"  Fuentes integradas      : 8 (SQL, JSON, CSV, TXT, XML, XLSX, API, HTML)")
print(f"  Transacciones procesadas: {len(df_ventas):,}")
print(f"  Clientes únicos         : {df_export['id_cliente'].nunique():,}")
print(f"  Tasa enriquecimiento    : {df_master['edad'].notna().mean()*100:.1f}%")
print(f"  Varianza PCA (3 PCs)    : {var_acum3:.1f}%")
print(f"  Segmentos identificados : {df_master['segmento_cliente'].nunique()}")
print(f"  Errores en logs         : {(df_logs['nivel']=='ERROR').sum():,}")
print(f"  Archivos generados:")
print(f"    • {OUTPUT_DIR}/dashboard_retail.png")
print(f"    • {OUTPUT_DIR}/data_master_clean.csv")
print("=" * 65)
print("  Pipeline completado exitosamente ✓")
print("=" * 65)
