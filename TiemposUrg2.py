#  original web

#%%writefile TiemposUrg.py

# cargar librerias
import streamlit as st
import types  # Importa types en lugar de builtins
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple

def set_page_config():
    st.set_page_config(
        page_title="Tiempos de Urgencias",
        page_icon=":bar_chart:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown("<style> footer {visibility: hidden;} </style>", unsafe_allow_html=True)

set_page_config()

#Define una función de hash personalizada para tu función
def my_hash_func(func):
    return id(func)

@st.cache_resource(hash_funcs={types.FunctionType: my_hash_func})

def load_data(url):
    # Cargamos los datos desde el archivo Excel
    return pd.read_excel(url)

url = "https://github.com/Vitotoju/Compensar/raw/main/tiempos_urgencias.xlsx"
#url = "https://github.com/Vitotoju/tiemposurgencias/blob/main/tiempos_urgencias.xlsx"
dataset = load_data(url)
df = dataset



# Cadena Más Común (Moda)  -  para reemplazar los datos vacios con el valor más frecuente o la moda
promedio = df['Tiempo_Minutos_Total'].median()
df.loc[df['Tiempo_Minutos_Total'] > 420, 'Tiempo_Minutos_Total'] = promedio
df.loc[df['Tiempo_Minutos_Total'] < 0, 'Tiempo_Minutos_Total'] = promedio


#Convertir el tipo de datos al formato apropiado 
st.sidebar.header("Opciones a filtrar: ")

# Filtros Laterales
filtro_centro = st.sidebar.selectbox('Filtrar por Centro', ['Todos'] + df['CENTRO_ATENCION'].unique().tolist())
filtro_mes = st.sidebar.selectbox('Filtrar por Mes', ['Todos'] + df['MES'].unique().tolist())
filtro_clasificacion = st.sidebar.selectbox('Filtrar por Triague', ['Todos'] + df['CLASIFICACION_TRIAGE'].unique().tolist())

st.sidebar.info('Created by Victor - Diana')

# Aplicar filtros a los datos
filtro_anos = df['AÑO'].unique().tolist()

if filtro_centro == 'Todos':
    mask_centro = df['CENTRO_ATENCION'].notna()
else:
    mask_centro = df['CENTRO_ATENCION'] == filtro_centro

if filtro_mes == 'Todos':
    mask_mes = df['MES'].notna()
else:
    mask_mes = df['MES'] == filtro_mes

if filtro_clasificacion == 'Todos':
    mask_clasificacion = df['CLASIFICACION_TRIAGE'].notna()
else:
    mask_clasificacion = df['CLASIFICACION_TRIAGE'] == filtro_clasificacion

# Crear gráficas de barras

with st.container():
  st.subheader("Bienvenidos  :wave:")
  st.title("📊 Tiempos Atencion de Urgencias")
  st.write(" Esta es una pagina para mostrar los resultados")


  # Combinar las máscaras de filtro
  mask = mask_centro & mask_mes & mask_clasificacion
  numero_resultados = df[mask].shape[0]
  st.markdown(f'*Resultados Disponibles:{numero_resultados}*')

## KPIs

  @st.cache_data
  def calculate_kpis(df: pd.DataFrame) -> List[float]:
        total_minutos1 =(df[mask]['Tiempo_Minutos_Total'].sum())
        Total_minutos = f"{total_minutos1:.2f}M"
        total_pacientes = df[mask]['PACIENTE_#_DOCUMENTO'].nunique()
        Promedio_minutos = f"{total_minutos1 / total_pacientes:.2f}K"
        return [Total_minutos, total_pacientes, Promedio_minutos, total_pacientes]
  

  def display_kpi_metrics(kpis: List[float], kpi_names: List[str]):
        st.header("KPI Metrics")
        for i, (col, (kpi_name, kpi_value)) in enumerate(zip(st.columns(4), zip(kpi_names, kpis))):
            col.metric(label=kpi_name, value=kpi_value)


  kpis = calculate_kpis(df)
  kpi_names = ["Total Minutos", "Cantidad Pacientes", "Promedio Minutos", "Cantidad Pacientes"]
  display_kpi_metrics(kpis, kpi_names)

  st.write("---")
  #st.subheader("Top 10 Atenciones")
  #st.write(df[mask].head(10))

with st.container():
    st.write("---")
    st.header("Tiempos de Atencion de pacientes  Historica en la Sala de Urgencias")
    left_column , right_column = st.columns(2)

    with left_column:
        st.header("DIA DE LA SEMANA")
        st.write("Esta imagen muestra Por dias de la semana del Dia")
    
    # Asegúrate de que la columna 'ds' sea de tipo datetime
        df[mask]['FECHA_LLEGADA'] = pd.to_datetime(df[mask]['FECHA_LLEGADA'])

    #   Ahora puedes acceder al día de la semana usando el atributo 'dayofweek'
        df[mask]['day_of_week'] = df[mask]['FECHA_LLEGADA'].dt.dayofweek

        promedio = df[mask]['Tiempo_Minutos_Total'].median()
        df[mask].loc[df[mask]['Tiempo_Minutos_Total'] > 420, 'Tiempo_Minutos_Total'] = promedio
        df[mask].loc[df[mask]['Tiempo_Minutos_Total'] < 0, 'Tiempo_Minutos_Total'] = promedio

    # Calcula el promedio de las predicciones para cada día de la semana
        average_predicted_minutes = df[mask].groupby('day_of_week')['Tiempo_Minutos_Total'].mean()

    # Establece los índices explícitamente
        average_predicted_minutes.index = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']

    # Trazar el gráfico de barras
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=average_predicted_minutes.index, y=average_predicted_minutes.values, palette='viridis', ax=ax)
        ax.set_xlabel('Día de la Semana')
        ax.set_ylabel('Promedio del Tiempo (minutos)')
        ax.set_title('Promedio del Tiempo por Día de la Semana')

    # Añade etiquetas a las barras
        for i, bar in enumerate(ax.patches):
            yval = bar.get_height()
            xval = bar.get_x() + bar.get_width() / 2
            ax.text(xval, yval, f"{round(yval, 2)}", ha='center', va='bottom')

    # Muestra la figura en Streamlit
        st.pyplot(fig)

    with right_column:
        st.header("HORAS DEL DIA")
        st.write("Esta imagen muestra Por Horas del Dia")
    
    #   Ahora puedes acceder al día de la semana usando el atributo 'dayofweek'
        df[mask]['Hora_del_dia'] = df[mask]['FECHA_LLEGADA'].dt.hour

        promedio = df[mask]['Tiempo_Minutos_Total'].median()
        df[mask].loc[df[mask]['Tiempo_Minutos_Total'] > 420, 'Tiempo_Minutos_Total'] = promedio
        df[mask].loc[df[mask]['Tiempo_Minutos_Total'] < 0, 'Tiempo_Minutos_Total'] = promedio

    # Calcula el promedio de las predicciones para cada día de la semana
        average_predicted_minutes = df[mask].groupby('Hora_del_dia')['Tiempo_Minutos_Total'].mean()

    # Trazar el gráfico de barras
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=average_predicted_minutes.index, y=average_predicted_minutes.values, palette='viridis', ax=ax)
        ax.set_xlabel('Hora del Dia')
        ax.set_ylabel('Promedio del Tiempo (minutos)')
        ax.set_title('Promedio del Tiempo por Día de la Semana')

    # Añade etiquetas a las barras
        for i, bar in enumerate(ax.patches):
            yval = bar.get_height()
            xval = bar.get_x() + bar.get_width() / 2
            ax.text(xval, yval, f"{round(yval, 2)}", ha='center', va='bottom')

    # Muestra la figura en Streamlit
        st.pyplot(fig)        

# %%
