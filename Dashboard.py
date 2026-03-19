import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime, timedelta
import base64
from io import BytesIO
import calendar

# Configuration de la page
st.set_page_config(
    page_title="🪐 Mars Data Dashboard",
    page_icon="🔴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé avec thème spatial
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF4500;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #000000, #1a1a1a, #000000);
        border-radius: 10px;
        margin-bottom: 2rem;
        border: 2px solid #FF4500;
        text-shadow: 0 0 10px #FF4500;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a1a, #000000);
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #FF4500;
        text-align: center;
        box-shadow: 0 0 20px rgba(255, 69, 0, 0.3);
    }
    .metric-label {
        color: #FFA07A;
        font-size: 1rem;
        font-weight: 600;
    }
    .metric-value {
        color: #FF4500;
        font-size: 2rem;
        font-weight: bold;
        text-shadow: 0 0 10px rgba(255, 69, 0, 0.5);
    }
    .info-box {
        background: linear-gradient(135deg, #1a1a1a, #000000);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #FF4500;
        margin: 1rem 0;
        box-shadow: 0 0 15px rgba(255, 69, 0, 0.2);
    }
    .event-marker {
        background: linear-gradient(135deg, #2a2a2a, #1a1a1a);
        padding: 0.5rem;
        border-radius: 5px;
        border: 1px solid #FF4500;
        margin: 0.2rem 0;
        transition: all 0.3s ease;
    }
    .event-marker:hover {
        transform: translateX(5px);
        box-shadow: 0 0 15px rgba(255, 69, 0, 0.5);
        border-color: #FFA07A;
    }
    .activity-low {
        color: #87CEEB;
        font-weight: bold;
    }
    .activity-moderate {
        color: #FFD700;
        font-weight: bold;
    }
    .activity-high {
        color: #FF4500;
        font-weight: bold;
    }
    .activity-extreme {
        color: #FF0000;
        font-weight: bold;
        text-shadow: 0 0 10px #FF0000;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #000000;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1a1a1a;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 20px;
        color: #FF4500;
        border: 1px solid #FF4500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FF4500;
        color: #000000;
    }
    .mission-badge {
        background: linear-gradient(135deg, #FF4500, #FF6347);
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 5px;
        font-weight: bold;
        display: inline-block;
        margin: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

class MarsDataAnalyzer:
    def __init__(self, data_type):
        self.data_type = data_type
        self.colors = ['#FF4500', '#C1440E', '#8B4513', '#D2691E', '#CD5C5C', 
                      '#A52A2A', '#800000', '#B22222', '#DC143C', '#FF6347']
        
        self.start_year = 1965
        self.end_year = 2025
        
        self.config = self._get_mars_config()
        
    def _get_mars_config(self):
        """Retourne la configuration spécifique pour chaque type de données martiennes"""
        configs = {
            "temperature": {
                "base_value": -63,
                "cycle_years": 1.88,
                "amplitude": 70,
                "trend": "stable",
                "unit": "°C",
                "description": "🌡️ Température moyenne martienne",
                "icon": "🌡️",
                "color": "#FF4500",
                "range": [-143, 35]
            },
            "atmospheric_pressure": {
                "base_value": 6.1,
                "cycle_years": 1.88,
                "amplitude": 0.5,
                "trend": "variable",
                "unit": "mbar",
                "description": "💨 Pression atmosphérique",
                "icon": "💨",
                "color": "#C1440E",
                "range": [4.0, 8.7]
            },
            "dust_storms": {
                "base_value": 50,
                "cycle_years": 3.0,
                "amplitude": 40,
                "trend": "cyclique",
                "unit": "Index",
                "description": "🌪️ Activité des tempêtes",
                "icon": "🌪️",
                "color": "#8B4513",
                "range": [0, 100]
            },
            "co2_ice": {
                "base_value": 30,
                "cycle_years": 1.88,
                "amplitude": 25,
                "trend": "saisonnier",
                "unit": "%",
                "description": "❄️ Glace carbonique aux pôles",
                "icon": "❄️",
                "color": "#D2691E",
                "range": [0, 100]
            },
            "water_ice": {
                "base_value": 15,
                "cycle_years": 1.88,
                "amplitude": 10,
                "trend": "stable",
                "unit": "%",
                "description": "💧 Glace d'eau souterraine",
                "icon": "💧",
                "color": "#CD5C5C",
                "range": [0, 30]
            },
            "solar_radiation": {
                "base_value": 600,
                "cycle_years": 11.0,
                "amplitude": 100,
                "trend": "cyclique",
                "unit": "W/m²",
                "description": "☢️ Radiation solaire",
                "icon": "☢️",
                "color": "#A52A2A",
                "range": [400, 800]
            },
            "magnetic_field": {
                "base_value": 0,
                "cycle_years": 1.88,
                "amplitude": 5,
                "trend": "localisé",
                "unit": "nT",
                "description": "🧲 Champ magnétique résiduel",
                "icon": "🧲",
                "color": "#800000",
                "range": [0, 500]
            },
            "seismic_activity": {
                "base_value": 1,
                "cycle_years": 1.88,
                "amplitude": 2,
                "trend": "sporadique",
                "unit": "Magnitude",
                "description": "📊 Activité sismique (marsquakes)",
                "icon": "📊",
                "color": "#B22222",
                "range": [0, 5]
            },
            "orbital_distance": {
                "base_value": 1.52,
                "cycle_years": 2.14,
                "amplitude": 0.14,
                "trend": "périodique",
                "unit": "UA",
                "description": "🛸 Distance au Soleil",
                "icon": "🛸",
                "color": "#DC143C",
                "range": [1.38, 1.67]
            }
        }
        return configs.get(self.data_type, configs["temperature"])
    
    def generate_mars_data(self):
        """Génère des données martiennes simulées"""
        dates = pd.date_range(start=f'{self.start_year}-01-01', 
                             end=f'{self.end_year}-12-31', freq='Y')
        
        data = {'Earth_Year': [date.year for date in dates]}
        data['Mars_Year'] = self._earth_to_mars_years(dates)
        
        # Données principales
        data['Base_Value'] = self._simulate_mars_cycle(dates)
        data['Seasonal_Min'] = self._simulate_seasonal_minima(dates)
        data['Seasonal_Max'] = self._simulate_seasonal_maxima(dates)
        data['Orbital_Phase'] = self._simulate_orbital_phase(dates)
        data['Climate_Trend'] = self._simulate_climate_trend(dates)
        data['Dust_Events'] = self._simulate_dust_events(dates)
        data['Solar_Influence'] = self._simulate_solar_influence(dates)
        data['Smoothed_Value'] = self._simulate_smoothed_data(dates)
        data['Seasonal_Variation'] = self._simulate_seasonal_variation(dates)
        data['Annual_Variation'] = self._simulate_annual_variation(dates)
        data['Mars_Index'] = self._simulate_mars_index(dates)
        data['Activity_Level'] = self._simulate_activity_level(dates)
        data['Future_Prediction'] = self._simulate_future_prediction(dates)
        data['Solar_Conjunction'] = self._simulate_solar_conjunction(dates)
        data['Dust_Opacity'] = self._simulate_dust_opacity(dates)
        
        df = pd.DataFrame(data)
        self._add_mars_events(df)
        
        return df
    
    def _earth_to_mars_years(self, dates):
        mars_years = []
        mars_year_start = 1965
        mars_year_duration = 1.88
        
        for date in dates:
            earth_year = date.year
            mars_year = 1 + (earth_year - mars_year_start) / mars_year_duration
            mars_years.append(mars_year)
        return mars_years
    
    def _simulate_mars_cycle(self, dates):
        base_value = self.config["base_value"]
        cycle_years = max(self.config["cycle_years"], 0.01)  # Éviter division par zéro
        amplitude = self.config["amplitude"]
        
        values = []
        for i, date in enumerate(dates):
            earth_year = date.year
            mars_phase = (earth_year - self.start_year) % cycle_years
            seasonal_cycle = np.sin(2 * np.pi * mars_phase / cycle_years)
            
            orbital_phase = (earth_year - self.start_year) % 2.14
            orbital_cycle = np.cos(2 * np.pi * orbital_phase / 2.14)
            
            if self.config["trend"] == "saisonnier":
                value = base_value + amplitude * seasonal_cycle
            elif self.config["trend"] == "périodique":
                value = base_value + amplitude * 0.7 * orbital_cycle
            elif self.config["trend"] == "cyclique":
                value = base_value + amplitude * (0.6 * seasonal_cycle + 0.4 * orbital_cycle)
            else:
                value = base_value + amplitude * 0.3 * seasonal_cycle
            
            noise = np.random.normal(0, amplitude * 0.08)
            values.append(value + noise)
        return values
    
    def _simulate_seasonal_minima(self, dates):
        minima = []
        for i, date in enumerate(dates):
            earth_year = date.year
            mars_phase = (earth_year - self.start_year) % 1.88
            
            if 1.4 <= mars_phase <= 1.88 or 0 <= mars_phase <= 0.48:
                min_factor = 0.3
            else:
                min_factor = 0.8
            minima.append(min_factor)
        return minima
    
    def _simulate_seasonal_maxima(self, dates):
        maxima = []
        for i, date in enumerate(dates):
            earth_year = date.year
            mars_phase = (earth_year - self.start_year) % 1.88
            
            if 0.9 <= mars_phase <= 1.4:
                max_factor = 1.0
            else:
                max_factor = 0.6
            maxima.append(max_factor)
        return maxima
    
    def _simulate_orbital_phase(self, dates):
        phases = []
        for date in dates:
            earth_year = date.year
            phase = (earth_year - self.start_year) % 2.14 / 2.14
            phases.append(phase)
        return phases
    
    def _simulate_climate_trend(self, dates):
        trends = []
        for i, date in enumerate(dates):
            earth_year = date.year
            
            if earth_year < 1975:
                trend = 0.95
            elif 1975 <= earth_year < 1990:
                trend = 1.0 + 0.001 * (earth_year - 1975)
            elif 1990 <= earth_year < 2000:
                trend = 1.02 + 0.002 * (earth_year - 1990)
            elif 2000 <= earth_year < 2010:
                trend = 1.04 + 0.003 * (earth_year - 2000)
            elif 2010 <= earth_year < 2020:
                trend = 1.07 + 0.004 * (earth_year - 2010)
            else:
                trend = 1.11 + 0.005 * (earth_year - 2020)
            
            trends.append(trend)
        return trends
    
    def _simulate_dust_events(self, dates):
        dust_events = []
        for date in dates:
            earth_year = date.year
            mars_year = 1 + (earth_year - 1965) / 1.88
            
            dust_cycle = mars_year % 3.0
            if 2.8 <= dust_cycle <= 3.0 or 0 <= dust_cycle <= 0.2:
                dust_level = 1.8
            elif 2.6 <= dust_cycle <= 2.8 or 0.2 <= dust_cycle <= 0.4:
                dust_level = 1.4
            else:
                dust_level = 1.0
            
            dust_events.append(dust_level)
        return dust_events
    
    def _simulate_dust_opacity(self, dates):
        opacities = []
        dust_events = self._simulate_dust_events(dates)
        for dust in dust_events:
            opacity = 0.1 + 0.3 * (dust - 1)
            opacities.append(opacity)
        return opacities
    
    def _simulate_solar_influence(self, dates):
        solar_effects = []
        for date in dates:
            earth_year = date.year
            solar_phase = (earth_year - self.start_year) % 11.0
            solar_effect = 1.0 + 0.1 * np.sin(2 * np.pi * solar_phase / 11.0)
            solar_effects.append(solar_effect)
        return solar_effects
    
    def _simulate_solar_conjunction(self, dates):
        conjunctions = []
        for date in dates:
            earth_year = date.year
            # Conjonction solaire environ tous les 26 mois
            conjunction_phase = (earth_year - self.start_year) % 2.17
            is_conjunction = 0.9 <= conjunction_phase <= 1.1
            conjunctions.append(1.0 if is_conjunction else 0.0)
        return conjunctions
    
    def _simulate_smoothed_data(self, dates):
        base_cycle = self._simulate_mars_cycle(dates)
        smoothed = []
        window_size = int(2 * 1.88)
        
        for i in range(len(base_cycle)):
            start_idx = max(0, i - window_size//2)
            end_idx = min(len(base_cycle), i + window_size//2 + 1)
            window = base_cycle[start_idx:end_idx]
            smoothed.append(np.mean(window))
        return smoothed
    
    def _simulate_seasonal_variation(self, dates):
        variations = []
        for date in dates:
            earth_year = date.year
            mars_season = (earth_year - self.start_year) % 1.88 / 1.88
            seasonal_variation = 0.2 * np.sin(2 * np.pi * mars_season)
            variations.append(1 + seasonal_variation)
        return variations
    
    def _simulate_annual_variation(self, dates):
        variations = []
        for i, date in enumerate(dates):
            earth_year = date.year
            annual_variation = 0.05 * np.sin(2 * np.pi * (earth_year - self.start_year) / 1.0)
            variations.append(1 + annual_variation)
        return variations
    
    def _simulate_mars_index(self, dates):
        indices = []
        base_cycle = self._simulate_mars_cycle(dates)
        climate_trend = self._simulate_climate_trend(dates)
        dust_events = self._simulate_dust_events(dates)
        
        for i in range(len(dates)):
            index = (base_cycle[i] * 0.5 + 
                    climate_trend[i] * self.config["base_value"] * 0.3 +
                    dust_events[i] * 10 * 0.2)
            indices.append(index)
        return indices
    
    def _simulate_activity_level(self, dates):
        activity_levels = []
        dust_events = self._simulate_dust_events(dates)
        
        for dust_level in dust_events:
            activity = min(100, (dust_level - 1) * 125)
            activity_levels.append(activity)
        return activity_levels
    
    def _simulate_future_prediction(self, dates):
        predictions = []
        base_cycle = self._simulate_mars_cycle(dates)
        climate_trend = self._simulate_climate_trend(dates)
        
        for i, date in enumerate(dates):
            earth_year = date.year
            current_value = base_cycle[i]
            trend_factor = climate_trend[i]
            
            if earth_year > 2020:
                years_since_2020 = earth_year - 2020
                uncertainty = 0.04 * years_since_2020
                prediction = current_value * trend_factor * (1 + np.random.normal(0, uncertainty))
            else:
                prediction = current_value
            predictions.append(prediction)
        return predictions
    
    def _add_mars_events(self, df):
        """Ajoute des événements martiens historiques"""
        events = []
        for i, row in df.iterrows():
            earth_year = row['Earth_Year']
            
            # Missions historiques
            mission_events = {
                1965: ("Mariner 4 - Premières images rapprochées", "flyby", "historique"),
                1971: ("Mariner 9 - Première cartographie", "orbiter", "majeur"),
                1976: ("Vikings 1 & 2 - Premiers atterrissages", "lander", "historique"),
                1997: ("Mars Pathfinder - Retour sur Mars", "lander", "majeur"),
                2004: ("Spirit & Opportunity - Exploration longue durée", "rover", "historique"),
                2006: ("MRO - Cartographie haute résolution", "orbiter", "majeur"),
                2012: ("Curiosity - Laboratoire mobile", "rover", "historique"),
                2018: ("InSight - Étude sismique", "lander", "majeur"),
                2021: ("Perseverance & Ingenuity - Premier vol", "rover", "historique")
            }
            
            if earth_year in mission_events:
                event_name, event_type, severity = mission_events[earth_year]
                events.append({
                    "year": earth_year,
                    "event": event_name,
                    "type": event_type,
                    "severity": severity,
                    "mars_year": row['Mars_Year']
                })
                df.loc[i, 'Activity_Level'] = min(100, df.loc[i, 'Activity_Level'] + 20)
            
            # Grandes tempêtes
            storm_years = [1971, 1977, 1982, 1994, 2001, 2007, 2018]
            if earth_year in storm_years:
                events.append({
                    "year": earth_year,
                    "event": "🌪️ Tempête globale de poussière",
                    "type": "storm",
                    "severity": "extreme",
                    "mars_year": row['Mars_Year']
                })
                df.loc[i, 'Dust_Events'] *= 1.8
                df.loc[i, 'Activity_Level'] = min(100, df.loc[i, 'Activity_Level'] * 1.5)
        
        self.events = events

def create_plotly_visualizations(df, analyzer):
    """Crée des visualisations Plotly interactives"""
    
    fig_main = make_subplots(
        rows=3, cols=2,
        subplot_titles=('🌡️ Cycle Martien Principal', '🌪️ Activité et Tempêtes',
                       '📈 Tendances Climatiques', '🔄 Phases Orbitales',
                       '📊 Données Brutes vs Lissées', '🔮 Projections Futures'),
        specs=[[{"secondary_y": True}, {"secondary_y": True}],
               [{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Cycle principal
    fig_main.add_trace(
        go.Scatter(x=df['Earth_Year'], y=df['Base_Value'],
                  mode='lines', name='Valeur observée',
                  line=dict(color=analyzer.config['color'], width=2),
                  hovertemplate='Année: %{x}<br>Valeur: %{y:.2f} ' + analyzer.config['unit']),
        row=1, col=1
    )
    
    # Activité et tempêtes
    fig_main.add_trace(
        go.Scatter(x=df['Earth_Year'], y=df['Activity_Level'],
                  mode='lines', name="Niveau d'activité",
                  line=dict(color='#FF4500', width=2),
                  fill='tozeroy'),
        row=1, col=2
    )
    fig_main.add_trace(
        go.Scatter(x=df['Earth_Year'], y=df['Dust_Events'] * 50,
                  mode='lines', name='Tempêtes de poussière',
                  line=dict(color='#8B4513', width=2, dash='dot')),
        row=1, col=2
    )
    
    # Tendances climatiques
    fig_main.add_trace(
        go.Scatter(x=df['Earth_Year'], y=df['Climate_Trend'],
                  mode='lines', name='Tendance climatique',
                  line=dict(color='#FFD700', width=2)),
        row=2, col=1
    )
    fig_main.add_trace(
        go.Scatter(x=df['Earth_Year'], y=df['Solar_Influence'],
                  mode='lines', name='Influence solaire',
                  line=dict(color='#00BFFF', width=2)),
        row=2, col=1
    )
    
    # Phases orbitales
    fig_main.add_trace(
        go.Scatter(x=df['Earth_Year'], y=df['Orbital_Phase'],
                  mode='lines', name='Phase orbitale',
                  line=dict(color='#FF69B4', width=2)),
        row=2, col=2
    )
    
    # Conjonctions solaires
    conjunction_years = df[df['Solar_Conjunction'] == 1]['Earth_Year']
    if len(conjunction_years) > 0:
        fig_main.add_trace(
            go.Scatter(x=conjunction_years, 
                      y=[0.5] * len(conjunction_years),
                      mode='markers', name='Conjonction solaire',
                      marker=dict(symbol='star', size=10, color='yellow')),
            row=2, col=2
        )
    
    # Données brutes vs lissées
    fig_main.add_trace(
        go.Scatter(x=df['Earth_Year'], y=df['Base_Value'],
                  mode='lines', name='Données brutes',
                  line=dict(color=analyzer.config['color'], width=1, dash='dot')),
        row=3, col=1
    )
    fig_main.add_trace(
        go.Scatter(x=df['Earth_Year'], y=df['Smoothed_Value'],
                  mode='lines', name='Données lissées',
                  line=dict(color='#00FF7F', width=3)),
        row=3, col=1
    )
    
    # Projections futures
    fig_main.add_trace(
        go.Scatter(x=df['Earth_Year'][df['Earth_Year'] <= 2020], 
                  y=df['Base_Value'][df['Earth_Year'] <= 2020],
                  mode='lines', name='Historique',
                  line=dict(color=analyzer.config['color'], width=2)),
        row=3, col=2
    )
    fig_main.add_trace(
        go.Scatter(x=df['Earth_Year'][df['Earth_Year'] >= 2020], 
                  y=df['Future_Prediction'][df['Earth_Year'] >= 2020],
                  mode='lines', name='Projections',
                  line=dict(color='#00FFFF', width=2, dash='dash')),
        row=3, col=2
    )
    
    # Mise en forme
    fig_main.update_layout(
        height=1200,
        showlegend=True,
        template='plotly_dark',
        title_text=f"🪐 Analyse Interactive des Données Martiennes - {analyzer.config['description']}",
        title_font_size=20,
        title_font_color='#FF4500',
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Mise à jour des axes
    for i in range(1, 4):
        for j in range(1, 3):
            fig_main.update_xaxes(title_text="Année Terrestre", row=i, col=j, gridcolor='#333333')
            fig_main.update_yaxes(gridcolor='#333333', row=i, col=j)
    
    fig_main.update_yaxes(title_text=analyzer.config['unit'], row=1, col=1)
    fig_main.update_yaxes(title_text="Niveau (0-100)", row=1, col=2, range=[0, 100])
    fig_main.update_yaxes(title_text="Facteur multiplicatif", row=2, col=1)
    fig_main.update_yaxes(title_text="Phase (0-1)", row=2, col=2)
    fig_main.update_yaxes(title_text=analyzer.config['unit'], row=3, col=1)
    fig_main.update_yaxes(title_text=analyzer.config['unit'], row=3, col=2)
    
    return fig_main

def create_mars_globe_visualization(df, analyzer):
    """Crée une visualisation 3D de Mars"""
    # Points de surface simulés
    np.random.seed(42)
    n_points = 1000
    
    lats = np.random.uniform(-90, 90, n_points)
    lons = np.random.uniform(-180, 180, n_points)
    
    # Valeur basée sur le type de données
    if analyzer.data_type == "temperature":
        values = -63 + 70 * np.sin(np.radians(lats)) + np.random.normal(0, 10, n_points)
        colorscale = 'Hot'
    elif analyzer.data_type == "water_ice":
        values = 15 * np.exp(-np.abs(lats)/45) + np.random.normal(0, 2, n_points)
        colorscale = 'Blues'
    elif analyzer.data_type == "dust_storms":
        values = 50 * np.random.beta(2, 5, n_points)
        colorscale = 'Earth'
    else:
        values = 100 * np.random.random(n_points)
        colorscale = 'Viridis'
    
    fig_globe = go.Figure(data=[go.Scattergeo(
        lon=lons,
        lat=lats,
        mode='markers',
        marker=dict(
            size=5,
            color=values,
            colorscale=colorscale,
            cmin=values.min(),
            cmax=values.max(),
            colorbar=dict(title=analyzer.config['unit']),
            line=dict(width=0)
        ),
        text=[f"Lat: {lat:.1f}°<br>Lon: {lon:.1f}°<br>Valeur: {val:.1f}" 
              for lat, lon, val in zip(lats, lons, values)],
        hoverinfo='text'
    )])
    
    fig_globe.update_layout(
        title=f"Carte de Surface - {analyzer.config['description']}",
        template='plotly_dark',
        geo=dict(
            projection_type='orthographic',
            projection_rotation=dict(lon=0, lat=0),
            showland=True,
            landcolor='#8B4513',
            showocean=True,
            oceancolor='#1a1a1a',
            showcountries=False,
            coastlinecolor='#FF4500',
            coastlinewidth=0.5
        ),
        height=600
    )
    
    return fig_globe

def create_mission_timeline(events):
    """Crée une timeline des missions martiennes"""
    if not events:
        return None
    
    df_events = pd.DataFrame(events)
    
    fig_timeline = go.Figure()
    
    # Couleurs par type de mission
    color_map = {
        'flyby': '#87CEEB',
        'orbiter': '#FFD700',
        'lander': '#FFA07A',
        'rover': '#FF4500',
        'storm': '#8B4513'
    }
    
    for event_type in df_events['type'].unique():
        df_type = df_events[df_events['type'] == event_type]
        
        fig_timeline.add_trace(go.Scatter(
            x=df_type['year'],
            y=[1] * len(df_type),
            mode='markers+text',
            name=event_type.capitalize(),
            marker=dict(
                size=15,
                color=color_map.get(event_type, '#808080'),
                symbol='diamond' if event_type == 'storm' else 'circle',
                line=dict(color='white', width=1)
            ),
            text=df_type['event'],
            textposition="top center",
            hoverinfo='text',
            showlegend=True
        ))
    
    fig_timeline.update_layout(
        title="🚀 Chronologie des Missions et Événements Martiens",
        template='plotly_dark',
        height=300,
        xaxis=dict(
            title="Année Terrestre",
            gridcolor='#333333',
            dtick=5
        ),
        yaxis=dict(
            showticklabels=False,
            gridcolor='#333333'
        ),
        hovermode='x',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig_timeline

def create_seasonal_pattern_mars(df, analyzer):
    """Crée une visualisation des patterns saisonniers martiens"""
    # Simuler des données mensuelles martiennes
    mars_months = list(range(1, 13))
    month_names = [f"M{m}" for m in mars_months]
    
    if analyzer.data_type == "temperature":
        seasonal_data = -63 + 35 * np.sin(2 * np.pi * (np.array(mars_months) - 1) / 12)
    elif analyzer.data_type == "co2_ice":
        seasonal_data = 30 + 25 * np.cos(2 * np.pi * (np.array(mars_months) - 3) / 12)
    elif analyzer.data_type == "dust_storms":
        seasonal_data = 50 + 40 * np.abs(np.sin(2 * np.pi * (np.array(mars_months) - 6) / 12))
    else:
        seasonal_data = 100 + 20 * np.sin(2 * np.pi * (np.array(mars_months) - 6) / 12)
    
    fig_seasonal = go.Figure()
    
    fig_seasonal.add_trace(go.Scatter(
        x=month_names,
        y=seasonal_data,
        mode='lines+markers',
        name='Pattern saisonnier',
        line=dict(color=analyzer.config['color'], width=3),
        marker=dict(size=8, color=analyzer.config['color']),
        fill='tozeroy',
        fillcolor=f'rgba{tuple(int(analyzer.config["color"].lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + (0.2,)}'
    ))
    
    fig_seasonal.update_layout(
        title="Pattern Saisonnier Martien",
        template='plotly_dark',
        xaxis_title="Mois Martien",
        yaxis_title=analyzer.config['unit'],
        height=400
    )
    
    return fig_seasonal

def get_activity_class(activity_level):
    """Retourne la classe CSS pour le niveau d'activité"""
    if activity_level < 25:
        return "activity-low"
    elif activity_level < 50:
        return "activity-moderate"
    elif activity_level < 75:
        return "activity-high"
    else:
        return "activity-extreme"

def main():
    # En-tête
    st.markdown('<h1 class="main-header">🪐 Mars Data Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/0/02/OSIRIS_Mars_true_color.jpg", 
                 use_column_width=True)
        
        st.markdown("## 🎯 Configuration")
        
        # Types de données
        mars_data_types = {
            "temperature": "🌡️ Température",
            "atmospheric_pressure": "💨 Pression atmosphérique",
            "dust_storms": "🌪️ Tempêtes de poussière",
            "co2_ice": "❄️ Glace carbonique",
            "water_ice": "💧 Glace d'eau",
            "solar_radiation": "☢️ Radiation solaire",
            "magnetic_field": "🧲 Champ magnétique",
            "seismic_activity": "📊 Activité sismique",
            "orbital_distance": "🛸 Distance orbitale"
        }
        
        selected_type = st.selectbox(
            "Type de données martiennes",
            options=list(mars_data_types.keys()),
            format_func=lambda x: mars_data_types[x]
        )
        
        # Période d'analyse
        st.markdown("### 📅 Période d'analyse")
        col1, col2 = st.columns(2)
        with col1:
            start_year = st.number_input("Début", min_value=1960, max_value=2000, value=1965)
        with col2:
            end_year = st.number_input("Fin", min_value=1966, max_value=2030, value=2025)
        
        # Options d'affichage
        st.markdown("### 🎨 Options d'affichage")
        show_missions = st.checkbox("Afficher les missions", value=True)
        show_storms = st.checkbox("Afficher les tempêtes", value=True)
        show_conjunctions = st.checkbox("Afficher les conjonctions", value=True)
        
        # Mode visualisation
        viz_mode = st.radio(
            "Mode de visualisation",
            ["Standard", "3D Globe", "Comparaison"]
        )
        
        # Bouton de génération
        if st.button("🪐 Générer l'analyse", use_container_width=True):
            st.session_state['generate'] = True
        
        st.markdown("---")
        st.markdown("### 📊 Métriques en direct")
        
        st.metric("Données chargées", "✓", delta=None)
        st.metric("Précision des mesures", "±2.5%", delta="0.5%")
        st.metric("Dernière mise à jour", datetime.now().strftime("%H:%M:%S"), delta=None)
    
    # Initialisation de l'analyseur
    global analyzer
    analyzer = MarsDataAnalyzer(selected_type)
    analyzer.start_year = start_year
    analyzer.end_year = end_year
    
    # Génération des données
    if 'generate' in st.session_state or 'df' not in st.session_state:
        with st.spinner("🪐 Génération des données martiennes en cours..."):
            df = analyzer.generate_mars_data()
            st.session_state['df'] = df
            st.session_state['analyzer'] = analyzer
    else:
        df = st.session_state['df']
        analyzer = st.session_state['analyzer']
    
    # Métriques principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_mars_year = df['Mars_Year'].iloc[-1]
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{analyzer.config['icon']} Année martienne</div>
            <div class="metric-value">{current_mars_year:.1f}</div>
            <div style="color: #FFA07A;">MY {int(current_mars_year)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        current_value = df[df['Earth_Year'] == 2025]['Base_Value'].values[0] if 2025 in df['Earth_Year'].values else df['Base_Value'].iloc[-1]
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">📈 Valeur actuelle</div>
            <div class="metric-value">{current_value:.1f}</div>
            <div style="color: #FFA07A;">{analyzer.config['unit']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        activity_current = df['Activity_Level'].iloc[-1]
        activity_class = get_activity_class(activity_current)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🌪️ Activité actuelle</div>
            <div class="metric-value {activity_class}">{activity_current:.0f}</div>
            <div style="color: #FFA07A;">/100</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        orbital_phase = df['Orbital_Phase'].iloc[-1]
        phase_desc = "Périhélie" if orbital_phase < 0.25 else "Aphélie" if orbital_phase > 0.75 else "Transition"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🛸 Phase orbitale</div>
            <div class="metric-value">{orbital_phase:.2f}</div>
            <div style="color: #FFA07A;">{phase_desc}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Tabs pour différentes analyses
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Analyse Principale", "🔄 Cycles", "🚀 Missions", 
        "📊 Statistiques", "🔮 Projections", "🪐 Surface"
    ])
    
    with tab1:
        st.markdown("## Visualisation Interactive")
        
        # Sélecteur de période
        col1, col2 = st.columns([3, 1])
        with col1:
            year_range = st.slider(
                "Période d'affichage",
                min_value=int(df['Earth_Year'].min()),
                max_value=int(df['Earth_Year'].max()),
                value=(2000, 2025)
            )
        
        with col2:
            chart_style = st.selectbox(
                "Style",
                ["Lignes", "Points", "Barres", "Area"]
            )
        
        # Filtrer les données
        df_filtered = df[(df['Earth_Year'] >= year_range[0]) & (df['Earth_Year'] <= year_range[1])]
        
        # Graphique principal
        if viz_mode == "Standard":
            fig_main = create_plotly_visualizations(df_filtered, analyzer)
            st.plotly_chart(fig_main, use_container_width=True)
        elif viz_mode == "3D Globe":
            fig_globe = create_mars_globe_visualization(df_filtered, analyzer)
            st.plotly_chart(fig_globe, use_container_width=True)
        else:
            # Mode comparaison
            col1, col2 = st.columns(2)
            with col1:
                fig1 = create_plotly_visualizations(df_filtered, analyzer)
                st.plotly_chart(fig1, use_container_width=True)
            with col2:
                fig2 = create_mars_globe_visualization(df_filtered, analyzer)
                st.plotly_chart(fig2, use_container_width=True)
        
        # Section des informations
        with st.expander("ℹ️ À propos des données martiennes"):
            st.markdown(f"""
            <div class="info-box">
                <h4>{analyzer.config['description']}</h4>
                <p><strong>Unité:</strong> {analyzer.config['unit']}</p>
                <p><strong>Plage typique:</strong> {analyzer.config['range'][0]} - {analyzer.config['range'][1]} {analyzer.config['unit']}</p>
                <p><strong>Cycle principal:</strong> {analyzer.config['cycle_years']} années terrestres</p>
                <p><strong>Année martienne:</strong> 1.88 années terrestres (687 jours)</p>
                <p><strong>Période d'analyse:</strong> {start_year} - {end_year}</p>
                <p><strong>Couverture:</strong> ~{(end_year-start_year)/1.88:.1f} années martiennes</p>
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("## Cycles Naturels Martiens")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Pattern saisonnier
            fig_seasonal = create_seasonal_pattern_mars(df, analyzer)
            st.plotly_chart(fig_seasonal, use_container_width=True)
        
        with col2:
            # Cycle solaire
            st.markdown("### Cycle Solaire")
            fig_solar = go.Figure()
            
            solar_years = df['Earth_Year']
            solar_values = df['Solar_Influence']
            
            fig_solar.add_trace(go.Scatter(
                x=solar_years,
                y=solar_values,
                mode='lines',
                name='Influence solaire',
                line=dict(color='#FFD700', width=2)
            ))
            
            # Pics solaires
            solar_max_years = solar_years[solar_values > 1.08]
            if len(solar_max_years) > 0:
                fig_solar.add_trace(go.Scatter(
                    x=solar_max_years,
                    y=[1.1] * len(solar_max_years),
                    mode='markers',
                    name='Maximum solaire',
                    marker=dict(symbol='star', size=8, color='red')
                ))
            
            fig_solar.update_layout(
                template='plotly_dark',
                xaxis_title="Année Terrestre",
                yaxis_title="Facteur d'influence",
                height=400
            )
            st.plotly_chart(fig_solar, use_container_width=True)
        
        # Matrice de corrélation
        st.markdown("### Corrélations entre Variables Martiennes")
        
        corr_cols = ['Base_Value', 'Climate_Trend', 'Solar_Influence', 
                    'Dust_Events', 'Activity_Level', 'Mars_Index']
        available_cols = [col for col in corr_cols if col in df.columns]
        corr_matrix = df[available_cols].corr()
        
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='Hot',
            zmin=-1, zmax=1,
            text=corr_matrix.values.round(2),
            texttemplate='%{text}',
            textfont={"size": 10, "color": "white"},
            hoverongaps=False
        ))
        
        fig_corr.update_layout(
            title="Matrice de Corrélation",
            template='plotly_dark',
            height=500
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    
    with tab3:
        st.markdown("## 🚀 Chronologie des Missions Martiennes")
        
        if hasattr(analyzer, 'events') and analyzer.events:
            # Timeline
            fig_timeline = create_mission_timeline(analyzer.events)
            if fig_timeline:
                st.plotly_chart(fig_timeline, use_container_width=True)
            
            # Statistiques des missions
            col1, col2, col3 = st.columns(3)
            
            df_events = pd.DataFrame(analyzer.events)
            missions_df = df_events[df_events['type'] != 'storm']
            storms_df = df_events[df_events['type'] == 'storm']
            
            with col1:
                st.markdown("### Types de missions")
                mission_counts = missions_df['type'].value_counts()
                fig_pie = go.Figure(data=[go.Pie(
                    labels=mission_counts.index,
                    values=mission_counts.values,
                    hole=.3,
                    marker_colors=['#87CEEB', '#FFD700', '#FFA07A', '#FF4500']
                )])
                fig_pie.update_layout(
                    template='plotly_dark',
                    height=300,
                    showlegend=True
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                st.markdown("### Missions par décennie")
                missions_df['decade'] = (missions_df['year'] // 10) * 10
                decade_counts = missions_df['decade'].value_counts().sort_index()
                
                fig_decade = go.Figure(data=[go.Bar(
                    x=decade_counts.index,
                    y=decade_counts.values,
                    marker_color='#FF4500'
                )])
                fig_decade.update_layout(
                    template='plotly_dark',
                    height=300,
                    xaxis_title="Décennie",
                    yaxis_title="Nombre de missions"
                )
                st.plotly_chart(fig_decade, use_container_width=True)
            
            with col3:
                st.markdown("### Tempêtes de poussière")
                st.metric("Nombre total", len(storms_df))
                if len(storms_df) > 0:
                    avg_interval = (storms_df['year'].max() - storms_df['year'].min()) / len(storms_df)
                    st.metric("Intervalle moyen", f"{avg_interval:.1f} ans")
                    st.metric("Dernière tempête", f"{int(storms_df['year'].max())}")
            
            # Liste détaillée
            st.markdown("### 📋 Liste détaillée des événements")
            
            # Filtres
            col1, col2 = st.columns(2)
            with col1:
                selected_types = st.multiselect(
                    "Filtrer par type",
                    options=df_events['type'].unique(),
                    default=df_events['type'].unique()
                )
            with col2:
                selected_severity = st.multiselect(
                    "Filtrer par sévérité",
                    options=df_events['severity'].unique(),
                    default=df_events['severity'].unique()
                )
            
            # Filtrer et afficher
            filtered_events = df_events[
                (df_events['type'].isin(selected_types)) & 
                (df_events['severity'].isin(selected_severity))
            ].sort_values('year')
            
            for _, event in filtered_events.iterrows():
                severity_color = {
                    'historique': '#FF4500',
                    'majeur': '#FFA07A',
                    'extreme': '#FF0000'
                }.get(event['severity'], '#808080')
                
                st.markdown(f"""
                <div class="event-marker">
                    <span class="mission-badge">MY {event['mars_year']:.1f}</span>
                    <strong style="color: {severity_color};">{int(event['year'])}</strong> - 
                    {event['event']} 
                    <span style="color: #FFA07A; font-size: 0.9em;">({event['type']})</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Aucun événement enregistré pour cette période.")
    
    with tab4:
        st.markdown("## Statistiques Détaillées")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Résumé statistique")
            stats_df = df[['Base_Value', 'Activity_Level', 'Mars_Index', 
                          'Dust_Events', 'Solar_Influence']].describe()
            st.dataframe(
                stats_df.style.format("{:.2f}"),
                use_container_width=True
            )
        
        with col2:
            # Distribution
            fig_dist = make_subplots(rows=2, cols=1, 
                                     subplot_titles=('Distribution', 'Box Plot'))
            
            fig_dist.add_trace(
                go.Histogram(x=df['Base_Value'], nbinsx=30,
                            marker_color=analyzer.config['color'],
                            name='Distribution'),
                row=1, col=1
            )
            
            fig_dist.add_trace(
                go.Box(y=df['Base_Value'], name='Box Plot',
                      marker_color=analyzer.config['color'],
                      boxmean=True),
                row=2, col=1
            )
            
            fig_dist.update_layout(
                template='plotly_dark',
                height=500,
                showlegend=False
            )
            st.plotly_chart(fig_dist, use_container_width=True)
        
        # Analyse par décennie
        st.markdown("### Analyse par Décennie")
        df['Decade'] = (df['Earth_Year'] // 10) * 10
        decadal_stats = df.groupby('Decade').agg({
            'Base_Value': ['mean', 'std'],
            'Activity_Level': 'mean',
            'Dust_Events': 'mean'
        }).round(2)
        
        decadal_stats.columns = ['Moyenne', 'Écart-type', 'Activité moyenne', 'Tempêtes moyennes']
        st.dataframe(decadal_stats, use_container_width=True)
    
    with tab5:
        st.markdown("## 🔮 Projections Futures")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Paramètres de Simulation")
            projection_years = st.slider("Années de projection", 10, 50, 30)
            scenario = st.selectbox(
                "Scénario climatique",
                ["Optimiste", "Standard", "Pessimiste"]
            )
            
            # Facteurs selon le scénario
            scenario_factors = {
                "Optimiste": 0.8,
                "Standard": 1.0,
                "Pessimiste": 1.2
            }
            factor = scenario_factors[scenario]
        
        with col2:
            st.markdown("### Résultats des Projections")
            
            last_value = df['Base_Value'].iloc[-1]
            trend_rate = (df['Base_Value'].iloc[-1] - df['Base_Value'].iloc[-10]) / 10
            
            future_years = list(range(2025, 2025 + projection_years + 1))
            future_values = []
            
            for i, year in enumerate(future_years):
                projected = last_value + trend_rate * (i + 1) * factor
                uncertainty = 0.03 * (i + 1) * abs(last_value) * 0.1
                future_values.append(projected + np.random.normal(0, uncertainty))
            
            st.metric(
                "Valeur projetée (2050)",
                f"{future_values[25]:.1f} {analyzer.config['unit']}",
                delta=f"{((future_values[25]/last_value)-1)*100:+.1f}%"
            )
            
            if len(future_values) > 40:
                st.metric(
                    "Valeur projetée (2065)",
                    f"{future_values[40]:.1f} {analyzer.config['unit']}",
                    delta=f"{((future_values[40]/last_value)-1)*100:+.1f}%"
                )
        
        # Graphique des projections
        fig_proj = go.Figure()
        
        # Données historiques
        fig_proj.add_trace(go.Scatter(
            x=df['Earth_Year'][-50:],
            y=df['Base_Value'][-50:],
            mode='lines',
            name='Données historiques',
            line=dict(color=analyzer.config['color'], width=2)
        ))
        
        # Projections
        fig_proj.add_trace(go.Scatter(
            x=future_years,
            y=future_values,
            mode='lines',
            name=f'Projection {scenario}',
            line=dict(color='#00FFFF', width=2, dash='dash')
        ))
        
        fig_proj.update_layout(
            title=f"Projections sur {projection_years} ans",
            template='plotly_dark',
            xaxis_title="Année Terrestre",
            yaxis_title=analyzer.config['unit'],
            height=500
        )
        
        st.plotly_chart(fig_proj, use_container_width=True)
        
        # Recommandations pour l'exploration
        st.markdown("### 🎯 Recommandations pour l'Exploration")
        
        activity_level = df['Activity_Level'].iloc[-1]
        
        if activity_level < 30:
            st.info("""
            **✅ Conditions favorables pour l'exploration**
            - Période calme pour les missions en surface
            - Fenêtre optimale pour les observations
            - Risque de tempêtes minimal
            """)
        elif activity_level < 60:
            st.warning("""
            **⚠️ Vigilance recommandée**
            - Surveillance accrue des conditions
            - Préparer les protocoles d'urgence
            - Optimiser les fenêtres de communication
            """)
        else:
            st.error("""
            **🚨 Conditions difficiles**
            - Reporter les opérations sensibles
            - Protéger les équipements de la poussière
            - Maintenir les communications en sécurité
            """)
    
    with tab6:
        st.markdown("## 🪐 Visualisation de la Surface Martienne")
        
        # Carte 3D de Mars
        fig_globe = create_mars_globe_visualization(df, analyzer)
        st.plotly_chart(fig_globe, use_container_width=True)
        
        # Caractéristiques de surface
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 🏔️ Régions Volcaniques")
            st.markdown("""
            - Olympus Mons: +35%
            - Tharsis Montes: +25%
            - Elysium Planitia: +15%
            """)
        
        with col2:
            st.markdown("### 🏜️ Plaines et Déserts")
            st.markdown("""
            - Vastitas Borealis: -20%
            - Amazonis Planitia: -10%
            - Utopia Planitia: -15%
            """)
        
        with col3:
            st.markdown("### ❄️ Régions Polaires")
            st.markdown("""
            - Planum Boreum: +40% glace
            - Planum Australe: +35% glace
            - Variation saisonnière: ±30%
            """)
        
        # Opacité de la poussière
        st.markdown("### 🌫️ Opacité Atmosphérique")
        
        fig_opacity = go.Figure()
        fig_opacity.add_trace(go.Scatter(
            x=df['Earth_Year'],
            y=df['Dust_Opacity'],
            mode='lines',
            name='Opacité',
            line=dict(color='#8B4513', width=2),
            fill='tozeroy'
        ))
        
        fig_opacity.update_layout(
            template='plotly_dark',
            xaxis_title="Année Terrestre",
            yaxis_title="Opacité",
            height=300
        )
        st.plotly_chart(fig_opacity, use_container_width=True)
    
    # Footer avec téléchargement
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="mars_data.csv" style="text-decoration: none; color: #FF4500;">📥 Télécharger CSV</a>'
        st.markdown(href, unsafe_allow_html=True)
    
    with col2:
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Mars Data')
        excel_data = output.getvalue()
        b64_excel = base64.b64encode(excel_data).decode()
        href_excel = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64_excel}" download="mars_data.xlsx" style="text-decoration: none; color: #FF4500;">📊 Télécharger Excel</a>'
        st.markdown(href_excel, unsafe_allow_html=True)
    
    with col3:
        st.markdown('<a href="#" style="text-decoration: none; color: #FF4500;">📑 Rapport PDF</a>', unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"🪐 **Dashboard v2.0 - {datetime.now().strftime('%Y-%m-%d')}**")

if __name__ == "__main__":
    main()
