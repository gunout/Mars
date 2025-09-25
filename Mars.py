import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class MarsDataAnalyzer:
    def __init__(self, data_type):
        self.data_type = data_type
        self.colors = ['#FF4500', '#C1440E', '#8B4513', '#D2691E', '#CD5C5C', 
                      '#A52A2A', '#800000', '#B22222', '#DC143C', '#FF6347']
        
        self.start_year = 1965  # Début des observations martiennes sérieuses
        self.end_year = 2025
        
        # Configuration spécifique pour chaque type de données martiennes
        self.config = self._get_mars_config()
        
    def _get_mars_config(self):
        """Retourne la configuration spécifique pour chaque type de données martiennes"""
        configs = {
            "temperature": {
                "base_value": -63,
                "cycle_years": 1.88,  # Année martienne en années terrestres
                "amplitude": 70,
                "trend": "stable",
                "unit": "°C",
                "description": "Température moyenne martienne"
            },
            "atmospheric_pressure": {
                "base_value": 6.1,
                "cycle_years": 1.88,
                "amplitude": 0.5,
                "trend": "variable",
                "unit": "mbar",
                "description": "Pression atmosphérique"
            },
            "dust_storms": {
                "base_value": 50,
                "cycle_years": 3.0,
                "amplitude": 40,
                "trend": "cyclique",
                "unit": "Index de poussière",
                "description": "Activité des tempêtes de poussière"
            },
            "co2_ice": {
                "base_value": 30,
                "cycle_years": 1.88,
                "amplitude": 25,
                "trend": "saisonnier",
                "unit": "% de couverture",
                "description": "Glace carbonique aux pôles"
            },
            "water_ice": {
                "base_value": 15,
                "cycle_years": 1.88,
                "amplitude": 10,
                "trend": "stable",
                "unit": "% de couverture",
                "description": "Glace d'eau souterraine"
            },
            "solar_radiation": {
                "base_value": 600,
                "cycle_years": 11.0,
                "amplitude": 100,
                "trend": "cyclique",
                "unit": "W/m²",
                "description": "Radiation solaire en surface"
            },
            "magnetic_field": {
                "base_value": 0,
                "cycle_years": 0,
                "amplitude": 5,
                "trend": "localisé",
                "unit": "nT",
                "description": "Champ magnétique résiduel"
            },
            "seismic_activity": {
                "base_value": 1,
                "cycle_years": 0,
                "amplitude": 2,
                "trend": "sporadique",
                "unit": "Magnitude",
                "description": "Activité sismique (marsquakes)"
            },
            "orbital_distance": {
                "base_value": 1.52,
                "cycle_years": 2.14,
                "amplitude": 0.14,
                "trend": "périodique",
                "unit": "UA",
                "description": "Distance au Soleil"
            },
            # Configuration par défaut
            "default": {
                "base_value": 100,
                "cycle_years": 1.88,
                "amplitude": 20,
                "trend": "stable",
                "unit": "Unités",
                "description": "Données martiennes génériques"
            }
        }
        
        return configs.get(self.data_type, configs["default"])
    
    def generate_mars_data(self):
        """Génère des données martiennes simulées basées sur les cycles saisonniers et orbitaux"""
        print(f"🪐 Génération des données martiennes pour {self.config['description']}...")
        
        # Créer une base de données annuelle (en années terrestres)
        dates = pd.date_range(start=f'{self.start_year}-01-01', 
                             end=f'{self.end_year}-12-31', freq='Y')
        
        data = {'Earth_Year': [date.year for date in dates]}
        data['Mars_Year'] = self._earth_to_mars_years(dates)
        
        # Données principales basées sur les cycles martiens
        data['Base_Value'] = self._simulate_mars_cycle(dates)
        data['Seasonal_Min'] = self._simulate_seasonal_minima(dates)
        data['Seasonal_Max'] = self._simulate_seasonal_maxima(dates)
        data['Orbital_Phase'] = self._simulate_orbital_phase(dates)
        
        # Variations climatiques
        data['Climate_Trend'] = self._simulate_climate_trend(dates)
        data['Dust_Events'] = self._simulate_dust_events(dates)
        data['Solar_Influence'] = self._simulate_solar_influence(dates)
        
        # Données dérivées
        data['Smoothed_Value'] = self._simulate_smoothed_data(dates)
        data['Seasonal_Variation'] = self._simulate_seasonal_variation(dates)
        data['Annual_Variation'] = self._simulate_annual_variation(dates)
        
        # Indices martiens complémentaires
        data['Mars_Index'] = self._simulate_mars_index(dates)
        data['Activity_Level'] = self._simulate_activity_level(dates)
        data['Future_Prediction'] = self._simulate_future_prediction(dates)
        
        df = pd.DataFrame(data)
        
        # Ajouter des événements martiens historiques
        self._add_mars_events(df)
        
        return df
    
    def _earth_to_mars_years(self, dates):
        """Convertit les années terrestres en années martiennes"""
        mars_years = []
        mars_year_start = 1965  # Correspond à l'année martienne 1 pour nos données
        mars_year_duration = 1.88  # Années terrestres
        
        for date in dates:
            earth_year = date.year
            mars_year = 1 + (earth_year - mars_year_start) / mars_year_duration
            mars_years.append(mars_year)
        
        return mars_years
    
    def _simulate_mars_cycle(self, dates):
        """Simule le cycle martien principal (saisons et orbite)"""
        base_value = self.config["base_value"]
        cycle_years = self.config["cycle_years"]
        amplitude = self.config["amplitude"]
        
        values = []
        for i, date in enumerate(dates):
            earth_year = date.year
            
            # Cycle saisonnier martien (1.88 années terrestres)
            mars_phase = (earth_year - self.start_year) % cycle_years
            seasonal_cycle = np.sin(2 * np.pi * mars_phase / cycle_years)
            
            # Cycle orbital (2.14 années terrestres)
            orbital_phase = (earth_year - self.start_year) % 2.14
            orbital_cycle = np.cos(2 * np.pi * orbital_phase / 2.14)
            
            # Combinaison des cycles
            if self.config["trend"] == "saisonnier":
                value = base_value + amplitude * seasonal_cycle
            elif self.config["trend"] == "périodique":
                value = base_value + amplitude * 0.7 * orbital_cycle
            elif self.config["trend"] == "cyclique":
                value = base_value + amplitude * (0.6 * seasonal_cycle + 0.4 * orbital_cycle)
            else:
                value = base_value + amplitude * 0.3 * seasonal_cycle
            
            # Bruit martien naturel
            noise = np.random.normal(0, amplitude * 0.08)
            values.append(value + noise)
        
        return values
    
    def _simulate_seasonal_minima(self, dates):
        """Simule les périodes de minimum saisonnier martien"""
        minima = []
        for i, date in enumerate(dates):
            earth_year = date.year
            mars_phase = (earth_year - self.start_year) % 1.88
            
            # Minimum pendant l'hiver martien
            if 1.4 <= mars_phase <= 1.88 or 0 <= mars_phase <= 0.48:
                min_factor = 0.3
            else:
                min_factor = 0.8
            
            minima.append(min_factor)
        
        return minima
    
    def _simulate_seasonal_maxima(self, dates):
        """Simule les périodes de maximum saisonnier martien"""
        maxima = []
        for i, date in enumerate(dates):
            earth_year = date.year
            mars_phase = (earth_year - self.start_year) % 1.88
            
            # Maximum pendant l'été martien
            if 0.9 <= mars_phase <= 1.4:
                max_factor = 1.0
            else:
                max_factor = 0.6
            
            maxima.append(max_factor)
        
        return maxima
    
    def _simulate_orbital_phase(self, dates):
        """Simule la phase orbitale (0-1)"""
        phases = []
        for date in dates:
            earth_year = date.year
            phase = (earth_year - self.start_year) % 2.14 / 2.14
            phases.append(phase)
        
        return phases
    
    def _simulate_climate_trend(self, dates):
        """Simule les tendances climatiques à long terme"""
        trends = []
        for i, date in enumerate(dates):
            earth_year = date.year
            
            # Variations climatiques à long terme sur Mars
            if earth_year < 1975:
                trend = 0.95  # Période pré-observation détaillée
            elif 1975 <= earth_year < 1990:
                trend = 1.0 + 0.001 * (earth_year - 1975)  # Missions Viking
            elif 1990 <= earth_year < 2000:
                trend = 1.02 + 0.002 * (earth_year - 1990)  # Début exploration moderne
            elif 2000 <= earth_year < 2010:
                trend = 1.04 + 0.003 * (earth_year - 2000)  # Rovers Spirit/Opportunity
            elif 2010 <= earth_year < 2020:
                trend = 1.07 + 0.004 * (earth_year - 2010)  # Curiosity
            else:
                trend = 1.11 + 0.005 * (earth_year - 2020)  # Perseverance
            
            trends.append(trend)
        
        return trends
    
    def _simulate_dust_events(self, dates):
        """Simule les événements de poussière (tempêtes globales)"""
        dust_events = []
        for date in dates:
            earth_year = date.year
            mars_year = 1 + (earth_year - 1965) / 1.88
            
            # Tempêtes globales environ tous les 3 ans martiens
            dust_cycle = mars_year % 3.0
            if 2.8 <= dust_cycle <= 3.0 or 0 <= dust_cycle <= 0.2:
                dust_level = 1.8  # Tempête globale
            elif 2.6 <= dust_cycle <= 2.8 or 0.2 <= dust_cycle <= 0.4:
                dust_level = 1.4  # Tempête régionale
            else:
                dust_level = 1.0  # Conditions normales
            
            dust_events.append(dust_level)
        
        return dust_events
    
    def _simulate_solar_influence(self, dates):
        """Simule l'influence du cycle solaire sur Mars"""
        solar_effects = []
        for date in dates:
            earth_year = date.year
            
            # Cycle solaire de 11 ans affectant Mars
            solar_phase = (earth_year - self.start_year) % 11.0
            solar_effect = 1.0 + 0.1 * np.sin(2 * np.pi * solar_phase / 11.0)
            
            solar_effects.append(solar_effect)
        
        return solar_effects
    
    def _simulate_smoothed_data(self, dates):
        """Simule des données lissées (moyenne mobile sur 2 années martiennes)"""
        base_cycle = self._simulate_mars_cycle(dates)
        
        smoothed = []
        window_size = int(2 * 1.88)  # 2 années martiennes en années terrestres
        
        for i in range(len(base_cycle)):
            start_idx = max(0, i - window_size//2)
            end_idx = min(len(base_cycle), i + window_size//2 + 1)
            window = base_cycle[start_idx:end_idx]
            smoothed.append(np.mean(window))
        
        return smoothed
    
    def _simulate_seasonal_variation(self, dates):
        """Simule les variations saisonnières"""
        variations = []
        for date in dates:
            earth_year = date.year
            # Variation basée sur la saison martienne
            mars_season = (earth_year - self.start_year) % 1.88 / 1.88
            seasonal_variation = 0.2 * np.sin(2 * np.pi * mars_season)
            variations.append(1 + seasonal_variation)
        
        return variations
    
    def _simulate_annual_variation(self, dates):
        """Simule les variations annuelles terrestres (pour calibration)"""
        variations = []
        for i, date in enumerate(dates):
            earth_year = date.year
            # Variation annuelle terrestre (pour les observations)
            annual_variation = 0.05 * np.sin(2 * np.pi * (earth_year - self.start_year) / 1.0)
            variations.append(1 + annual_variation)
        
        return variations
    
    def _simulate_mars_index(self, dates):
        """Simule un indice martien composite"""
        indices = []
        base_cycle = self._simulate_mars_cycle(dates)
        climate_trend = self._simulate_climate_trend(dates)
        dust_events = self._simulate_dust_events(dates)
        
        for i in range(len(dates)):
            # Indice composite pondéré
            index = (base_cycle[i] * 0.5 + 
                    climate_trend[i] * self.config["base_value"] * 0.3 +
                    dust_events[i] * 10 * 0.2)
            indices.append(index)
        
        return indices
    
    def _simulate_activity_level(self, dates):
        """Simule le niveau d'activité environnementale (0-100)"""
        activity_levels = []
        dust_events = self._simulate_dust_events(dates)
        
        for dust_level in dust_events:
            # Niveau d'activité basé sur les tempêtes de poussière
            activity = min(100, (dust_level - 1) * 125)
            activity_levels.append(activity)
        
        return activity_levels
    
    def _simulate_future_prediction(self, dates):
        """Simule des prédictions futures"""
        predictions = []
        base_cycle = self._simulate_mars_cycle(dates)
        climate_trend = self._simulate_climate_trend(dates)
        
        for i, date in enumerate(dates):
            earth_year = date.year
            current_value = base_cycle[i]
            trend_factor = climate_trend[i]
            
            if earth_year > 2020:  # Période de prédiction
                # Ajouter une incertitude croissante
                years_since_2020 = earth_year - 2020
                uncertainty = 0.04 * years_since_2020
                prediction = current_value * trend_factor * (1 + np.random.normal(0, uncertainty))
            else:
                prediction = current_value
            
            predictions.append(prediction)
        
        return predictions
    
    def _add_mars_events(self, df):
        """Ajoute des événements martiens historiques significatifs"""
        for i, row in df.iterrows():
            earth_year = row['Earth_Year']
            
            # Événements d'observation martienne
            if earth_year == 1965:
                # Mariner 4 - premières images rapprochées
                df.loc[i, 'Activity_Level'] = 20
            
            elif earth_year == 1971:
                # Mariner 9 - première cartographie
                df.loc[i, 'Activity_Level'] = 40
            
            elif 1976 <= earth_year <= 1977:
                # Vikings 1 et 2 - premières atterrissages
                df.loc[i, 'Base_Value'] *= 1.1
                df.loc[i, 'Activity_Level'] = 60
            
            elif earth_year == 1997:
                # Mars Pathfinder - retour après 20 ans
                df.loc[i, 'Activity_Level'] = 50
            
            elif 2004 <= earth_year <= 2005:
                # Rovers Spirit et Opportunity
                df.loc[i, 'Base_Value'] *= 1.15
                df.loc[i, 'Activity_Level'] = 70
            
            elif earth_year == 2006:
                # MRO - orbiter haute résolution
                df.loc[i, 'Activity_Level'] = 65
            
            elif earth_year == 2012:
                # Curiosity - rover avancé
                df.loc[i, 'Base_Value'] *= 1.2
                df.loc[i, 'Activity_Level'] = 80
            
            elif earth_year == 2018:
                # InSight - sismomètre
                if self.data_type == "seismic_activity":
                    df.loc[i, 'Base_Value'] *= 2.0
            
            elif earth_year == 2021:
                # Perseverance + Ingenuity
                df.loc[i, 'Base_Value'] *= 1.25
                df.loc[i, 'Activity_Level'] = 90
            
            # Grandes tempêtes de poussière documentées
            if earth_year in [1971, 1977, 1982, 1994, 2001, 2007, 2018]:
                df.loc[i, 'Dust_Events'] *= 1.8
                df.loc[i, 'Activity_Level'] = min(100, df.loc[i, 'Activity_Level'] * 1.5)
    
    def create_mars_analysis(self, df):
        """Crée une analyse complète des données martiennes"""
        plt.style.use('dark_background')  # Fond sombre pour l'espace
        fig = plt.figure(figsize=(20, 28))
        
        # 1. Cycle martien principal
        ax1 = plt.subplot(5, 2, 1)
        self._plot_mars_cycle(df, ax1)
        
        # 2. Activité environnementale
        ax2 = plt.subplot(5, 2, 2)
        self._plot_environmental_activity(df, ax2)
        
        # 3. Variations saisonnières
        ax3 = plt.subplot(5, 2, 3)
        self._plot_seasonal_variations(df, ax3)
        
        # 4. Tendances climatiques
        ax4 = plt.subplot(5, 2, 4)
        self._plot_climate_trends(df, ax4)
        
        # 5. Phase orbitale
        ax5 = plt.subplot(5, 2, 5)
        self._plot_orbital_phase(df, ax5)
        
        # 6. Données lissées
        ax6 = plt.subplot(5, 2, 6)
        self._plot_smoothed_data_plot(df, ax6)
        
        # 7. Niveau d'activité
        ax7 = plt.subplot(5, 2, 7)
        self._plot_activity_level_plot(df, ax7)
        
        # 8. Événements de poussière
        ax8 = plt.subplot(5, 2, 8)
        self._plot_dust_events(df, ax8)
        
        # 9. Indice martien
        ax9 = plt.subplot(5, 2, 9)
        self._plot_mars_index(df, ax9)
        
        # 10. Prédictions futures
        ax10 = plt.subplot(5, 2, 10)
        self._plot_future_predictions(df, ax10)
        
        plt.suptitle(f'Analyse des Données Martiennes: {self.config["description"]} ({self.start_year}-{self.end_year})', 
                    fontsize=16, fontweight='bold', color='#FF4500')
        plt.tight_layout()
        plt.savefig(f'mars_{self.data_type}_analysis.png', dpi=300, bbox_inches='tight', 
                   facecolor='black', edgecolor='none')
        plt.show()
        
        # Générer les insights
        self._generate_mars_insights(df)
    
    def _plot_mars_cycle(self, df, ax):
        """Plot du cycle martien principal"""
        ax.plot(df['Earth_Year'], df['Base_Value'], label='Valeur de base', 
               linewidth=2, color='#FF4500', alpha=0.9)
        
        ax.set_title(f'Cycle Martien Principal - {self.config["description"]}', 
                    fontsize=12, fontweight='bold', color='#FF4500')
        ax.set_ylabel(self.config["unit"], color='#FF4500')
        ax.tick_params(axis='y', labelcolor='#FF4500')
        ax.grid(True, alpha=0.2, color='white')
        ax.set_facecolor('black')
        
        # Ajouter des annotations pour les années martiennes
        for mars_year in range(1, 35, 5):  # Tous les 5 ans martiens
            earth_year = 1965 + (mars_year - 1) * 1.88
            if earth_year in df['Earth_Year'].values:
                ax.axvline(x=earth_year, alpha=0.3, color='orange', linestyle='--')
                ax.text(earth_year, ax.get_ylim()[1]*0.9, f'MY{mars_year}', 
                       rotation=90, color='orange', alpha=0.7, fontsize=8)
    
    def _plot_environmental_activity(self, df, ax):
        """Plot de l'activité environnementale martienne"""
        ax.fill_between(df['Earth_Year'], df['Base_Value'], alpha=0.7, 
                       color='#C1440E', label='Activité environnementale')
        
        ax.set_title('Activité Environnementale Martienne', fontsize=12, fontweight='bold', color='#FF4500')
        ax.set_ylabel(self.config["unit"], color='#C1440E')
        ax.set_xlabel('Année Terrestre', color='white')
        ax.tick_params(axis='y', labelcolor='#C1440E')
        ax.tick_params(axis='x', labelcolor='white')
        ax.grid(True, alpha=0.2, color='white')
        ax.set_facecolor('black')
        
        # Marquer les missions importantes
        missions = {
            1976: 'Viking\natterrit',
            1997: 'Pathfinder',
            2004: 'Spirit/\nOpportunity',
            2012: 'Curiosity',
            2021: 'Perseverance'
        }
        
        for year, label in missions.items():
            if year in df['Earth_Year'].values:
                y_val = df.loc[df['Earth_Year'] == year, 'Base_Value'].values[0]
                ax.annotate(label, xy=(year, y_val), xytext=(year, y_val*1.2),
                           arrowprops=dict(arrowstyle='->', color='yellow'),
                           color='yellow', fontsize=8, ha='center')
    
    def _plot_seasonal_variations(self, df, ax):
        """Plot des variations saisonnières martiennes"""
        ax.plot(df['Earth_Year'], df['Seasonal_Min'], label='Minimum saisonnier', 
               color='#1E90FF', alpha=0.7)
        ax.plot(df['Earth_Year'], df['Seasonal_Max'], label='Maximum saisonnier', 
               color='#FF6347', alpha=0.7)
        
        ax.set_title('Variations Saisonnières Martiennes', fontsize=12, fontweight='bold', color='#FF4500')
        ax.set_ylabel('Facteur d\'amplitude', color='white')
        ax.legend()
        ax.grid(True, alpha=0.2, color='white')
        ax.set_facecolor('black')
        ax.tick_params(colors='white')
    
    def _plot_climate_trends(self, df, ax):
        """Plot des tendances climatiques martiennes"""
        ax.plot(df['Earth_Year'], df['Climate_Trend'], label='Tendance climatique', 
               linewidth=2, color='#FFD700')
        ax.plot(df['Earth_Year'], df['Solar_Influence'], label='Influence solaire', 
               linewidth=2, color='#00BFFF')
        
        ax.set_title('Tendances Climatiques et Influence Solaire', fontsize=12, fontweight='bold', color='#FF4500')
        ax.set_ylabel('Facteur multiplicatif', color='white')
        ax.legend()
        ax.grid(True, alpha=0.2, color='white')
        ax.set_facecolor('black')
        ax.tick_params(colors='white')
    
    def _plot_orbital_phase(self, df, ax):
        """Plot de la phase orbitale"""
        scatter = ax.scatter(df['Earth_Year'], df['Orbital_Phase'], c=df['Orbital_Phase'], 
                           cmap='coolwarm', alpha=0.7, s=20)
        
        ax.set_title('Phase Orbitale Martienne (0-1)', fontsize=12, fontweight='bold', color='#FF4500')
        ax.set_ylabel('Phase orbitale', color='white')
        ax.set_xlabel('Année Terrestre', color='white')
        plt.colorbar(scatter, ax=ax, label='Phase')
        ax.grid(True, alpha=0.2, color='white')
        ax.set_facecolor('black')
        ax.tick_params(colors='white')
    
    def _plot_smoothed_data_plot(self, df, ax):
        """Plot des données lissées"""
        ax.plot(df['Earth_Year'], df['Base_Value'], label='Données brutes', 
               alpha=0.5, color='#FF6347')
        ax.plot(df['Earth_Year'], df['Smoothed_Value'], label='Données lissées (2 ans martiens)', 
               linewidth=2, color='#00FF7F')
        
        ax.set_title('Données Brutes vs Lissées', fontsize=12, fontweight='bold', color='#FF4500')
        ax.set_ylabel(self.config["unit"], color='white')
        ax.legend()
        ax.grid(True, alpha=0.2, color='white')
        ax.set_facecolor('black')
        ax.tick_params(colors='white')
    
    def _plot_activity_level_plot(self, df, ax):
        """Plot du niveau d'activité environnementale"""
        ax.fill_between(df['Earth_Year'], df['Activity_Level'], alpha=0.6, 
                       color='#FF4500', label='Niveau d\'activité')
        ax.plot(df['Earth_Year'], df['Activity_Level'], color='#FF8C00', alpha=0.8)
        
        ax.set_title('Niveau d\'Activité Environnementale (0-100)', fontsize=12, fontweight='bold', color='#FF4500')
        ax.set_ylabel('Niveau d\'activité', color='white')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.2, color='white')
        ax.set_facecolor('black')
        ax.tick_params(colors='white')
    
    def _plot_dust_events(self, df, ax):
        """Plot des événements de poussière"""
        ax.fill_between(df['Earth_Year'], df['Dust_Events'], alpha=0.6, 
                       color='#8B4513', label='Activité des tempêtes')
        
        ax.set_title('Tempêtes de Poussière Martiennes', fontsize=12, fontweight='bold', color='#FF4500')
        ax.set_ylabel('Intensité relative', color='white')
        ax.grid(True, alpha=0.2, color='white')
        ax.set_facecolor('black')
        ax.tick_params(colors='white')
    
    def _plot_mars_index(self, df, ax):
        """Plot de l'indice martien composite"""
        ax.plot(df['Earth_Year'], df['Mars_Index'], label='Indice martien composite', 
               linewidth=2, color='#DA70D6')
        
        ax.set_title('Indice Martien Composite', fontsize=12, fontweight='bold', color='#FF4500')
        ax.set_ylabel('Valeur de l\'indice', color='white')
        ax.grid(True, alpha=0.2, color='white')
        ax.set_facecolor('black')
        ax.tick_params(colors='white')
    
    def _plot_future_predictions(self, df, ax):
        """Plot des prédictions futures"""
        ax.plot(df['Earth_Year'], df['Base_Value'], label='Données historiques', 
               color='#FF6347', alpha=0.7)
        ax.plot(df['Earth_Year'], df['Future_Prediction'], label='Projections', 
               linewidth=2, color='#00FFFF', linestyle='--')
        
        ax.axvline(x=2020, color='yellow', linestyle=':', alpha=0.7, label='Début des prédictions')
        
        ax.set_title('Données Historiques et Projections Futures', fontsize=12, fontweight='bold', color='#FF4500')
        ax.set_ylabel(self.config["unit"], color='white')
        ax.legend()
        ax.grid(True, alpha=0.2, color='white')
        ax.set_facecolor('black')
        ax.tick_params(colors='white')
    
    def _generate_mars_insights(self, df):
        """Génère des insights analytiques sur les données martiennes"""
        print(f"🪐 INSIGHTS ANALYTIQUES - {self.config['description']}")
        print("=" * 70)
        
        # 1. Statistiques de base
        print("\n1. 📊 STATISTIQUES FONDAMENTALES:")
        avg_value = df['Base_Value'].mean()
        max_value = df['Base_Value'].max()
        min_value = df['Base_Value'].min()
        current_value = df['Base_Value'].iloc[-1]
        
        print(f"Valeur moyenne: {avg_value:.2f} {self.config['unit']}")
        print(f"Valeur maximale: {max_value:.2f} {self.config['unit']}")
        print(f"Valeur minimale: {min_value:.2f} {self.config['unit']}")
        print(f"Valeur actuelle: {current_value:.2f} {self.config['unit']}")
        
        # 2. Analyse des cycles
        print("\n2. 🔄 ANALYSE DES CYCLES MARTIENS:")
        mars_year_current = df['Mars_Year'].iloc[-1]
        orbital_period = 2.14  # Années terrestres
        seasonal_period = 1.88  # Années terrestres
        
        print(f"Année martienne actuelle: {mars_year_current:.1f}")
        print(f"Période orbitale: {orbital_period} années terrestres")
        print(f"Période saisonnière: {seasonal_period} années terrestres")
        print(f"Type de tendance: {self.config['trend']}")
        
        # 3. Activité récente
        print("\n3. 📈 ACTIVITÉ RÉCENTE:")
        recent_data = df[df['Earth_Year'] >= 2010]
        avg_recent = recent_data['Base_Value'].mean()
        trend_recent = (recent_data['Base_Value'].iloc[-1] / 
                       recent_data['Base_Value'].iloc[0] - 1) * 100
        
        print(f"Moyenne depuis 2010: {avg_recent:.2f} {self.config['unit']}")
        print(f"Évolution depuis 2010: {trend_recent:+.1f}%")
        
        # 4. Événements majeurs
        print("\n4. 🌪️ ÉVÉNEMENTS MARTIENS MARQUANTS:")
        print("• 1965: Mariner 4 - premières images rapprochées")
        print("• 1971: Mariner 9 - première cartographie complète")
        print("• 1976: Vikings 1 et 2 - premiers atterrissages")
        print("• 1997: Mars Pathfinder - retour après 20 ans")
        print("• 2004: Spirit et Opportunity - exploration longue durée")
        print("• 2012: Curiosity - laboratoire mobile avancé")
        print("• 2018: InSight - étude de la structure interne")
        print("• 2021: Perseverance - recherche de vie + Ingenuity")
        
        # 5. Caractéristiques orbitales
        print("\n5. 🛰️ CARACTÉRISTIQUES ORBITALES:")
        phase_current = df['Orbital_Phase'].iloc[-1]
        activity_current = df['Activity_Level'].iloc[-1]
        
        print(f"Phase orbitale actuelle: {phase_current:.2f}")
        print(f"Niveau d'activité actuel: {activity_current:.1f}%")
        
        if phase_current < 0.25:
            print("→ Périhélie approchant - plus proche du Soleil")
        elif phase_current < 0.75:
            print("→ Aphélie approchant - plus loin du Soleil")
        else:
            print("→ Retour vers périhélie")
        
        # 6. Projections futures
        print("\n6. 🔮 PROJECTIONS FUTURES:")
        predicted_growth = ((df['Future_Prediction'].iloc[-1] / 
                           df['Base_Value'].iloc[-1]) - 1) * 100
        
        print(f"Tendance projetée: {predicted_growth:+.1f}%")
        print("2026: Mission de retour d'échantillons (planifiée)")
        print("2030s: Première mission habitée potentielle")
        print("2040s: Base permanente possible")
        
        # 7. Implications scientifiques
        print("\n7. 🎯 IMPLICATIONS SCIENTIFIQUES:")
        if self.data_type == "temperature":
            print("• Compréhension de l'évolution climatique")
            print("• Implications pour l'habitabilité")
            print("• Comparaison avec le changement climatique terrestre")
        
        elif self.data_type == "dust_storms":
            print("• Impact sur les missions robotiques")
            print("• Implications pour les futures missions habitées")
            print("• Compréhension des phénomènes atmosphériques")
        
        elif self.data_type == "water_ice":
            print("• Ressource cruciale pour l'exploration")
            print("• Indicateur d'habitabilité passée")
            print("• Potentiel pour la production de carburant")
        
        elif self.data_type == "seismic_activity":
            print("• Compréhension de la structure interne")
            print("• Évaluation de l'activité géologique")
            print("• Implications pour la présence d'eau liquide")
        
        print("• Préparation pour l'exploration humaine")
        print("• Recherche de signes de vie passée")
        print("• Compréhension de l'évolution planétaire")

def main():
    """Fonction principale pour l'analyse des données martiennes"""
    # Types de données martiennes disponibles
    mars_data_types = [
        "temperature", "atmospheric_pressure", "dust_storms", "co2_ice",
        "water_ice", "solar_radiation", "magnetic_field", "seismic_activity", "orbital_distance"
    ]
    
    print("🪐 ANALYSE DES DONNÉES NUMÉRIQUES DE MARS (1965-2025)")
    print("=" * 65)
    
    # Demander à l'utilisateur de choisir un type de données
    print("Types de données martiennes disponibles:")
    for i, data_type in enumerate(mars_data_types, 1):
        analyzer_temp = MarsDataAnalyzer(data_type)
        print(f"{i}. {analyzer_temp.config['description']}")
    
    try:
        choix = int(input("\nChoisissez le numéro du type de données à analyser: "))
        if choix < 1 or choix > len(mars_data_types):
            raise ValueError
        selected_type = mars_data_types[choix-1]
    except (ValueError, IndexError):
        print("Choix invalide. Sélection de la température par défaut.")
        selected_type = "temperature"
    
    # Initialiser l'analyseur
    analyzer = MarsDataAnalyzer(selected_type)
    
    # Générer les données
    mars_data = analyzer.generate_mars_data()
    
    # Sauvegarder les données
    output_file = f'mars_{selected_type}_data_1965_2025.csv'
    mars_data.to_csv(output_file, index=False)
    print(f"💾 Données sauvegardées: {output_file}")
    
    # Aperçu des données
    print("\n👀 Aperçu des données:")
    print(mars_data[['Earth_Year', 'Mars_Year', 'Base_Value', 'Activity_Level', 'Mars_Index']].head())
    
    # Créer l'analyse
    print("\n📈 Création de l'analyse des données martiennes...")
    analyzer.create_mars_analysis(mars_data)
    
    print(f"\n✅ Analyse des données {analyzer.config['description']} terminée!")
    print(f"📊 Période: {analyzer.start_year}-{analyzer.end_year} (années terrestres)")
    print(f"🪐 Couverture: ~{(2025-1965)/1.88:.1f} années martiennes")
    print("🌡️ Données: Climat, géologie, orbite, activités")

if __name__ == "__main__":
    main()