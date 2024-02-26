import dash
from dash import dcc, html
import plotly.express as px


class AccidentRoadDashboard:
    def __init__(self, t_test, anova_test, df, kpi):
        # Initialize the Dash app within the class
        self.t_test = t_test
        self.anova_test = anova_test
        self.dataframe = df
        self.kpi = kpi
        self.app = dash.Dash(__name__)
        self.setup_layout()

    def pre_process(self):
        self.dataframe['sex_of_casualty'] = self.dataframe['sex_of_casualty'].replace(1, 'male')
        self.dataframe['sex_of_casualty'] = self.dataframe['sex_of_casualty'].replace(2, 'female')
        self.dataframe = self.dataframe[(self.dataframe['sex_of_casualty'] == 'male') |
                                        (self.dataframe['sex_of_casualty'] == 'female')]
        self.dataframe['casualty_severity'] = self.dataframe['casualty_severity'].replace(1, 'fatal')
        self.dataframe['casualty_severity'] = self.dataframe['casualty_severity'].replace(2, 'serious')
        self.dataframe['casualty_severity'] = self.dataframe['casualty_severity'].replace(3, 'slight')
        self.dataframe = self.dataframe[(self.dataframe['sex_of_casualty'] == 'fatal') |
                                        (self.dataframe['sex_of_casualty'] == 'serious') |
                                        (self.dataframe['sex_of_casualty'] == 'slight')]

    def setup_layout(self):
        df_heatmap = self.dataframe[['pedestrian_location', 'pedestrian_movement', 'casualty_severity']]
        df_heatmap = df_heatmap.corr()
        # Heatmap
        heatmap_fig = px.density_heatmap(df_heatmap, x='pedestrian_location', y='pedestrian_movement',
                                         z='casualty_severity', histfunc="avg", color_continuous_scale="Viridis")
        heatmap_fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

        # Pie Chart 1
        pie_fig1 = px.pie(self.dataframe, values='accident_index', names='pedestrian_movement',
                          title='Moving Pedestrian Percentage')
        pie_fig1.update_layout(paper_bgcolor='rgba(0, 0, 0, 0)', plot_bgcolor='rgba(0,0,0,0)')

        # Pie Chart 2
        pie_fig2 = px.pie(self.dataframe, values='accident_index', names='casualty_severity',
                          title='Percent of Class Casualties')
        pie_fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

        # Bar Plot
        bar_fig = px.bar(self.dataframe, x='casualty_severity', y='car_passenger', color='sex_of_casualty',
                         barmode='group')
        bar_fig.update_layout(paper_bgcolor='rgba(0, 0, 0, 0)', plot_bgcolor='rgba(0, 0, 0, 0)')

        self.app.layout = html.Div(children=[
            # KPIs and other layout elements here...

            html.Div(children=[
                # Sub-divs for KPIs, displayed inline
                html.Div([html.H1(f"The p_value of the t_test between mean of casualty of male and female: {self.t_test}")],
                         style={'display': 'inline-block', 'margin': '10px'}),
                html.Div([html.H1(f"The f_value of the anova_test for type of casualty is: {self.anova_test}")],
                         style={'display': 'inline-block', 'margin': '10px'}),
                html.Div([html.H1(f"f1 score of model is  : {self.kpi}")],
                         style={'display': 'inline-block', 'margin': '10px'}),
            ], style={'display': 'flex', 'justify-content': 'space-around'}),
            html.Div(children=[
                dcc.Graph(figure=heatmap_fig),
                dcc.Graph(figure=pie_fig1),
            ], style={'display': 'flex', 'justify-content': 'space-around',
                      'backgroundColor': 'rgba(0, 0, 0, 0)'}),

            html.Div(children=[
                dcc.Graph(figure=pie_fig2),
                dcc.Graph(figure=bar_fig),
            ], style={'display': 'flex', 'justify-content': 'space-around',
                      'backgroundColor': 'rgba(0, 0, 0, 0)'}),
        ], style={'backgroundColor': 'rgba(224, 224, 224, 0.5)'})

    def run(self):
            # Method to run the server
            if __name__ == '__main__':
                self.app.run_server(debug=True)


