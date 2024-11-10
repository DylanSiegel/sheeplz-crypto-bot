import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

class Visualizer:
    def __init__(self, data_array, timestamps, price_columns, indicator_columns, sequence_length):
        self.data_array = data_array
        self.timestamps = timestamps
        self.price_columns = price_columns
        self.indicator_columns = indicator_columns
        self.sequence_length = sequence_length

    def _reduce_dimensions(self, encoded_states, method='pca', perplexity=50):
        if method == 'pca':
            reducer = PCA(n_components=3, random_state=42)
        elif method == 'tsne':
            reducer = TSNE(
                n_components=3,
                perplexity=min(perplexity, len(encoded_states) - 1),
                n_jobs=1,
                random_state=42,
                init='pca',
                method='barnes_hut'
            )
        else:
            raise ValueError(f"Unsupported projection method: {method}")

        return reducer.fit_transform(encoded_states)

    def visualize(self, encoded_states, timestamps, indices, color_feature='macd', hover_features=None, projection_method='pca', create_subplots=True):
        reduced_states = self._reduce_dimensions(encoded_states, method=projection_method)
        timestamps = [datetime.fromtimestamp(ts) for ts in timestamps]

        if create_subplots:
            return self._create_subplot_visualization(reduced_states, timestamps, indices, color_feature, hover_features)
        else:
            return self._create_single_visualization(reduced_states, timestamps, indices, color_feature, hover_features)

    def _create_subplot_visualization(self, reduced_states, timestamps, indices, color_feature, hover_features):
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                '3D View',
                'Top View (XY)',
                'Front View (XZ)',
                'Side View (YZ)'
            ),
            specs=[[{'type': 'scatter3d'}, {'type': 'scatter'}],
                   [{'type': 'scatter'}, {'type': 'scatter'}]]
        )

        color_data = self._get_feature_values(color_feature, indices)
        hover_data = self._prepare_hover_data(hover_features, timestamps, indices)

        fig.add_trace(
            go.Scatter3d(
                x=reduced_states[:, 0],
                y=reduced_states[:, 1],
                z=reduced_states[:, 2],
                mode='markers',
                marker=dict(
                    size=4,
                    color=color_data,
                    colorscale='Viridis',
                    showscale=True
                ),
                text=hover_data,
                hoverinfo='text'
            ),
            row=1, col=1
        )

        projections = [
            (0, 1, 1, 2),
            (0, 2, 2, 1),
            (1, 2, 2, 2)
        ]

        for x_idx, y_idx, row, col in projections:
            fig.add_trace(
                go.Scatter(
                    x=reduced_states[:, x_idx],
                    y=reduced_states[:, y_idx],
                    mode='markers',
                    marker=dict(
                        size=4,
                        color=color_data,
                        colorscale='Viridis',
                        showscale=False
                    ),
                    text=hover_data,
                    hoverinfo='text'
                ),
                row=row, col=col
            )

        fig.update_layout(
            title='Market State Visualization - Multiple Views',
            showlegend=False,
            height=800,
            width=1200,
            template='plotly_dark',
            margin=dict(l=20, r=20, t=40, b=20)
        )

        return fig

    def _create_single_visualization(self, reduced_states, timestamps, indices, color_feature, hover_features):
        color_data = self._get_feature_values(color_feature, indices)
        hover_data = self._prepare_hover_data(hover_features, timestamps, indices)

        fig = go.Figure(data=[
            go.Scatter3d(
                x=reduced_states[:, 0],
                y=reduced_states[:, 1],
                z=reduced_states[:, 2],
                mode='markers',
                marker=dict(
                    size=4,
                    color=color_data,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title=color_feature)
                ),
                text=hover_data,
                hoverinfo='text'
            )
        ])

        fig.update_layout(
            title=f'Market State Visualization - {color_feature}',
            template='plotly_dark',
            scene=dict(
                xaxis_title='Component 1',
                yaxis_title='Component 2',
                zaxis_title='Component 3'
            ),
            width=1000,
            height=800,
            margin=dict(l=20, r=20, t=40, b=20)
        )

        return fig

    def _get_feature_values(self, feature, indices):
        feature_idx = self._get_feature_index(feature)
        return self.data_array[indices + self.sequence_length - 1, feature_idx]

    def _prepare_hover_data(self, hover_features, timestamps, indices):
        hover_text = []

        if hover_features is None:
            hover_features = []

        for i, ts in enumerate(timestamps):
            text_parts = [f"Time: {ts.strftime('%Y-%m-%d %H:%M:%S')}"]

            for feature in hover_features:
                feature_idx = self._get_feature_index(feature)
                value = self.data_array[indices[i] + self.sequence_length - 1, feature_idx]
                text_parts.append(f"{feature}: {value:.2f}")

            hover_text.append("<br>".join(text_parts))

        return hover_text

    def _get_feature_index(self, feature):
        if feature in self.price_columns:
            return self.price_columns.index(feature)
        elif feature in self.indicator_columns:
            return len(self.price_columns) + self.indicator_columns.index(feature)
        else:
            raise ValueError(f"Feature '{feature}' not found in price or indicator columns.")
