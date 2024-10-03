# models/gmn/gmn.py

import networkx as nx
import pandas as pd
import numpy as np

class GraphMetanetwork:
    def __init__(self):
        self.graph = nx.DiGraph()

    def initialize_nodes(self, time_frames, indicators):
        # Create nodes for time frames and indicators
        for tf in time_frames:
            self.graph.add_node(tf, type='time_frame')
        for ind in indicators:
            self.graph.add_node(ind, type='indicator')

    def update_graph(self, market_data):
        # Update nodes and edges with new market data
        # market_data is a dict with keys as indicators and values as dataframes
        for ind, df in market_data.items():
            # Update node attributes
            self.graph.nodes[ind]['data'] = df

        # Update edges based on correlations or other relationships
        self.calculate_correlations(market_data)

    def calculate_correlations(self, market_data):
        # Example: calculate correlations between indicators
        indicators = list(market_data.keys())
        for i in range(len(indicators)):
            for j in range(i+1, len(indicators)):
                ind1 = indicators[i]
                ind2 = indicators[j]
                corr = market_data[ind1]['value'].corr(market_data[ind2]['value'])
                self.graph.add_edge(ind1, ind2, weight=corr)

    def get_subgraph(self, nodes):
        return self.graph.subgraph(nodes)

    def visualize_graph(self):
        # Optional: Code to visualize the graph
        pass
