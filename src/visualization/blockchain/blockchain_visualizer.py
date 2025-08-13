import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import seaborn as sns

class BlockchainVisualizer:
    """
    Visualize blockchain structure and metrics.
    
    Features:
    - Blockchain network visualization
    - Transaction flow analysis
    - Block structure visualization
    - Quantum signature verification
    - Network metrics visualization
    - Interactive blockchain explorer
    """
    
    def __init__(self, save_dir: Optional[str] = None):
        """
        Initialize visualizer.
        
        Args:
            save_dir: Directory to save visualizations
        """
        self.save_dir = Path(save_dir) if save_dir else None
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            
        # Set style
        plt.style.use('seaborn')
        
    def plot_blockchain_network(self,
                              blocks: List[Dict[str, Any]],
                              title: str = "Blockchain Network",
                              save: bool = False) -> None:
        """
        Plot blockchain network structure.
        
        Args:
            blocks: List of blocks
            title: Plot title
            save: Whether to save plot
        """
        # Create graph
        G = nx.DiGraph()
        
        # Add nodes and edges
        for block in blocks:
            G.add_node(block['hash'],
                      timestamp=block['timestamp'],
                      n_transactions=len(block['transactions']))
                      
            if block['previous_hash'] != "0":  # Not genesis block
                G.add_edge(block['previous_hash'], block['hash'])
                
        # Create figure
        plt.figure(figsize=(15, 10))
        
        # Draw network
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True,
               node_color='lightblue',
               node_size=[G.nodes[node]['n_transactions']*100 for node in G.nodes],
               arrowsize=20)
               
        plt.title(title)
        
        if save and self.save_dir:
            plt.savefig(self.save_dir / f"{title.lower().replace(' ', '_')}.png")
        else:
            plt.show()
            
    def plot_transaction_flow(self,
                            transactions: List[Dict[str, Any]],
                            title: str = "Transaction Flow",
                            save: bool = False) -> None:
        """
        Plot transaction flow network.
        
        Args:
            transactions: List of transactions
            title: Plot title
            save: Whether to save plot
        """
        # Create graph
        G = nx.DiGraph()
        
        # Add nodes and edges
        for tx in transactions:
            G.add_edge(tx['sender'], tx['receiver'],
                      weight=tx['amount'])
                      
        # Create figure
        plt.figure(figsize=(15, 10))
        
        # Draw network
        pos = nx.spring_layout(G)
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        
        nx.draw(G, pos, with_labels=True,
               node_color='lightgreen',
               width=np.array(weights)/max(weights)*5,
               edge_color='gray',
               arrowsize=20)
               
        plt.title(title)
        
        if save and self.save_dir:
            plt.savefig(self.save_dir / f"{title.lower().replace(' ', '_')}.png")
        else:
            plt.show()
            
    def plot_block_structure(self,
                           block: Dict[str, Any],
                           title: str = "Block Structure",
                           save: bool = False) -> None:
        """
        Plot block internal structure.
        
        Args:
            block: Block data
            title: Plot title
            save: Whether to save plot
        """
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot transaction distribution
        amounts = [tx['amount'] for tx in block['transactions']]
        sns.histplot(amounts, ax=ax1)
        ax1.set_title("Transaction Amounts")
        
        # Plot quantum state
        if block['quantum_state'] is not None:
            quantum_state = np.array(block['quantum_state'])
            sns.heatmap(np.outer(quantum_state, quantum_state.conj()),
                       ax=ax2)
            ax2.set_title("Quantum State")
            
        plt.suptitle(title)
        
        if save and self.save_dir:
            plt.savefig(self.save_dir / f"{title.lower().replace(' ', '_')}.png")
        else:
            plt.show()
            
    def plot_signature_verification(self,
                                  signatures: List[Dict[str, Any]],
                                  title: str = "Quantum Signatures",
                                  save: bool = False) -> None:
        """
        Plot quantum signature verification results.
        
        Args:
            signatures: List of quantum signatures
            title: Plot title
            save: Whether to save plot
        """
        # Create figure
        plt.figure(figsize=(15, 5))
        
        # Plot signature states
        for i, sig in enumerate(signatures):
            plt.subplot(1, len(signatures), i+1)
            state = np.array(sig['signature_state'])
            plt.bar(range(len(state)), np.abs(state))
            plt.title(f"Signature {i+1}")
            
        plt.suptitle(title)
        
        if save and self.save_dir:
            plt.savefig(self.save_dir / f"{title.lower().replace(' ', '_')}.png")
        else:
            plt.show()
            
    def plot_network_metrics(self,
                           metrics: Dict[str, List[float]],
                           title: str = "Network Metrics",
                           save: bool = False) -> None:
        """
        Plot blockchain network metrics.
        
        Args:
            metrics: Dictionary of metrics
            title: Plot title
            save: Whether to save plot
        """
        n_metrics = len(metrics)
        fig, axes = plt.subplots(n_metrics, 1, figsize=(15, 5*n_metrics))
        
        if n_metrics == 1:
            axes = [axes]
            
        for ax, (metric_name, values) in zip(axes, metrics.items()):
            ax.plot(values)
            ax.set_title(f"{metric_name} History")
            ax.set_xlabel("Block Number")
            ax.set_ylabel(metric_name)
            
        plt.tight_layout()
        plt.suptitle(title)
        
        if save and self.save_dir:
            plt.savefig(self.save_dir / f"{title.lower().replace(' ', '_')}.png")
        else:
            plt.show()
            
    def create_interactive_explorer(self,
                                  blocks: List[Dict[str, Any]],
                                  title: str = "Blockchain Explorer") -> None:
        """
        Create interactive blockchain explorer.
        
        Args:
            blocks: List of blocks
            title: Plot title
        """
        # Create nodes and edges
        nodes = []
        edges = []
        
        for block in blocks:
            nodes.append({
                'id': block['hash'],
                'label': f"Block {block['index']}",
                'size': len(block['transactions'])*10,
                'title': json.dumps(block, indent=2)
            })
            
            if block['previous_hash'] != "0":
                edges.append({
                    'from': block['previous_hash'],
                    'to': block['hash']
                })
                
        # Create node trace
        node_trace = go.Scatter(
            x=[],
            y=[],
            text=[],
            mode='markers+text',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                size=[node['size'] for node in nodes]
            )
        )
        
        # Create edge trace
        edge_trace = go.Scatter(
            x=[],
            y=[],
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Create layout
        G = nx.DiGraph()
        for node in nodes:
            G.add_node(node['id'])
        for edge in edges:
            G.add_edge(edge['from'], edge['to'])
            
        pos = nx.spring_layout(G)
        
        # Add node positions
        for node in nodes:
            x, y = pos[node['id']]
            node_trace['x'] += tuple([x])
            node_trace['y'] += tuple([y])
            node_trace['text'] += tuple([node['label']])
            
        # Add edge positions
        for edge in edges:
            x0, y0 = pos[edge['from']]
            x1, y1 = pos[edge['to']]
            edge_trace['x'] += tuple([x0, x1, None])
            edge_trace['y'] += tuple([y0, y1, None])
            
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title=title,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           xaxis=dict(showgrid=False, zeroline=False),
                           yaxis=dict(showgrid=False, zeroline=False)
                       ))
                       
        fig.show()
        
    def animate_blockchain_growth(self,
                                blocks: List[Dict[str, Any]],
                                interval: int = 1000,
                                title: str = "Blockchain Growth",
                                save: bool = False) -> None:
        """
        Create animation of blockchain growth.
        
        Args:
            blocks: List of blocks
            interval: Animation interval in milliseconds
            title: Animation title
            save: Whether to save plot
        """
        import matplotlib.animation as animation
        
        # Create figure
        fig, ax = plt.subplots(figsize=(15, 10))
        
        G = nx.DiGraph()
        
        def update(frame):
            ax.clear()
            
            # Add nodes and edges up to current frame
            current_blocks = blocks[:frame+1]
            for block in current_blocks:
                G.add_node(block['hash'],
                          timestamp=block['timestamp'],
                          n_transactions=len(block['transactions']))
                          
                if block['previous_hash'] != "0":
                    G.add_edge(block['previous_hash'], block['hash'])
                    
            # Draw network
            pos = nx.spring_layout(G)
            nx.draw(G, pos, ax=ax, with_labels=True,
                   node_color='lightblue',
                   node_size=[G.nodes[node]['n_transactions']*100
                            for node in G.nodes],
                   arrowsize=20)
                   
            ax.set_title(f"{title} - Block {frame+1}")
            
        # Create animation
        anim = animation.FuncAnimation(
            fig, update,
            frames=len(blocks),
            interval=interval
        )
        
        if save and self.save_dir:
            anim.save(self.save_dir / f"{title.lower().replace(' ', '_')}.gif")
        else:
            plt.show()
            
    def plot_quantum_consensus(self,
                             consensus_data: List[Dict[str, Any]],
                             title: str = "Quantum Consensus",
                             save: bool = False) -> None:
        """
        Plot quantum consensus metrics.
        
        Args:
            consensus_data: List of consensus data
            title: Plot title
            save: Whether to save plot
        """
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot consensus states
        states = np.array([data['quantum_state'] for data in consensus_data])
        sns.heatmap(np.abs(states), ax=ax1)
        ax1.set_title("Consensus States")
        ax1.set_xlabel("Qubit")
        ax1.set_ylabel("Block")
        
        # Plot consensus metrics
        metrics = [data['consensus_metric'] for data in consensus_data]
        ax2.plot(metrics)
        ax2.set_title("Consensus Metric")
        ax2.set_xlabel("Block")
        ax2.set_ylabel("Metric Value")
        
        plt.suptitle(title)
        
        if save and self.save_dir:
            plt.savefig(self.save_dir / f"{title.lower().replace(' ', '_')}.png")
        else:
            plt.show()
            
    def plot_transaction_patterns(self,
                                transactions: List[Dict[str, Any]],
                                title: str = "Transaction Patterns",
                                save: bool = False) -> None:
        """
        Plot transaction patterns and statistics.
        
        Args:
            transactions: List of transactions
            title: Plot title
            save: Whether to save plot
        """
        # Create figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot transaction amounts
        amounts = [tx['amount'] for tx in transactions]
        sns.histplot(amounts, ax=ax1)
        ax1.set_title("Transaction Amounts")
        
        # Plot transaction times
        times = [datetime.fromtimestamp(tx['timestamp'])
                for tx in transactions]
        ax2.hist([t.hour for t in times], bins=24)
        ax2.set_title("Transaction Times")
        ax2.set_xlabel("Hour")
        
        # Plot sender-receiver heatmap
        senders = list(set(tx['sender'] for tx in transactions))
        receivers = list(set(tx['receiver'] for tx in transactions))
        matrix = np.zeros((len(senders), len(receivers)))
        
        for tx in transactions:
            i = senders.index(tx['sender'])
            j = receivers.index(tx['receiver'])
            matrix[i, j] += 1
            
        sns.heatmap(matrix, ax=ax3)
        ax3.set_title("Sender-Receiver Matrix")
        
        # Plot quantum signatures
        if 'quantum_signature' in transactions[0]:
            signatures = [np.array(tx['quantum_signature']['signature_state'])
                        for tx in transactions]
            avg_signature = np.mean(signatures, axis=0)
            ax4.bar(range(len(avg_signature)), np.abs(avg_signature))
            ax4.set_title("Average Quantum Signature")
            
        plt.suptitle(title)
        
        if save and self.save_dir:
            plt.savefig(self.save_dir / f"{title.lower().replace(' ', '_')}.png")
        else:
            plt.show()
