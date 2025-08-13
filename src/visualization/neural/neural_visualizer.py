import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional, Tuple
import plotly.graph_objects as go
from pathlib import Path
import networkx as nx

class NeuralVisualizer:
    """
    Visualize quantum-neural network components and training.
    
    Features:
    - Network architecture visualization
    - Training metrics plots
    - Quantum-classical parameter visualization
    - Layer activation analysis
    - Gradient flow visualization
    - Interactive network exploration
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
        
    def plot_network_architecture(self,
                                model: torch.nn.Module,
                                title: str = "Network Architecture",
                                save: bool = False) -> None:
        """
        Plot neural network architecture.
        
        Args:
            model: Neural network model
            title: Plot title
            save: Whether to save plot
        """
        # Create graph
        G = nx.DiGraph()
        
        # Add nodes and edges
        layer_nodes = []
        current_pos = 0
        
        for name, module in model.named_modules():
            if len(name.split('.')) == 1:  # Only main layers
                G.add_node(name, pos=(current_pos, 0))
                layer_nodes.append(name)
                current_pos += 1
                
        # Add edges between layers
        for i in range(len(layer_nodes)-1):
            G.add_edge(layer_nodes[i], layer_nodes[i+1])
            
        # Create figure
        plt.figure(figsize=(15, 10))
        
        # Draw network
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='lightblue',
               node_size=2000, arrowsize=20)
        
        plt.title(title)
        
        if save and self.save_dir:
            plt.savefig(self.save_dir / f"{title.lower().replace(' ', '_')}.png")
        else:
            plt.show()
            
    def plot_training_history(self,
                            history: Dict[str, List[float]],
                            title: str = "Training History",
                            save: bool = False) -> None:
        """
        Plot training metrics history.
        
        Args:
            history: Training history dictionary
            title: Plot title
            save: Whether to save plot
        """
        n_metrics = len(history)
        fig, axes = plt.subplots(n_metrics, 1, figsize=(15, 5*n_metrics))
        
        if n_metrics == 1:
            axes = [axes]
            
        for ax, (metric_name, values) in zip(axes, history.items()):
            ax.plot(values)
            ax.set_title(f"{metric_name} History")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(metric_name)
            
        plt.tight_layout()
        plt.suptitle(title)
        
        if save and self.save_dir:
            plt.savefig(self.save_dir / f"{title.lower().replace(' ', '_')}.png")
        else:
            plt.show()
            
    def plot_parameter_distribution(self,
                                  model: torch.nn.Module,
                                  title: str = "Parameter Distribution",
                                  save: bool = False) -> None:
        """
        Plot model parameter distributions.
        
        Args:
            model: Neural network model
            title: Plot title
            save: Whether to save plot
        """
        # Collect parameters
        classical_params = []
        quantum_params = []
        
        for name, param in model.named_parameters():
            if 'quantum' in name:
                quantum_params.extend(param.detach().numpy().flatten())
            else:
                classical_params.extend(param.detach().numpy().flatten())
                
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot classical parameters
        sns.histplot(classical_params, ax=ax1)
        ax1.set_title("Classical Parameters")
        
        # Plot quantum parameters
        sns.histplot(quantum_params, ax=ax2)
        ax2.set_title("Quantum Parameters")
        
        plt.suptitle(title)
        
        if save and self.save_dir:
            plt.savefig(self.save_dir / f"{title.lower().replace(' ', '_')}.png")
        else:
            plt.show()
            
    def plot_layer_activations(self,
                             activations: Dict[str, torch.Tensor],
                             title: str = "Layer Activations",
                             save: bool = False) -> None:
        """
        Plot layer activation patterns.
        
        Args:
            activations: Dictionary of layer activations
            title: Plot title
            save: Whether to save plot
        """
        n_layers = len(activations)
        fig, axes = plt.subplots(n_layers, 1, figsize=(15, 5*n_layers))
        
        if n_layers == 1:
            axes = [axes]
            
        for ax, (layer_name, activation) in zip(axes, activations.items()):
            sns.heatmap(activation.detach().numpy(), ax=ax)
            ax.set_title(f"{layer_name} Activation")
            
        plt.tight_layout()
        plt.suptitle(title)
        
        if save and self.save_dir:
            plt.savefig(self.save_dir / f"{title.lower().replace(' ', '_')}.png")
        else:
            plt.show()
            
    def plot_gradient_flow(self,
                          model: torch.nn.Module,
                          title: str = "Gradient Flow",
                          save: bool = False) -> None:
        """
        Plot gradient flow through network.
        
        Args:
            model: Neural network model
            title: Plot title
            save: Whether to save plot
        """
        # Collect gradients
        gradients = []
        layer_names = []
        
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                gradients.append(param.grad.detach().numpy().flatten())
                layer_names.append(name)
                
        # Create figure
        plt.figure(figsize=(15, 5))
        
        # Plot gradients
        plt.boxplot(gradients, labels=layer_names, vert=False)
        plt.title(title)
        plt.xlabel("Gradient Value")
        
        if save and self.save_dir:
            plt.savefig(self.save_dir / f"{title.lower().replace(' ', '_')}.png")
        else:
            plt.show()
            
    def create_interactive_network(self,
                                 model: torch.nn.Module,
                                 title: str = "Interactive Network") -> None:
        """
        Create interactive network visualization.
        
        Args:
            model: Neural network model
            title: Plot title
        """
        # Create graph
        G = nx.DiGraph()
        
        # Add nodes and edges
        nodes = []
        edges = []
        current_layer = 0
        
        for name, module in model.named_modules():
            if len(name.split('.')) == 1:
                nodes.append({
                    'id': name,
                    'label': name,
                    'level': current_layer
                })
                if current_layer > 0:
                    edges.append({
                        'from': nodes[current_layer-1]['id'],
                        'to': name
                    })
                current_layer += 1
                
        # Create figure
        node_trace = go.Scatter(
            x=[],
            y=[],
            text=[],
            mode='markers+text',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                size=50
            )
        )
        
        edge_trace = go.Scatter(
            x=[],
            y=[],
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Add node positions
        pos = nx.spring_layout(G)
        
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
        
    def animate_training(self,
                        history: Dict[str, List[float]],
                        interval: int = 100,
                        title: str = "Training Animation",
                        save: bool = False) -> None:
        """
        Create animation of training progress.
        
        Args:
            history: Training history
            interval: Animation interval in milliseconds
            title: Animation title
            save: Whether to save animation
        """
        import matplotlib.animation as animation
        
        # Create figure
        fig, axes = plt.subplots(len(history), 1, figsize=(15, 5*len(history)))
        
        if len(history) == 1:
            axes = [axes]
            
        lines = []
        for ax, (metric_name, values) in zip(axes, history.items()):
            line, = ax.plot([], [])
            lines.append(line)
            ax.set_xlim(0, len(values))
            ax.set_ylim(min(values), max(values))
            ax.set_title(f"{metric_name} History")
            ax.set_xlabel("Epoch")
            ax.set_ylabel(metric_name)
            
        def update(frame):
            for line, (_, values) in zip(lines, history.items()):
                line.set_data(range(frame), values[:frame])
            return lines
            
        # Create animation
        anim = animation.FuncAnimation(
            fig, update,
            frames=len(next(iter(history.values()))),
            interval=interval, blit=True
        )
        
        if save and self.save_dir:
            anim.save(self.save_dir / f"{title.lower().replace(' ', '_')}.gif")
        else:
            plt.show()
