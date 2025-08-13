import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional, Tuple
import torch
from pathlib import Path
import seaborn as sns

class QuantumStateVisualizer:
    """
    Visualize quantum states and operations.
    
    Features:
    - Bloch sphere visualization
    - State vector plots
    - Density matrix heatmaps
    - Quantum circuit diagrams
    - Entanglement visualization
    - Interactive 3D plots
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
        
    def plot_bloch_sphere(self,
                         quantum_state: np.ndarray,
                         title: str = "Quantum State Bloch Sphere",
                         save: bool = False) -> None:
        """
        Plot quantum state on Bloch sphere.
        
        Args:
            quantum_state: Quantum state vector
            title: Plot title
            save: Whether to save plot
        """
        # Convert to density matrix
        rho = self._state_to_density_matrix(quantum_state)
        
        # Calculate Bloch sphere coordinates
        x = np.real(rho[0, 1] + rho[1, 0])
        y = np.imag(rho[1, 0] - rho[0, 1])
        z = rho[0, 0] - rho[1, 1]
        
        # Create figure
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Draw Bloch sphere
        self._draw_bloch_sphere(ax)
        
        # Plot state vector
        ax.quiver(0, 0, 0, x, y, z, color='r', arrow_length_ratio=0.1)
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        
        if save and self.save_dir:
            plt.savefig(self.save_dir / f"{title.lower().replace(' ', '_')}.png")
        else:
            plt.show()
            
    def plot_state_vector(self,
                         quantum_state: np.ndarray,
                         title: str = "Quantum State Vector",
                         save: bool = False) -> None:
        """
        Plot quantum state vector.
        
        Args:
            quantum_state: Quantum state vector
            title: Plot title
            save: Whether to save plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot amplitudes
        amplitudes = np.abs(quantum_state)
        ax1.bar(range(len(amplitudes)), amplitudes)
        ax1.set_title("State Amplitudes")
        ax1.set_xlabel("Basis State")
        ax1.set_ylabel("Amplitude")
        
        # Plot phases
        phases = np.angle(quantum_state)
        ax2.bar(range(len(phases)), phases)
        ax2.set_title("State Phases")
        ax2.set_xlabel("Basis State")
        ax2.set_ylabel("Phase")
        
        plt.suptitle(title)
        
        if save and self.save_dir:
            plt.savefig(self.save_dir / f"{title.lower().replace(' ', '_')}.png")
        else:
            plt.show()
            
    def plot_density_matrix(self,
                          quantum_state: np.ndarray,
                          title: str = "Density Matrix",
                          save: bool = False) -> None:
        """
        Plot density matrix heatmap.
        
        Args:
            quantum_state: Quantum state vector
            title: Plot title
            save: Whether to save plot
        """
        # Calculate density matrix
        rho = self._state_to_density_matrix(quantum_state)
        
        # Create figure
        plt.figure(figsize=(10, 10))
        
        # Plot real part
        plt.subplot(121)
        sns.heatmap(np.real(rho), annot=True, cmap='RdBu_r')
        plt.title("Real Part")
        
        # Plot imaginary part
        plt.subplot(122)
        sns.heatmap(np.imag(rho), annot=True, cmap='RdBu_r')
        plt.title("Imaginary Part")
        
        plt.suptitle(title)
        
        if save and self.save_dir:
            plt.savefig(self.save_dir / f"{title.lower().replace(' ', '_')}.png")
        else:
            plt.show()
            
    def plot_circuit(self,
                    gates: List[Tuple[str, int]],
                    n_qubits: int,
                    title: str = "Quantum Circuit",
                    save: bool = False) -> None:
        """
        Plot quantum circuit diagram.
        
        Args:
            gates: List of (gate_name, qubit) tuples
            n_qubits: Number of qubits
            title: Plot title
            save: Whether to save plot
        """
        fig, ax = plt.subplots(figsize=(15, n_qubits))
        
        # Draw qubit lines
        for i in range(n_qubits):
            ax.hlines(y=i, xmin=0, xmax=len(gates), color='black')
            ax.text(-0.5, i, f'q{i}|0⟩', ha='right', va='center')
            
        # Draw gates
        for i, (gate, qubit) in enumerate(gates):
            self._draw_gate(ax, i, qubit, gate)
            
        # Set layout
        ax.set_xlim(-1, len(gates))
        ax.set_ylim(-0.5, n_qubits-0.5)
        ax.axis('off')
        ax.set_title(title)
        
        if save and self.save_dir:
            plt.savefig(self.save_dir / f"{title.lower().replace(' ', '_')}.png")
        else:
            plt.show()
            
    def plot_entanglement(self,
                         quantum_state: np.ndarray,
                         title: str = "Quantum Entanglement",
                         save: bool = False) -> None:
        """
        Plot quantum entanglement visualization.
        
        Args:
            quantum_state: Quantum state vector
            title: Plot title
            save: Whether to save plot
        """
        # Calculate entanglement
        n_qubits = int(np.log2(len(quantum_state)))
        entanglement = self._calculate_entanglement_matrix(quantum_state, n_qubits)
        
        # Create figure
        plt.figure(figsize=(10, 10))
        
        # Plot entanglement matrix
        sns.heatmap(entanglement, annot=True, cmap='viridis')
        plt.title(title)
        plt.xlabel("Qubit")
        plt.ylabel("Qubit")
        
        if save and self.save_dir:
            plt.savefig(self.save_dir / f"{title.lower().replace(' ', '_')}.png")
        else:
            plt.show()
            
    def create_interactive_bloch(self,
                               quantum_state: np.ndarray,
                               title: str = "Interactive Bloch Sphere") -> None:
        """
        Create interactive 3D Bloch sphere visualization.
        
        Args:
            quantum_state: Quantum state vector
            title: Plot title
        """
        # Calculate Bloch sphere coordinates
        rho = self._state_to_density_matrix(quantum_state)
        x = np.real(rho[0, 1] + rho[1, 0])
        y = np.imag(rho[1, 0] - rho[0, 1])
        z = rho[0, 0] - rho[1, 1]
        
        # Create sphere
        phi = np.linspace(0, 2*np.pi, 100)
        theta = np.linspace(0, np.pi, 100)
        phi, theta = np.meshgrid(phi, theta)
        
        x_sphere = np.sin(theta) * np.cos(phi)
        y_sphere = np.sin(theta) * np.sin(phi)
        z_sphere = np.cos(theta)
        
        # Create figure
        fig = go.Figure()
        
        # Add sphere surface
        fig.add_surface(x=x_sphere, y=y_sphere, z=z_sphere,
                       opacity=0.3, showscale=False)
        
        # Add state vector
        fig.add_scatter3d(x=[0, x], y=[0, y], z=[0, z],
                         mode='lines+markers',
                         line=dict(color='red', width=5),
                         marker=dict(size=5))
        
        # Update layout
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            )
        )
        
        # Show figure
        fig.show()
        
    def animate_evolution(self,
                        states: List[np.ndarray],
                        interval: int = 100,
                        title: str = "Quantum State Evolution",
                        save: bool = False) -> None:
        """
        Create animation of quantum state evolution.
        
        Args:
            states: List of quantum states
            interval: Animation interval in milliseconds
            title: Animation title
            save: Whether to save animation
        """
        import matplotlib.animation as animation
        
        # Create figure
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Draw Bloch sphere
        self._draw_bloch_sphere(ax)
        
        # Initialize state vector
        quiver = ax.quiver(0, 0, 0, 0, 0, 0, color='r')
        
        def update(frame):
            # Calculate new coordinates
            rho = self._state_to_density_matrix(states[frame])
            x = np.real(rho[0, 1] + rho[1, 0])
            y = np.imag(rho[1, 0] - rho[0, 1])
            z = rho[0, 0] - rho[1, 1]
            
            # Update vector
            quiver.set_segments([[[0, 0, 0], [x, y, z]]])
            return quiver,
            
        # Create animation
        anim = animation.FuncAnimation(
            fig, update, frames=len(states),
            interval=interval, blit=True
        )
        
        if save and self.save_dir:
            anim.save(self.save_dir / f"{title.lower().replace(' ', '_')}.gif")
        else:
            plt.show()
            
    def _state_to_density_matrix(self, state: np.ndarray) -> np.ndarray:
        """Convert state vector to density matrix."""
        return np.outer(state, state.conj())
        
    def _draw_bloch_sphere(self, ax: plt.Axes):
        """Draw Bloch sphere wireframe."""
        # Create sphere
        phi = np.linspace(0, 2*np.pi, 100)
        theta = np.linspace(0, np.pi, 100)
        phi, theta = np.meshgrid(phi, theta)
        
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        
        # Plot sphere
        ax.plot_surface(x, y, z, alpha=0.1)
        
        # Plot coordinate axes
        ax.quiver(0, 0, 0, 1, 0, 0, color='k', alpha=0.5)
        ax.quiver(0, 0, 0, 0, 1, 0, color='k', alpha=0.5)
        ax.quiver(0, 0, 0, 0, 0, 1, color='k', alpha=0.5)
        
    def _draw_gate(self, ax: plt.Axes, x: int, y: int, gate: str):
        """Draw quantum gate symbol."""
        # Gate symbols
        symbols = {
            'H': 'H',
            'X': 'X',
            'Y': 'Y',
            'Z': 'Z',
            'CNOT': '•'
        }
        
        # Draw gate box
        rect = plt.Rectangle((x-0.3, y-0.3), 0.6, 0.6,
                           fill=True, color='white',
                           ec='black')
        ax.add_patch(rect)
        
        # Add gate symbol
        symbol = symbols.get(gate, gate)
        ax.text(x, y, symbol, ha='center', va='center')
        
    def _calculate_entanglement_matrix(self,
                                     state: np.ndarray,
                                     n_qubits: int) -> np.ndarray:
        """Calculate pairwise entanglement between qubits."""
        entanglement = np.zeros((n_qubits, n_qubits))
        
        # Reshape state
        state_matrix = state.reshape([2] * n_qubits)
        
        # Calculate pairwise entanglement
        for i in range(n_qubits):
            for j in range(i+1, n_qubits):
                # Trace out other qubits
                reduced = self._partial_trace(state_matrix, [i, j])
                
                # Calculate von Neumann entropy
                eigenvalues = np.linalg.eigvalsh(reduced)
                entropy = -np.sum(eigenvalues * np.log2(eigenvalues + 1e-10))
                
                entanglement[i, j] = entropy
                entanglement[j, i] = entropy
                
        return entanglement
        
    def _partial_trace(self, state: np.ndarray,
                      keep_qubits: List[int]) -> np.ndarray:
        """Calculate partial trace over system."""
        n_qubits = len(state.shape)
        trace_qubits = [i for i in range(n_qubits) if i not in keep_qubits]
        
        # Reshape state
        shape = [2] * n_qubits
        reshaped = state.reshape(shape)
        
        # Trace out qubits
        for qubit in reversed(trace_qubits):
            reshaped = np.trace(reshaped, axis1=qubit, axis2=qubit+1)
            
        return reshaped
