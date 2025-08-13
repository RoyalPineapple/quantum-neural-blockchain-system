import pytest
import numpy as np
import torch
from typing import Dict, Any, Generator, List
from pathlib import Path

from src.quantum.core.quantum_register import QuantumRegister
from src.neural.core.quantum_neural_layer import QuantumNeuralLayer, QuantumNeuralConfig
from src.blockchain.core.blockchain import QuantumBlockchain
from src.applications.financial.core.financial_system import QuantumFinancialSystem, QuantumFinanceConfig
from src.optimization.core.circuit_optimizer import QuantumCircuitOptimizer, CircuitOptimizationConfig

@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Get test data directory."""
    return Path(__file__).parent / "test_data"

@pytest.fixture(scope="session")
def quantum_register() -> QuantumRegister:
    """Create quantum register for testing."""
    return QuantumRegister(n_qubits=4)

@pytest.fixture(scope="session")
def quantum_neural_config() -> QuantumNeuralConfig:
    """Create quantum neural network configuration."""
    return QuantumNeuralConfig(
        n_qubits=4,
        n_quantum_layers=2,
        n_classical_layers=2,
        learning_rate=0.01,
        quantum_circuit_depth=3
    )

@pytest.fixture(scope="session")
def quantum_neural_layer(quantum_neural_config: QuantumNeuralConfig) -> QuantumNeuralLayer:
    """Create quantum neural layer for testing."""
    return QuantumNeuralLayer(quantum_neural_config)

@pytest.fixture(scope="session")
def quantum_blockchain() -> QuantumBlockchain:
    """Create quantum blockchain for testing."""
    return QuantumBlockchain(difficulty=2)

@pytest.fixture(scope="session")
def quantum_finance_config() -> QuantumFinanceConfig:
    """Create quantum finance configuration."""
    return QuantumFinanceConfig(
        n_assets=3,
        n_qubits_per_asset=2,
        n_quantum_layers=2,
        risk_tolerance=0.1,
        trading_frequency=1.0,
        portfolio_rebalance_period=300,
        market_data_resolution='1m',
        initial_capital=1000000.0,
        transaction_cost=0.001,
        slippage_model='linear'
    )

@pytest.fixture(scope="session")
def quantum_financial_system(quantum_finance_config: QuantumFinanceConfig) -> QuantumFinancialSystem:
    """Create quantum financial system for testing."""
    return QuantumFinancialSystem(quantum_finance_config)

@pytest.fixture(scope="session")
def circuit_optimization_config() -> CircuitOptimizationConfig:
    """Create circuit optimization configuration."""
    return CircuitOptimizationConfig(
        n_qubits=4,
        n_layers=2,
        optimization_steps=100,
        learning_rate=0.01,
        convergence_threshold=1e-6,
        max_depth=5,
        error_threshold=1e-5,
        optimization_strategy='gradient'
    )

@pytest.fixture(scope="session")
def circuit_optimizer(circuit_optimization_config: CircuitOptimizationConfig) -> QuantumCircuitOptimizer:
    """Create quantum circuit optimizer for testing."""
    return QuantumCircuitOptimizer(circuit_optimization_config)

@pytest.fixture(scope="function")
def random_quantum_state(quantum_register: QuantumRegister) -> np.ndarray:
    """Generate random quantum state for testing."""
    state = np.random.randn(2**quantum_register.n_qubits) + \
            1j * np.random.randn(2**quantum_register.n_qubits)
    state = state / np.linalg.norm(state)
    return state

@pytest.fixture(scope="function")
def random_unitary_matrix(quantum_register: QuantumRegister) -> np.ndarray:
    """Generate random unitary matrix for testing."""
    dim = 2**quantum_register.n_qubits
    matrix = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    q, r = np.linalg.qr(matrix)
    return q

@pytest.fixture(scope="function")
def sample_market_data() -> Dict[str, Any]:
    """Generate sample market data for testing."""
    return {
        'asset_0': {
            'price': 100.0,
            'volume': 1000000,
            'volatility': 0.15
        },
        'asset_1': {
            'price': 50.0,
            'volume': 500000,
            'volatility': 0.2
        },
        'asset_2': {
            'price': 75.0,
            'volume': 750000,
            'volatility': 0.18
        }
    }

@pytest.fixture(scope="function")
def sample_portfolio() -> Dict[str, Any]:
    """Generate sample portfolio for testing."""
    return {
        'cash': 500000.0,
        'positions': {
            'asset_0': 1000,
            'asset_1': 2000,
            'asset_2': 1500
        },
        'total_value': 1000000.0,
        'returns': [0.01, -0.005, 0.02, 0.015],
        'sharpe_ratio': 1.5,
        'max_drawdown': 0.05
    }

@pytest.fixture(scope="function")
def optimization_test_cases() -> List[Dict[str, Any]]:
    """Generate test cases for circuit optimization."""
    return [
        {
            'n_qubits': 2,
            'target_unitary': np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0]
            ]) / np.sqrt(2),
            'max_iterations': 100,
            'target_fidelity': 0.99
        },
        {
            'n_qubits': 3,
            'target_unitary': np.kron(
                np.array([[1, 1], [1, -1]]) / np.sqrt(2),
                np.eye(4)
            ),
            'max_iterations': 200,
            'target_fidelity': 0.95
        }
    ]

@pytest.fixture(scope="function")
def quantum_test_circuits() -> List[Dict[str, Any]]:
    """Generate quantum test circuits."""
    return [
        {
            'name': 'bell_state',
            'n_qubits': 2,
            'gates': [
                ('h', 0),
                ('cnot', 0, 1)
            ],
            'expected_state': np.array([1, 0, 0, 1]) / np.sqrt(2)
        },
        {
            'name': 'ghz_state',
            'n_qubits': 3,
            'gates': [
                ('h', 0),
                ('cnot', 0, 1),
                ('cnot', 1, 2)
            ],
            'expected_state': np.array([1, 0, 0, 0, 0, 0, 0, 1]) / np.sqrt(2)
        }
    ]

@pytest.fixture(scope="function")
def performance_benchmarks() -> Dict[str, float]:
    """Define performance benchmarks."""
    return {
        'quantum_gate_time': 0.001,  # seconds
        'neural_forward_time': 0.01,
        'blockchain_mining_time': 1.0,
        'optimization_iteration_time': 0.1,
        'financial_update_time': 0.05
    }

@pytest.fixture(scope="function")
def error_thresholds() -> Dict[str, float]:
    """Define error thresholds for testing."""
    return {
        'quantum_state_fidelity': 0.99,
        'neural_loss': 0.1,
        'optimization_convergence': 1e-6,
        'financial_prediction_error': 0.05,
        'blockchain_verification': 1e-8
    }

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers",
        "quantum: mark test as quantum computing related"
    )
    config.addinivalue_line(
        "markers",
        "neural: mark test as neural network related"
    )
    config.addinivalue_line(
        "markers",
        "blockchain: mark test as blockchain related"
    )
    config.addinivalue_line(
        "markers",
        "financial: mark test as financial system related"
    )
    config.addinivalue_line(
        "markers",
        "optimization: mark test as circuit optimization related"
    )
    config.addinivalue_line(
        "markers",
        "performance: mark test as performance related"
    )
    config.addinivalue_line(
        "markers",
        "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers",
        "system: mark test as system test"
    )

@pytest.fixture(autouse=True)
def run_around_tests():
    """Setup and teardown for each test."""
    # Setup
    torch.manual_seed(42)
    np.random.seed(42)
    
    yield
    
    # Teardown
    torch.cuda.empty_cache()  # Clear GPU memory if used
