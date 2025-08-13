import pytest
import numpy as np
import torch
from typing import Dict, Any, List
from datetime import datetime

from src.quantum.core.quantum_register import QuantumRegister
from src.neural.core.quantum_neural_layer import QuantumNeuralLayer, QuantumNeuralConfig
from src.blockchain.core.blockchain import QuantumBlockchain, Transaction
from src.optimization.core.circuit_optimizer import QuantumCircuitOptimizer
from src.applications.financial.core.financial_system import QuantumFinancialSystem

@pytest.mark.integration
class TestSystemIntegration:
    """Test integration between system components."""
    
    def test_quantum_neural_integration(self, quantum_register: QuantumRegister,
                                      quantum_neural_layer: QuantumNeuralLayer):
        """Test integration between quantum and neural components."""
        batch_size = 2
        input_size = 2**quantum_register.n_qubits
        
        # Create quantum state
        quantum_state = quantum_register.measure()
        
        # Process through neural network
        input_tensor = torch.from_numpy(quantum_state).unsqueeze(0).repeat(batch_size, 1)
        output = quantum_neural_layer(input_tensor)
        
        # Verify output
        assert output.shape == (batch_size, input_size)
        assert not torch.isnan(output).any()
        
        # Verify quantum state preservation
        output_states = output.detach().numpy()
        for state in output_states:
            assert np.abs(np.sum(np.abs(state)**2) - 1.0) < 1e-6
            
    def test_quantum_blockchain_integration(self, quantum_register: QuantumRegister,
                                         quantum_blockchain: QuantumBlockchain):
        """Test integration between quantum and blockchain components."""
        # Create quantum-signed transaction
        transaction = Transaction(
            sender="alice",
            receiver="bob",
            amount=100.0,
            timestamp=datetime.now().timestamp()
        )
        
        # Use quantum state for signature
        quantum_state = quantum_register.measure()
        transaction.quantum_signature = self._create_test_signature(quantum_state)
        
        # Add to blockchain
        success = quantum_blockchain.add_transaction(transaction)
        assert success
        
        # Mine block with quantum state
        block = quantum_blockchain.mine_pending_transactions("miner_1")
        assert block is not None
        assert block.quantum_state is not None
        
        # Verify quantum state integrity
        assert np.abs(np.sum(np.abs(block.quantum_state)**2) - 1.0) < 1e-6
        
    def test_neural_blockchain_integration(self, quantum_neural_layer: QuantumNeuralLayer,
                                        quantum_blockchain: QuantumBlockchain):
        """Test integration between neural and blockchain components."""
        # Create and process transaction data
        transaction_data = self._create_test_transaction_data(5)
        processed_data = quantum_neural_layer(transaction_data)
        
        # Use processed data in blockchain
        for i in range(len(processed_data)):
            transaction = Transaction(
                sender=f"sender_{i}",
                receiver=f"receiver_{i}",
                amount=float(processed_data[i].sum()),
                timestamp=datetime.now().timestamp()
            )
            transaction.quantum_signature = self._create_test_signature(
                processed_data[i].detach().numpy()
            )
            quantum_blockchain.add_transaction(transaction)
            
        # Mine block
        block = quantum_blockchain.mine_pending_transactions("miner_1")
        assert block is not None
        assert len(block.transactions) == len(processed_data) + 1  # +1 for mining reward
        
    def test_quantum_optimization_integration(self, quantum_register: QuantumRegister,
                                           circuit_optimizer: QuantumCircuitOptimizer):
        """Test integration between quantum and optimization components."""
        # Create target quantum state
        target_state = quantum_register.measure()
        
        # Define optimization task
        def cost_function(state: np.ndarray) -> float:
            return np.sum(np.abs(state - target_state)**2)
            
        # Optimize quantum circuit
        results = circuit_optimizer.optimize_ansatz(cost_function)
        
        # Verify results
        assert results['final_cost'] < 0.1
        assert 'optimized_parameters' in results
        assert 'optimization_steps' in results
        
    def test_full_system_integration(self, quantum_financial_system: QuantumFinancialSystem,
                                   sample_market_data: Dict[str, Any]):
        """Test integration of all system components."""
        # Process market data
        results = quantum_financial_system.update(sample_market_data)
        
        # Verify results
        assert 'portfolio' in results
        assert 'predictions' in results
        assert 'risk_metrics' in results
        assert 'trading_signals' in results
        
        # Verify quantum state consistency
        portfolio_state = quantum_financial_system._encode_market_data(sample_market_data)
        assert isinstance(portfolio_state, torch.Tensor)
        assert not torch.isnan(portfolio_state).any()
        
        # Verify blockchain integration
        if results['trading_signals']:
            # Check if trades were recorded on blockchain
            latest_block = quantum_financial_system.blockchain.get_latest_block()
            assert latest_block is not None
            assert len(latest_block.transactions) > 0
            
    def test_error_propagation(self, quantum_financial_system: QuantumFinancialSystem):
        """Test error handling and propagation between components."""
        # Test invalid market data
        invalid_data = {'invalid_asset': {'price': 'invalid'}}
        with pytest.raises(ValueError):
            quantum_financial_system.update(invalid_data)
            
        # Test invalid quantum state
        with pytest.raises(ValueError):
            quantum_financial_system._encode_market_data({})
            
        # Test invalid transaction
        with pytest.raises(ValueError):
            quantum_financial_system._execute_trades({
                'invalid_trade': {'action': 'invalid'}
            })
            
    def test_performance_integration(self, quantum_financial_system: QuantumFinancialSystem,
                                  sample_market_data: Dict[str, Any],
                                  performance_benchmarks: Dict[str, float]):
        """Test performance integration between components."""
        # Measure update performance
        start_time = datetime.now()
        results = quantum_financial_system.update(sample_market_data)
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Verify performance meets benchmarks
        assert execution_time < performance_benchmarks['financial_update_time']
        
        # Verify quantum operations performance
        quantum_start = datetime.now()
        quantum_state = quantum_financial_system._encode_market_data(sample_market_data)
        quantum_time = (datetime.now() - quantum_start).total_seconds()
        assert quantum_time < performance_benchmarks['quantum_gate_time']
        
    def test_state_consistency(self, quantum_financial_system: QuantumFinancialSystem,
                             sample_market_data: Dict[str, Any],
                             error_thresholds: Dict[str, float]):
        """Test state consistency across components."""
        # Initial state
        initial_state = quantum_financial_system.get_portfolio_state()
        
        # Update system
        results = quantum_financial_system.update(sample_market_data)
        
        # Final state
        final_state = quantum_financial_system.get_portfolio_state()
        
        # Verify quantum state consistency
        quantum_state1 = quantum_financial_system._encode_market_data(sample_market_data)
        quantum_state2 = quantum_financial_system._encode_market_data(sample_market_data)
        quantum_diff = torch.norm(quantum_state1 - quantum_state2)
        assert quantum_diff < error_thresholds['quantum_state_fidelity']
        
        # Verify blockchain consistency
        assert quantum_financial_system.blockchain.verify_chain()
        
    @staticmethod
    def _create_test_transaction_data(batch_size: int) -> torch.Tensor:
        """Create test transaction data."""
        return torch.randn(batch_size, 16)  # 16 = 2^4 (4 qubits)
        
    @staticmethod
    def _create_test_signature(state: np.ndarray) -> Any:
        """Create test quantum signature."""
        from src.blockchain.utils.cryptography import QuantumSignature
        
        return QuantumSignature(
            signature_state=state,
            verification_state=state.copy(),
            classical_hash="test_hash"
        )
