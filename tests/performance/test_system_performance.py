import pytest
import numpy as np
import torch
import time
from typing import Dict, Any, List, Tuple
from datetime import datetime
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from src.quantum.core.quantum_register import QuantumRegister
from src.neural.core.quantum_neural_layer import QuantumNeuralLayer
from src.blockchain.core.blockchain import QuantumBlockchain
from src.optimization.core.circuit_optimizer import QuantumCircuitOptimizer
from src.applications.financial.core.financial_system import QuantumFinancialSystem

@pytest.mark.performance
class TestSystemPerformance:
    """Test system performance and scalability."""
    
    @pytest.mark.parametrize("n_qubits", [2, 4, 6, 8, 10])
    def test_quantum_scaling(self, n_qubits: int):
        """Test quantum system scaling with number of qubits."""
        # Initialize quantum register
        quantum_register = QuantumRegister(n_qubits)
        
        # Measure gate application time
        gate_times = []
        for _ in range(100):
            start_time = time.perf_counter()
            quantum_register.apply_gate(
                np.eye(2**n_qubits),
                0
            )
            gate_times.append(time.perf_counter() - start_time)
            
        avg_gate_time = np.mean(gate_times)
        print(f"\nQuantum gate time for {n_qubits} qubits: {avg_gate_time:.6f}s")
        
        # Verify exponential scaling
        if n_qubits > 2:
            # Should scale roughly as 2^n
            expected_ratio = 4.0  # 2^2
            actual_ratio = avg_gate_time / self._get_baseline_time(n_qubits-2)
            assert actual_ratio < expected_ratio * 1.5  # Allow 50% margin
            
    @pytest.mark.parametrize("batch_size,n_qubits", [
        (16, 4),
        (32, 4),
        (64, 4),
        (128, 4)
    ])
    def test_neural_batch_scaling(self, batch_size: int, n_qubits: int):
        """Test neural network scaling with batch size."""
        # Initialize neural layer
        layer = QuantumNeuralLayer(
            QuantumNeuralConfig(
                n_qubits=n_qubits,
                n_quantum_layers=2,
                n_classical_layers=2,
                learning_rate=0.01,
                quantum_circuit_depth=3
            )
        )
        
        # Create input data
        input_size = 2**n_qubits
        x = torch.randn(batch_size, input_size)
        
        # Measure forward pass time
        forward_times = []
        for _ in range(10):
            start_time = time.perf_counter()
            with torch.no_grad():
                _ = layer(x)
            forward_times.append(time.perf_counter() - start_time)
            
        avg_forward_time = np.mean(forward_times)
        print(f"\nNeural forward time for batch size {batch_size}: {avg_forward_time:.6f}s")
        
        # Verify linear scaling with batch size
        if batch_size > 16:
            expected_ratio = batch_size / 16
            actual_ratio = avg_forward_time / self._get_baseline_neural_time()
            assert actual_ratio < expected_ratio * 1.5
            
    @pytest.mark.parametrize("n_transactions", [10, 100, 1000])
    def test_blockchain_scaling(self, n_transactions: int):
        """Test blockchain scaling with number of transactions."""
        # Initialize blockchain
        blockchain = QuantumBlockchain(difficulty=2)
        
        # Create transactions
        transactions = self._create_test_transactions(n_transactions)
        
        # Measure transaction processing time
        start_time = time.perf_counter()
        for tx in transactions:
            blockchain.add_transaction(tx)
        processing_time = time.perf_counter() - start_time
        
        print(f"\nTransaction processing time for {n_transactions} transactions: {processing_time:.6f}s")
        
        # Measure mining time
        start_time = time.perf_counter()
        blockchain.mine_pending_transactions("miner_1")
        mining_time = time.perf_counter() - start_time
        
        print(f"Mining time for {n_transactions} transactions: {mining_time:.6f}s")
        
        # Verify sub-linear scaling (due to batching)
        if n_transactions > 10:
            expected_ratio = n_transactions / 10
            actual_ratio = processing_time / self._get_baseline_blockchain_time()
            assert actual_ratio < expected_ratio  # Should be sub-linear
            
    def test_parallel_quantum_operations(self):
        """Test parallel quantum operation performance."""
        n_processes = mp.cpu_count()
        n_qubits = 4
        n_operations = 1000
        
        # Serial execution
        start_time = time.perf_counter()
        self._perform_quantum_operations(n_qubits, n_operations)
        serial_time = time.perf_counter() - start_time
        
        # Parallel execution (processes)
        start_time = time.perf_counter()
        with ProcessPoolExecutor(max_workers=n_processes) as executor:
            ops_per_process = n_operations // n_processes
            futures = [
                executor.submit(self._perform_quantum_operations, n_qubits, ops_per_process)
                for _ in range(n_processes)
            ]
            _ = [f.result() for f in futures]
        parallel_process_time = time.perf_counter() - start_time
        
        # Parallel execution (threads)
        start_time = time.perf_counter()
        with ThreadPoolExecutor(max_workers=n_processes) as executor:
            ops_per_thread = n_operations // n_processes
            futures = [
                executor.submit(self._perform_quantum_operations, n_qubits, ops_per_thread)
                for _ in range(n_processes)
            ]
            _ = [f.result() for f in futures]
        parallel_thread_time = time.perf_counter() - start_time
        
        print(f"\nSerial execution time: {serial_time:.6f}s")
        print(f"Parallel process time: {parallel_process_time:.6f}s")
        print(f"Parallel thread time: {parallel_thread_time:.6f}s")
        
        # Verify parallel speedup
        assert parallel_process_time < serial_time
        
    def test_optimization_convergence_speed(self, circuit_optimizer: QuantumCircuitOptimizer):
        """Test optimization convergence performance."""
        # Create simple optimization task
        target_unitary = np.eye(2**circuit_optimizer.config.n_qubits)
        
        # Measure optimization time and convergence
        start_time = time.perf_counter()
        results = circuit_optimizer.optimize_circuit(target_unitary)
        optimization_time = time.perf_counter() - start_time
        
        print(f"\nOptimization time: {optimization_time:.6f}s")
        print(f"Final fidelity: {results['final_fidelity']:.6f}")
        print(f"Steps to converge: {results['optimization_steps']}")
        
        # Analyze convergence rate
        fidelities = [step['metric'] for step in results['optimization_history']]
        convergence_rate = (fidelities[-1] - fidelities[0]) / len(fidelities)
        print(f"Convergence rate: {convergence_rate:.6f} per step")
        
        # Verify reasonable convergence time
        assert optimization_time < 60.0  # Should converge within 60 seconds
        assert results['final_fidelity'] > 0.9  # Should achieve high fidelity
        
    def test_financial_system_throughput(self, quantum_financial_system: QuantumFinancialSystem):
        """Test financial system transaction throughput."""
        # Generate test market data
        n_updates = 100
        market_data = self._generate_market_data_stream(n_updates)
        
        # Measure update throughput
        start_time = time.perf_counter()
        for data in market_data:
            quantum_financial_system.update(data)
        total_time = time.perf_counter() - start_time
        
        updates_per_second = n_updates / total_time
        print(f"\nMarket updates per second: {updates_per_second:.2f}")
        
        # Verify minimum throughput
        assert updates_per_second > 1.0  # Should handle at least 1 update per second
        
    def test_memory_usage(self):
        """Test system memory usage patterns."""
        import psutil
        process = psutil.Process()
        
        # Measure baseline memory
        baseline_memory = process.memory_info().rss
        
        # Create large quantum system
        n_qubits = 10
        quantum_register = QuantumRegister(n_qubits)
        
        # Measure quantum system memory
        quantum_memory = process.memory_info().rss - baseline_memory
        print(f"\nQuantum system memory usage: {quantum_memory / 1024 / 1024:.2f} MB")
        
        # Create neural network
        layer = QuantumNeuralLayer(
            QuantumNeuralConfig(
                n_qubits=n_qubits,
                n_quantum_layers=2,
                n_classical_layers=2,
                learning_rate=0.01,
                quantum_circuit_depth=3
            )
        )
        
        # Measure neural network memory
        neural_memory = process.memory_info().rss - quantum_memory - baseline_memory
        print(f"Neural network memory usage: {neural_memory / 1024 / 1024:.2f} MB")
        
        # Verify reasonable memory usage
        max_memory_mb = 1024  # 1 GB
        total_memory = (quantum_memory + neural_memory) / 1024 / 1024
        assert total_memory < max_memory_mb
        
    def test_gpu_acceleration(self):
        """Test GPU acceleration if available."""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")
            
        # Create neural network
        layer = QuantumNeuralLayer(
            QuantumNeuralConfig(
                n_qubits=6,
                n_quantum_layers=2,
                n_classical_layers=2,
                learning_rate=0.01,
                quantum_circuit_depth=3
            )
        )
        
        # Move to GPU
        layer.cuda()
        
        # Test forward pass
        batch_size = 32
        input_size = 2**6
        x = torch.randn(batch_size, input_size, device='cuda')
        
        # Measure GPU forward pass time
        start_time = time.perf_counter()
        with torch.no_grad():
            _ = layer(x)
        gpu_time = time.perf_counter() - start_time
        
        # Compare with CPU
        layer.cpu()
        x = x.cpu()
        start_time = time.perf_counter()
        with torch.no_grad():
            _ = layer(x)
        cpu_time = time.perf_counter() - start_time
        
        print(f"\nGPU forward pass time: {gpu_time:.6f}s")
        print(f"CPU forward pass time: {cpu_time:.6f}s")
        
        # Verify GPU acceleration
        assert gpu_time < cpu_time
        
    @staticmethod
    def _get_baseline_time(n_qubits: int) -> float:
        """Get baseline execution time for n_qubits."""
        return 1e-6 * (2**n_qubits)  # Approximate baseline
        
    @staticmethod
    def _get_baseline_neural_time() -> float:
        """Get baseline neural network execution time."""
        return 1e-3  # 1ms baseline
        
    @staticmethod
    def _get_baseline_blockchain_time() -> float:
        """Get baseline blockchain execution time."""
        return 1e-2  # 10ms baseline
        
    @staticmethod
    def _create_test_transactions(n: int) -> List[Any]:
        """Create test transactions."""
        from src.blockchain.core.blockchain import Transaction
        
        transactions = []
        for i in range(n):
            tx = Transaction(
                sender=f"sender_{i}",
                receiver=f"receiver_{i}",
                amount=100.0,
                timestamp=datetime.now().timestamp()
            )
            transactions.append(tx)
        return transactions
        
    @staticmethod
    def _perform_quantum_operations(n_qubits: int, n_operations: int) -> None:
        """Perform quantum operations for parallel testing."""
        quantum_register = QuantumRegister(n_qubits)
        for _ in range(n_operations):
            quantum_register.apply_gate(np.eye(2**n_qubits), 0)
            
    @staticmethod
    def _generate_market_data_stream(n_updates: int) -> List[Dict[str, Any]]:
        """Generate stream of test market data."""
        data_stream = []
        base_price = 100.0
        
        for i in range(n_updates):
            data = {
                'asset_0': {
                    'price': base_price * (1 + 0.001 * np.random.randn()),
                    'volume': 1000000 * np.random.random(),
                    'volatility': 0.15 * np.random.random()
                }
            }
            data_stream.append(data)
            
        return data_stream
