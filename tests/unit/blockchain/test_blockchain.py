import pytest
import numpy as np
from typing import List, Dict, Any
from datetime import datetime, timedelta

from src.blockchain.core.blockchain import QuantumBlockchain, Block, Transaction
from src.blockchain.utils.cryptography import QuantumSignature, generate_quantum_signature
from src.blockchain.utils.consensus import QuantumConsensus

@pytest.mark.blockchain
class TestQuantumBlockchain:
    """Test quantum blockchain functionality."""
    
    def test_initialization(self, quantum_blockchain: QuantumBlockchain):
        """Test blockchain initialization."""
        # Check genesis block
        assert len(quantum_blockchain.chain) == 1
        genesis_block = quantum_blockchain.chain[0]
        assert genesis_block.index == 0
        assert genesis_block.previous_hash == "0"
        assert len(genesis_block.transactions) == 0
        
        # Check initial state
        assert len(quantum_blockchain.pending_transactions) == 0
        assert quantum_blockchain.difficulty > 0
        
    def test_transaction_creation(self, quantum_blockchain: QuantumBlockchain):
        """Test transaction creation and validation."""
        # Create transaction
        transaction = Transaction(
            sender="alice",
            receiver="bob",
            amount=100.0,
            timestamp=datetime.now().timestamp()
        )
        
        # Generate quantum signature
        transaction.quantum_signature = generate_quantum_signature(
            transaction.to_dict()
        )
        
        # Add transaction
        success = quantum_blockchain.add_transaction(transaction)
        assert success
        assert len(quantum_blockchain.pending_transactions) == 1
        
        # Verify transaction
        assert quantum_blockchain.verify_transaction(transaction)
        
    def test_block_mining(self, quantum_blockchain: QuantumBlockchain):
        """Test block mining process."""
        # Add some transactions
        for i in range(3):
            transaction = Transaction(
                sender=f"sender_{i}",
                receiver=f"receiver_{i}",
                amount=50.0 * (i + 1),
                timestamp=datetime.now().timestamp()
            )
            transaction.quantum_signature = generate_quantum_signature(
                transaction.to_dict()
            )
            quantum_blockchain.add_transaction(transaction)
            
        # Mine block
        miner_address = "miner_1"
        new_block = quantum_blockchain.mine_pending_transactions(miner_address)
        
        # Verify block was mined
        assert new_block is not None
        assert len(quantum_blockchain.chain) == 2
        assert len(quantum_blockchain.pending_transactions) == 0
        
        # Verify block properties
        assert new_block.index == 1
        assert new_block.previous_hash == quantum_blockchain.chain[0].hash
        assert len(new_block.transactions) == 4  # 3 regular + 1 mining reward
        assert new_block.quantum_state is not None
        
    def test_chain_validation(self, quantum_blockchain: QuantumBlockchain):
        """Test blockchain validation."""
        # Add and mine some blocks
        for i in range(3):
            transaction = Transaction(
                sender=f"sender_{i}",
                receiver=f"receiver_{i}",
                amount=100.0,
                timestamp=datetime.now().timestamp()
            )
            transaction.quantum_signature = generate_quantum_signature(
                transaction.to_dict()
            )
            quantum_blockchain.add_transaction(transaction)
            quantum_blockchain.mine_pending_transactions("miner_1")
            
        # Verify chain
        assert quantum_blockchain.verify_chain()
        
        # Tamper with a block
        quantum_blockchain.chain[1].transactions[0].amount = 200.0
        
        # Chain should be invalid
        assert not quantum_blockchain.verify_chain()
        
    def test_quantum_consensus(self, quantum_blockchain: QuantumBlockchain):
        """Test quantum consensus mechanism."""
        # Create and mine a block
        transaction = Transaction(
            sender="alice",
            receiver="bob",
            amount=100.0,
            timestamp=datetime.now().timestamp()
        )
        transaction.quantum_signature = generate_quantum_signature(
            transaction.to_dict()
        )
        quantum_blockchain.add_transaction(transaction)
        
        new_block = quantum_blockchain.mine_pending_transactions("miner_1")
        
        # Verify quantum consensus
        consensus = QuantumConsensus()
        assert consensus.verify_block(new_block, quantum_blockchain.chain)
        
        # Tamper with quantum state
        original_state = new_block.quantum_state.copy()
        new_block.quantum_state = np.random.randn(*original_state.shape)
        
        # Consensus should fail
        assert not consensus.verify_block(new_block, quantum_blockchain.chain)
        
    def test_fork_resolution(self, quantum_blockchain: QuantumBlockchain):
        """Test blockchain fork resolution."""
        # Create two competing chains
        chain1 = quantum_blockchain.chain.copy()
        chain2 = quantum_blockchain.chain.copy()
        
        # Add different blocks to each chain
        for chain in [chain1, chain2]:
            transaction = Transaction(
                sender="alice",
                receiver="bob",
                amount=100.0,
                timestamp=datetime.now().timestamp()
            )
            transaction.quantum_signature = generate_quantum_signature(
                transaction.to_dict()
            )
            block = Block(
                index=len(chain),
                transactions=[transaction],
                timestamp=datetime.now().timestamp(),
                previous_hash=chain[-1].hash
            )
            block.hash = block.calculate_hash()
            chain.append(block)
            
        # Resolve fork (longer chain should win)
        chain1.append(Block(
            index=len(chain1),
            transactions=[],
            timestamp=datetime.now().timestamp(),
            previous_hash=chain1[-1].hash
        ))
        
        # Verify longer chain is valid
        assert len(chain1) > len(chain2)
        for block in chain1:
            assert quantum_blockchain.verify_block(block)
            
    def test_transaction_validation(self, quantum_blockchain: QuantumBlockchain):
        """Test transaction validation rules."""
        # Test valid transaction
        valid_transaction = Transaction(
            sender="alice",
            receiver="bob",
            amount=100.0,
            timestamp=datetime.now().timestamp()
        )
        valid_transaction.quantum_signature = generate_quantum_signature(
            valid_transaction.to_dict()
        )
        assert quantum_blockchain.verify_transaction(valid_transaction)
        
        # Test transaction without signature
        invalid_transaction = Transaction(
            sender="alice",
            receiver="bob",
            amount=100.0,
            timestamp=datetime.now().timestamp()
        )
        assert not quantum_blockchain.verify_transaction(invalid_transaction)
        
        # Test transaction with invalid signature
        tampered_transaction = Transaction(
            sender="alice",
            receiver="bob",
            amount=100.0,
            timestamp=datetime.now().timestamp()
        )
        tampered_transaction.quantum_signature = generate_quantum_signature(
            valid_transaction.to_dict()  # Use signature from different transaction
        )
        assert not quantum_blockchain.verify_transaction(tampered_transaction)
        
    def test_block_validation(self, quantum_blockchain: QuantumBlockchain):
        """Test block validation rules."""
        # Create valid block
        transaction = Transaction(
            sender="alice",
            receiver="bob",
            amount=100.0,
            timestamp=datetime.now().timestamp()
        )
        transaction.quantum_signature = generate_quantum_signature(
            transaction.to_dict()
        )
        
        valid_block = Block(
            index=len(quantum_blockchain.chain),
            transactions=[transaction],
            timestamp=datetime.now().timestamp(),
            previous_hash=quantum_blockchain.chain[-1].hash
        )
        valid_block.hash = valid_block.calculate_hash()
        
        # Test valid block
        assert quantum_blockchain.verify_block(valid_block)
        
        # Test invalid index
        invalid_block = valid_block.copy()
        invalid_block.index += 1
        assert not quantum_blockchain.verify_block(invalid_block)
        
        # Test invalid previous hash
        invalid_block = valid_block.copy()
        invalid_block.previous_hash = "invalid_hash"
        assert not quantum_blockchain.verify_block(invalid_block)
        
        # Test invalid transaction
        invalid_block = valid_block.copy()
        invalid_block.transactions[0].amount = 200.0  # Tamper with transaction
        assert not quantum_blockchain.verify_block(invalid_block)
        
    @pytest.mark.parametrize("test_case", [
        {
            'n_transactions': 5,
            'amount_range': (10, 100),
            'expected_reward': 10.0
        },
        {
            'n_transactions': 10,
            'amount_range': (50, 500),
            'expected_reward': 10.0
        }
    ])
    def test_mining_rewards(self, quantum_blockchain: QuantumBlockchain,
                          test_case: Dict[str, Any]):
        """Test mining reward distribution."""
        # Add transactions
        for i in range(test_case['n_transactions']):
            amount = np.random.uniform(*test_case['amount_range'])
            transaction = Transaction(
                sender=f"sender_{i}",
                receiver=f"receiver_{i}",
                amount=amount,
                timestamp=datetime.now().timestamp()
            )
            transaction.quantum_signature = generate_quantum_signature(
                transaction.to_dict()
            )
            quantum_blockchain.add_transaction(transaction)
            
        # Mine block
        miner_address = "miner_1"
        new_block = quantum_blockchain.mine_pending_transactions(miner_address)
        
        # Verify mining reward
        reward_transaction = new_block.transactions[-1]
        assert reward_transaction.sender == "network"
        assert reward_transaction.receiver == miner_address
        assert reward_transaction.amount == test_case['expected_reward']
        
    def test_quantum_signature_verification(self, quantum_blockchain: QuantumBlockchain):
        """Test quantum signature verification process."""
        # Create transaction
        transaction = Transaction(
            sender="alice",
            receiver="bob",
            amount=100.0,
            timestamp=datetime.now().timestamp()
        )
        
        # Generate and verify valid signature
        valid_signature = generate_quantum_signature(transaction.to_dict())
        transaction.quantum_signature = valid_signature
        assert quantum_blockchain.verify_transaction(transaction)
        
        # Test signature tampering
        tampered_signature = QuantumSignature(
            signature_state=np.random.randn(*valid_signature.signature_state.shape),
            verification_state=valid_signature.verification_state,
            classical_hash=valid_signature.classical_hash
        )
        transaction.quantum_signature = tampered_signature
        assert not quantum_blockchain.verify_transaction(transaction)
        
    def test_blockchain_persistence(self, quantum_blockchain: QuantumBlockchain,
                                 tmp_path):
        """Test blockchain state persistence."""
        # Add some blocks
        for i in range(3):
            transaction = Transaction(
                sender=f"sender_{i}",
                receiver=f"receiver_{i}",
                amount=100.0,
                timestamp=datetime.now().timestamp()
            )
            transaction.quantum_signature = generate_quantum_signature(
                transaction.to_dict()
            )
            quantum_blockchain.add_transaction(transaction)
            quantum_blockchain.mine_pending_transactions("miner_1")
            
        # Save state
        save_path = tmp_path / "blockchain_state.pkl"
        state_dict = quantum_blockchain.to_dict()
        
        # Create new blockchain from state
        new_blockchain = QuantumBlockchain.from_dict(state_dict)
        
        # Verify state
        assert len(new_blockchain.chain) == len(quantum_blockchain.chain)
        assert new_blockchain.difficulty == quantum_blockchain.difficulty
        
        for block1, block2 in zip(new_blockchain.chain, quantum_blockchain.chain):
            assert block1.hash == block2.hash
            assert block1.previous_hash == block2.previous_hash
            assert len(block1.transactions) == len(block2.transactions)
            
    @pytest.mark.parametrize("difficulty", [2, 3, 4])
    def test_mining_difficulty(self, difficulty: int):
        """Test mining difficulty adjustment."""
        blockchain = QuantumBlockchain(difficulty=difficulty)
        
        # Add transaction
        transaction = Transaction(
            sender="alice",
            receiver="bob",
            amount=100.0,
            timestamp=datetime.now().timestamp()
        )
        transaction.quantum_signature = generate_quantum_signature(
            transaction.to_dict()
        )
        blockchain.add_transaction(transaction)
        
        # Mine block
        new_block = blockchain.mine_pending_transactions("miner_1")
        
        # Verify block hash meets difficulty requirement
        assert new_block.hash.startswith("0" * difficulty)
        
    def test_concurrent_transactions(self, quantum_blockchain: QuantumBlockchain):
        """Test handling of concurrent transactions."""
        # Create multiple transactions with same timestamp
        timestamp = datetime.now().timestamp()
        transactions = []
        
        for i in range(5):
            transaction = Transaction(
                sender=f"sender_{i}",
                receiver=f"receiver_{i}",
                amount=100.0,
                timestamp=timestamp
            )
            transaction.quantum_signature = generate_quantum_signature(
                transaction.to_dict()
            )
            transactions.append(transaction)
            
        # Add transactions
        for transaction in transactions:
            quantum_blockchain.add_transaction(transaction)
            
        # Mine block
        new_block = quantum_blockchain.mine_pending_transactions("miner_1")
        
        # Verify all transactions were included
        assert len(new_block.transactions) == len(transactions) + 1  # +1 for mining reward
        
    def test_chain_reorganization(self, quantum_blockchain: QuantumBlockchain):
        """Test chain reorganization with competing chains."""
        # Create competing chain
        competing_chain = quantum_blockchain.chain.copy()
        
        # Add blocks to original chain
        for i in range(2):
            transaction = Transaction(
                sender=f"sender_{i}",
                receiver=f"receiver_{i}",
                amount=100.0,
                timestamp=datetime.now().timestamp()
            )
            transaction.quantum_signature = generate_quantum_signature(
                transaction.to_dict()
            )
            quantum_blockchain.add_transaction(transaction)
            quantum_blockchain.mine_pending_transactions("miner_1")
            
        # Add more blocks to competing chain
        for i in range(3):
            transaction = Transaction(
                sender=f"alt_sender_{i}",
                receiver=f"alt_receiver_{i}",
                amount=200.0,
                timestamp=datetime.now().timestamp()
            )
            transaction.quantum_signature = generate_quantum_signature(
                transaction.to_dict()
            )
            block = Block(
                index=len(competing_chain),
                transactions=[transaction],
                timestamp=datetime.now().timestamp(),
                previous_hash=competing_chain[-1].hash
            )
            block.hash = block.calculate_hash()
            competing_chain.append(block)
            
        # Verify competing chain is longer
        assert len(competing_chain) > len(quantum_blockchain.chain)
        
        # Verify all blocks in competing chain are valid
        for block in competing_chain:
            assert quantum_blockchain.verify_block(block)
