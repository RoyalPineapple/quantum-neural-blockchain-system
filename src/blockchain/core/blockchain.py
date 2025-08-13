import hashlib
import time
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import json
import numpy as np
from ..utils.cryptography import QuantumSignature, generate_quantum_signature
from ..utils.consensus import QuantumConsensus

@dataclass
class Transaction:
    """Represents a quantum-secured blockchain transaction."""
    sender: str
    receiver: str
    amount: float
    quantum_signature: Optional[QuantumSignature] = None
    timestamp: float = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert transaction to dictionary format."""
        return {
            'sender': self.sender,
            'receiver': self.receiver,
            'amount': self.amount,
            'quantum_signature': self.quantum_signature.to_dict() if self.quantum_signature else None,
            'timestamp': self.timestamp
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Transaction':
        """Create transaction from dictionary format."""
        return cls(
            sender=data['sender'],
            receiver=data['receiver'],
            amount=data['amount'],
            quantum_signature=QuantumSignature.from_dict(data['quantum_signature']) 
                if data['quantum_signature'] else None,
            timestamp=data['timestamp']
        )

@dataclass
class Block:
    """Represents a block in the quantum-secured blockchain."""
    index: int
    transactions: List[Transaction]
    timestamp: float
    previous_hash: str
    nonce: int = 0
    hash: str = ""
    quantum_state: Optional[np.ndarray] = None
    
    def calculate_hash(self) -> str:
        """Calculate block hash including quantum state."""
        block_string = json.dumps(self.to_dict(), sort_keys=True)
        if self.quantum_state is not None:
            # Include quantum state in hash calculation
            quantum_string = np.array2string(self.quantum_state, precision=8)
            block_string += quantum_string
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert block to dictionary format."""
        return {
            'index': self.index,
            'transactions': [t.to_dict() for t in self.transactions],
            'timestamp': self.timestamp,
            'previous_hash': self.previous_hash,
            'nonce': self.nonce,
            'hash': self.hash,
            'quantum_state': self.quantum_state.tolist() if self.quantum_state is not None else None
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Block':
        """Create block from dictionary format."""
        return cls(
            index=data['index'],
            transactions=[Transaction.from_dict(t) for t in data['transactions']],
            timestamp=data['timestamp'],
            previous_hash=data['previous_hash'],
            nonce=data['nonce'],
            hash=data['hash'],
            quantum_state=np.array(data['quantum_state']) if data['quantum_state'] else None
        )

class QuantumBlockchain:
    """
    Quantum-secured blockchain implementation with hybrid classical-quantum security.
    """
    
    def __init__(self, difficulty: int = 4):
        """
        Initialize quantum blockchain.
        
        Args:
            difficulty: Mining difficulty (number of leading zeros required)
        """
        self.chain: List[Block] = []
        self.pending_transactions: List[Transaction] = []
        self.difficulty = difficulty
        self.quantum_consensus = QuantumConsensus()
        
        # Create genesis block
        self.create_genesis_block()
        
    def create_genesis_block(self) -> None:
        """Create and add genesis block to chain."""
        genesis_block = Block(
            index=0,
            transactions=[],
            timestamp=time.time(),
            previous_hash="0"
        )
        genesis_block.hash = genesis_block.calculate_hash()
        self.chain.append(genesis_block)
        
    def get_latest_block(self) -> Block:
        """Get the most recent block in the chain."""
        return self.chain[-1]
        
    def add_transaction(self, transaction: Transaction) -> bool:
        """
        Add new transaction to pending transactions.
        
        Args:
            transaction: Transaction to add
            
        Returns:
            bool: Success status
        """
        if not transaction.quantum_signature:
            # Generate quantum signature for transaction
            transaction.quantum_signature = generate_quantum_signature(transaction.to_dict())
            
        if not self.verify_transaction(transaction):
            return False
            
        self.pending_transactions.append(transaction)
        return True
        
    def verify_transaction(self, transaction: Transaction) -> bool:
        """
        Verify transaction using quantum signature.
        
        Args:
            transaction: Transaction to verify
            
        Returns:
            bool: Verification status
        """
        if not transaction.quantum_signature:
            return False
            
        # Verify quantum signature
        transaction_data = transaction.to_dict()
        transaction_data['quantum_signature'] = None  # Remove signature for verification
        return transaction.quantum_signature.verify(transaction_data)
        
    def mine_pending_transactions(self, miner_address: str) -> Optional[Block]:
        """
        Mine pending transactions into new block.
        
        Args:
            miner_address: Address to receive mining reward
            
        Returns:
            Optional[Block]: Newly mined block if successful
        """
        if not self.pending_transactions:
            return None
            
        # Create mining reward transaction
        reward_transaction = Transaction(
            sender="network",
            receiver=miner_address,
            amount=10.0  # Mining reward
        )
        self.pending_transactions.append(reward_transaction)
        
        # Create new block
        block = Block(
            index=len(self.chain),
            transactions=self.pending_transactions,
            timestamp=time.time(),
            previous_hash=self.get_latest_block().hash
        )
        
        # Perform proof of work
        block = self.proof_of_work(block)
        
        # Add quantum state to block
        block.quantum_state = self.quantum_consensus.generate_quantum_state(block)
        
        # Add block to chain
        if self.add_block(block):
            self.pending_transactions = []
            return block
        return None
        
    def proof_of_work(self, block: Block) -> Block:
        """
        Perform proof of work to mine block.
        
        Args:
            block: Block to mine
            
        Returns:
            Block: Mined block
        """
        target = "0" * self.difficulty
        
        while True:
            block.hash = block.calculate_hash()
            if block.hash.startswith(target):
                break
            block.nonce += 1
            
        return block
        
    def add_block(self, block: Block) -> bool:
        """
        Add new block to chain after verification.
        
        Args:
            block: Block to add
            
        Returns:
            bool: Success status
        """
        if not self.verify_block(block):
            return False
            
        # Verify quantum consensus
        if not self.quantum_consensus.verify_block(block, self.chain):
            return False
            
        self.chain.append(block)
        return True
        
    def verify_block(self, block: Block) -> bool:
        """
        Verify block integrity.
        
        Args:
            block: Block to verify
            
        Returns:
            bool: Verification status
        """
        # Verify index
        if block.index != len(self.chain):
            return False
            
        # Verify previous hash
        if block.previous_hash != self.get_latest_block().hash:
            return False
            
        # Verify block hash
        if block.hash != block.calculate_hash():
            return False
            
        # Verify transactions
        for transaction in block.transactions:
            if not self.verify_transaction(transaction):
                return False
                
        return True
        
    def verify_chain(self) -> bool:
        """
        Verify integrity of entire blockchain.
        
        Returns:
            bool: Verification status
        """
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i-1]
            
            # Verify block hash
            if current_block.hash != current_block.calculate_hash():
                return False
                
            # Verify chain continuity
            if current_block.previous_hash != previous_block.hash:
                return False
                
            # Verify quantum states
            if not self.quantum_consensus.verify_block(current_block, self.chain[:i]):
                return False
                
        return True
        
    def get_balance(self, address: str) -> float:
        """
        Calculate balance for given address.
        
        Args:
            address: Address to check
            
        Returns:
            float: Current balance
        """
        balance = 0.0
        
        for block in self.chain:
            for transaction in block.transactions:
                if transaction.sender == address:
                    balance -= transaction.amount
                if transaction.receiver == address:
                    balance += transaction.amount
                    
        return balance
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert blockchain to dictionary format."""
        return {
            'chain': [block.to_dict() for block in self.chain],
            'pending_transactions': [t.to_dict() for t in self.pending_transactions],
            'difficulty': self.difficulty
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QuantumBlockchain':
        """Create blockchain from dictionary format."""
        blockchain = cls(difficulty=data['difficulty'])
        blockchain.chain = [Block.from_dict(b) for b in data['chain']]
        blockchain.pending_transactions = [Transaction.from_dict(t) 
                                        for t in data['pending_transactions']]
        return blockchain
