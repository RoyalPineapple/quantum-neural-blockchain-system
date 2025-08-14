from typing import List, Optional, Dict, Any
from datetime import datetime
from ...blockchain.core.blockchain import QuantumBlockchain
from ...blockchain.core.quantum_block import QuantumBlock
from ...blockchain.utils.cryptography import QuantumSignature

class BlockchainService:
    """Service layer for blockchain operations."""
    
    def __init__(self, n_qubits: int = 8):
        """Initialize blockchain service."""
        self.n_qubits = n_qubits
        self.blockchain = QuantumBlockchain(n_qubits=n_qubits)
        self.signature = QuantumSignature(n_qubits=n_qubits)
        
        # Transaction pool
        self.pending_transactions: List[Dict[str, Any]] = []
        
        # Block cache
        self.block_cache: Dict[str, QuantumBlock] = {}
        
    def create_transaction(
        self,
        sender: str,
        receiver: str,
        amount: float,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create new blockchain transaction."""
        try:
            # Create transaction
            transaction = {
                'sender': sender,
                'receiver': receiver,
                'amount': amount,
                'timestamp': datetime.utcnow().timestamp(),
                'data': data
            }
            
            # Add to pending transactions
            self.pending_transactions.append(transaction)
            
            # Add to blockchain
            block_index = self.blockchain.add_transaction(transaction)
            
            # Get transaction hash
            transaction_hash = self.blockchain.get_last_block().hash
            
            return {
                'transaction_hash': transaction_hash,
                'block_index': block_index,
                'status': 'pending',
                'timestamp': transaction['timestamp']
            }
            
        except Exception as e:
            raise ValueError(f"Transaction creation failed: {str(e)}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get blockchain status."""
        try:
            last_block = self.blockchain.get_last_block()
            
            return {
                'chain_length': len(self.blockchain.chain),
                'last_block_hash': last_block.hash,
                'last_block_timestamp': last_block.timestamp,
                'pending_transactions': len(self.pending_transactions),
                'quantum_state': last_block.quantum_state.tolist()
                if last_block.quantum_state is not None else None,
                'is_valid': self.blockchain.is_chain_valid()
            }
            
        except Exception as e:
            raise ValueError(f"Failed to get status: {str(e)}")
    
    def mine_block(self) -> Dict[str, Any]:
        """Mine new block."""
        try:
            # Create mining reward transaction
            reward_transaction = {
                'sender': 'network',
                'receiver': 'miner',  # In production, use actual miner address
                'amount': self.blockchain.get_mining_reward(),
                'timestamp': datetime.utcnow().timestamp(),
                'data': {'type': 'mining_reward'}
            }
            
            # Add reward to pending transactions
            self.pending_transactions.append(reward_transaction)
            
            # Mine block
            block = self.blockchain.mine_pending_transactions('miner')
            
            # Cache block
            self.block_cache[block.hash] = block
            
            # Clear pending transactions
            self.pending_transactions = []
            
            return {
                'block_hash': block.hash,
                'block_index': block.index,
                'transactions': block.transactions,
                'mining_reward': reward_transaction['amount'],
                'quantum_state': block.quantum_state.tolist()
                if block.quantum_state is not None else None
            }
            
        except Exception as e:
            raise ValueError(f"Block mining failed: {str(e)}")
    
    def verify_chain(self) -> bool:
        """Verify blockchain integrity."""
        try:
            return self.blockchain.is_chain_valid()
            
        except Exception as e:
            raise ValueError(f"Chain verification failed: {str(e)}")
    
    def get_balance(self, address: str) -> float:
        """Get balance for address."""
        try:
            return self.blockchain.get_balance(address)
            
        except Exception as e:
            raise ValueError(f"Failed to get balance: {str(e)}")
    
    def get_block(self, block_hash: str) -> Dict[str, Any]:
        """Get block by hash."""
        try:
            # Check cache first
            block = self.block_cache.get(block_hash)
            
            if block is None:
                # Search in blockchain
                for b in self.blockchain.chain:
                    if b.hash == block_hash:
                        block = b
                        break
            
            if block is None:
                raise ValueError(f"Block not found: {block_hash}")
            
            return {
                'index': block.index,
                'timestamp': block.timestamp,
                'transactions': block.transactions,
                'previous_hash': block.previous_hash,
                'hash': block.hash,
                'nonce': block.nonce,
                'quantum_state': block.quantum_state.tolist()
                if block.quantum_state is not None else None,
                'signature': block.signature.tolist()
                if block.signature is not None else None
            }
            
        except Exception as e:
            raise ValueError(f"Failed to get block: {str(e)}")
    
    def get_transaction(self, transaction_hash: str) -> Dict[str, Any]:
        """Get transaction by hash."""
        try:
            # Search in all blocks
            for block in self.blockchain.chain:
                for transaction in block.transactions:
                    # Calculate transaction hash
                    tx_hash = self.blockchain.quantum_hash.generate(
                        str(transaction)
                    )
                    
                    if tx_hash == transaction_hash:
                        return {
                            'transaction': transaction,
                            'block_hash': block.hash,
                            'block_index': block.index,
                            'timestamp': transaction['timestamp'],
                            'status': 'confirmed'
                        }
            
            # Check pending transactions
            for transaction in self.pending_transactions:
                tx_hash = self.blockchain.quantum_hash.generate(
                    str(transaction)
                )
                
                if tx_hash == transaction_hash:
                    return {
                        'transaction': transaction,
                        'timestamp': transaction['timestamp'],
                        'status': 'pending'
                    }
            
            raise ValueError(f"Transaction not found: {transaction_hash}")
            
        except Exception as e:
            raise ValueError(f"Failed to get transaction: {str(e)}")
    
    def get_address_history(self, address: str) -> Dict[str, Any]:
        """Get transaction history for address."""
        try:
            sent_transactions = []
            received_transactions = []
            
            # Search in all blocks
            for block in self.blockchain.chain:
                for transaction in block.transactions:
                    if transaction['sender'] == address:
                        sent_transactions.append({
                            'transaction': transaction,
                            'block_hash': block.hash,
                            'block_index': block.index,
                            'timestamp': transaction['timestamp'],
                            'status': 'confirmed'
                        })
                    
                    if transaction['receiver'] == address:
                        received_transactions.append({
                            'transaction': transaction,
                            'block_hash': block.hash,
                            'block_index': block.index,
                            'timestamp': transaction['timestamp'],
                            'status': 'confirmed'
                        })
            
            # Check pending transactions
            for transaction in self.pending_transactions:
                if transaction['sender'] == address:
                    sent_transactions.append({
                        'transaction': transaction,
                        'timestamp': transaction['timestamp'],
                        'status': 'pending'
                    })
                
                if transaction['receiver'] == address:
                    received_transactions.append({
                        'transaction': transaction,
                        'timestamp': transaction['timestamp'],
                        'status': 'pending'
                    })
            
            return {
                'address': address,
                'balance': self.blockchain.get_balance(address),
                'sent_transactions': sent_transactions,
                'received_transactions': received_transactions,
                'total_sent': sum(tx['transaction']['amount']
                                for tx in sent_transactions),
                'total_received': sum(tx['transaction']['amount']
                                   for tx in received_transactions)
            }
            
        except Exception as e:
            raise ValueError(f"Failed to get address history: {str(e)}")
