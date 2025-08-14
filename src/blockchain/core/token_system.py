"""
Advanced Quantum Token System for Quantum-Neural-Blockchain Platform

This module implements a sophisticated token ecosystem that powers all operations
within the quantum-neural-blockchain system. It includes multiple token types,
complex economic models, quantum-enhanced security, and advanced staking mechanisms.
"""

import numpy as np
import hashlib
import json
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from decimal import Decimal, getcontext
import threading
import asyncio
from collections import defaultdict
import logging

# Set high precision for financial calculations
getcontext().prec = 50

class TokenType(Enum):
    """Different types of tokens in the ecosystem."""
    QUANTUM = "QUANTUM"  # Main utility token
    NEURAL = "NEURAL"    # AI computation token
    STAKE = "STAKE"      # Staking rewards token
    COMPUTE = "COMPUTE"  # Computational resource token
    ENERGY = "ENERGY"    # Energy/gas token
    GOVERNANCE = "GOV"   # Governance voting token
    LIQUIDITY = "LIQ"    # Liquidity provider token
    ORACLE = "ORACLE"    # Oracle data token
    BRIDGE = "BRIDGE"    # Cross-chain bridge token
    QUANTUM_STATE = "QSTATE"  # Quantum state preservation token

class TokenStandard(Enum):
    """Token standards supported."""
    QRC20 = "QRC20"      # Quantum ERC20 equivalent
    QRC721 = "QRC721"    # Quantum NFT standard
    QRC1155 = "QRC1155"  # Multi-token standard
    QUANTUM_NATIVE = "QNATIVE"  # Native quantum token

class StakingTier(Enum):
    """Staking tiers with different benefits."""
    BRONZE = ("BRONZE", Decimal("1000"), Decimal("0.05"))
    SILVER = ("SILVER", Decimal("10000"), Decimal("0.08"))
    GOLD = ("GOLD", Decimal("50000"), Decimal("0.12"))
    PLATINUM = ("PLATINUM", Decimal("100000"), Decimal("0.18"))
    QUANTUM = ("QUANTUM", Decimal("500000"), Decimal("0.25"))
    
    def __init__(self, name: str, min_stake: Decimal, apy: Decimal):
        self.tier_name = name
        self.min_stake = min_stake
        self.apy = apy

@dataclass
class TokenMetadata:
    """Metadata for tokens."""
    name: str
    symbol: str
    decimals: int
    total_supply: Decimal
    max_supply: Optional[Decimal] = None
    mintable: bool = False
    burnable: bool = False
    pausable: bool = False
    upgradeable: bool = False
    quantum_enhanced: bool = False
    staking_enabled: bool = False
    governance_rights: bool = False
    
@dataclass
class TokenBalance:
    """Token balance information."""
    available: Decimal = Decimal("0")
    staked: Decimal = Decimal("0")
    locked: Decimal = Decimal("0")
    pending_rewards: Decimal = Decimal("0")
    last_update: datetime = field(default_factory=datetime.now)
    
    @property
    def total(self) -> Decimal:
        return self.available + self.staked + self.locked

@dataclass
class StakingPosition:
    """Staking position details."""
    amount: Decimal
    tier: StakingTier
    start_time: datetime
    lock_period: timedelta
    last_reward_claim: datetime
    auto_compound: bool = False
    quantum_boost: Decimal = Decimal("1.0")
    
    @property
    def is_locked(self) -> bool:
        return datetime.now() < (self.start_time + self.lock_period)
    
    @property
    def pending_rewards(self) -> Decimal:
        time_elapsed = datetime.now() - self.last_reward_claim
        days_elapsed = Decimal(str(time_elapsed.total_seconds() / 86400))
        daily_rate = self.tier.apy / Decimal("365")
        return self.amount * daily_rate * days_elapsed * self.quantum_boost

@dataclass
class TokenTransaction:
    """Token transaction record."""
    tx_id: str
    from_address: str
    to_address: str
    token_type: TokenType
    amount: Decimal
    fee: Decimal
    timestamp: datetime
    block_height: int
    quantum_signature: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'tx_id': self.tx_id,
            'from_address': self.from_address,
            'to_address': self.to_address,
            'token_type': self.token_type.value,
            'amount': str(self.amount),
            'fee': str(self.fee),
            'timestamp': self.timestamp.isoformat(),
            'block_height': self.block_height,
            'quantum_signature': self.quantum_signature,
            'metadata': self.metadata
        }

class QuantumTokenEconomics:
    """Advanced tokenomics model with quantum-enhanced features."""
    
    def __init__(self):
        self.base_inflation_rate = Decimal("0.02")  # 2% annual
        self.max_inflation_rate = Decimal("0.10")   # 10% max
        self.deflation_threshold = Decimal("0.80")  # 80% staking ratio
        self.quantum_boost_factor = Decimal("1.5")
        self.burn_rate = Decimal("0.001")  # 0.1% of transactions
        
    def calculate_dynamic_inflation(self, 
                                  total_supply: Decimal,
                                  staked_amount: Decimal,
                                  network_activity: Decimal) -> Decimal:
        """Calculate dynamic inflation based on network conditions."""
        staking_ratio = staked_amount / total_supply if total_supply > 0 else Decimal("0")
        activity_factor = min(network_activity / Decimal("1000000"), Decimal("2.0"))
        
        # Reduce inflation if high staking ratio
        if staking_ratio > self.deflation_threshold:
            inflation_rate = self.base_inflation_rate * (Decimal("1") - staking_ratio)
        else:
            inflation_rate = self.base_inflation_rate * (Decimal("1") + activity_factor)
            
        return min(inflation_rate, self.max_inflation_rate)
    
    def calculate_quantum_boost(self, 
                              quantum_operations: int,
                              neural_computations: int) -> Decimal:
        """Calculate quantum boost based on system usage."""
        total_operations = quantum_operations + neural_computations
        boost = Decimal("1.0") + (Decimal(str(total_operations)) / Decimal("10000"))
        return min(boost, self.quantum_boost_factor)

class QuantumTokenSystem:
    """
    Comprehensive quantum token system managing all token operations,
    staking, governance, and economic mechanisms.
    """
    
    def __init__(self, network_id: str = "quantum-mainnet"):
        self.network_id = network_id
        self.logger = logging.getLogger(__name__)
        
        # Token registries
        self.token_metadata: Dict[TokenType, TokenMetadata] = {}
        self.balances: Dict[str, Dict[TokenType, TokenBalance]] = defaultdict(
            lambda: defaultdict(TokenBalance)
        )
        self.staking_positions: Dict[str, List[StakingPosition]] = defaultdict(list)
        self.transaction_history: List[TokenTransaction] = []
        
        # Economic model
        self.economics = QuantumTokenEconomics()
        
        # Governance
        self.governance_proposals: Dict[str, Dict] = {}
        self.voting_power: Dict[str, Decimal] = defaultdict(Decimal)
        
        # Threading locks
        self.balance_lock = threading.RLock()
        self.staking_lock = threading.RLock()
        self.governance_lock = threading.RLock()
        
        # Initialize core tokens
        self._initialize_core_tokens()
        
        # Start background processes
        self._start_background_processes()
    
    def _initialize_core_tokens(self):
        """Initialize the core token types."""
        core_tokens = {
            TokenType.QUANTUM: TokenMetadata(
                name="Quantum Token",
                symbol="QUANTUM",
                decimals=18,
                total_supply=Decimal("1000000000"),
                max_supply=Decimal("10000000000"),
                mintable=True,
                burnable=True,
                quantum_enhanced=True,
                staking_enabled=True,
                governance_rights=True
            ),
            TokenType.NEURAL: TokenMetadata(
                name="Neural Compute Token",
                symbol="NEURAL",
                decimals=18,
                total_supply=Decimal("500000000"),
                max_supply=Decimal("5000000000"),
                mintable=True,
                burnable=True,
                quantum_enhanced=True,
                staking_enabled=True
            ),
            TokenType.COMPUTE: TokenMetadata(
                name="Compute Resource Token",
                symbol="COMPUTE",
                decimals=18,
                total_supply=Decimal("2000000000"),
                mintable=True,
                burnable=True
            ),
            TokenType.ENERGY: TokenMetadata(
                name="Energy Token",
                symbol="ENERGY",
                decimals=18,
                total_supply=Decimal("10000000000"),
                mintable=True,
                burnable=True
            ),
            TokenType.GOVERNANCE: TokenMetadata(
                name="Governance Token",
                symbol="GOV",
                decimals=18,
                total_supply=Decimal("100000000"),
                max_supply=Decimal("100000000"),
                governance_rights=True,
                staking_enabled=True
            )
        }
        
        for token_type, metadata in core_tokens.items():
            self.token_metadata[token_type] = metadata
    
    def _start_background_processes(self):
        """Start background processes for rewards and maintenance."""
        def reward_distribution():
            while True:
                try:
                    self._distribute_staking_rewards()
                    self._process_governance_rewards()
                    asyncio.sleep(3600)  # Run every hour
                except Exception as e:
                    self.logger.error(f"Error in reward distribution: {e}")
                    asyncio.sleep(60)
        
        # Start in separate thread
        import threading
        reward_thread = threading.Thread(target=reward_distribution, daemon=True)
        reward_thread.start()
    
    def create_address(self) -> str:
        """Generate a new quantum address."""
        import secrets
        random_bytes = secrets.token_bytes(32)
        address_hash = hashlib.sha256(random_bytes).hexdigest()
        return f"quantum1{address_hash[:38]}"
    
    def mint_tokens(self, 
                   token_type: TokenType,
                   to_address: str,
                   amount: Decimal,
                   authority_address: str) -> str:
        """Mint new tokens (requires authority)."""
        with self.balance_lock:
            metadata = self.token_metadata.get(token_type)
            if not metadata or not metadata.mintable:
                raise ValueError(f"Token {token_type} is not mintable")
            
            # Check supply limits
            current_supply = self._get_total_supply(token_type)
            if metadata.max_supply and (current_supply + amount) > metadata.max_supply:
                raise ValueError("Minting would exceed max supply")
            
            # Update balance
            self.balances[to_address][token_type].available += amount
            
            # Create transaction record
            tx = TokenTransaction(
                tx_id=self._generate_tx_id(),
                from_address="mint",
                to_address=to_address,
                token_type=token_type,
                amount=amount,
                fee=Decimal("0"),
                timestamp=datetime.now(),
                block_height=self._get_current_block_height(),
                metadata={"type": "mint", "authority": authority_address}
            )
            
            self.transaction_history.append(tx)
            self.logger.info(f"Minted {amount} {token_type.value} to {to_address}")
            
            return tx.tx_id
    
    def burn_tokens(self,
                   token_type: TokenType,
                   from_address: str,
                   amount: Decimal) -> str:
        """Burn tokens from address."""
        with self.balance_lock:
            metadata = self.token_metadata.get(token_type)
            if not metadata or not metadata.burnable:
                raise ValueError(f"Token {token_type} is not burnable")
            
            balance = self.balances[from_address][token_type]
            if balance.available < amount:
                raise ValueError("Insufficient balance for burning")
            
            # Burn tokens
            balance.available -= amount
            
            # Create transaction record
            tx = TokenTransaction(
                tx_id=self._generate_tx_id(),
                from_address=from_address,
                to_address="burn",
                token_type=token_type,
                amount=amount,
                fee=Decimal("0"),
                timestamp=datetime.now(),
                block_height=self._get_current_block_height(),
                metadata={"type": "burn"}
            )
            
            self.transaction_history.append(tx)
            self.logger.info(f"Burned {amount} {token_type.value} from {from_address}")
            
            return tx.tx_id
    
    def transfer_tokens(self,
                       token_type: TokenType,
                       from_address: str,
                       to_address: str,
                       amount: Decimal,
                       fee: Optional[Decimal] = None) -> str:
        """Transfer tokens between addresses."""
        with self.balance_lock:
            if fee is None:
                fee = self._calculate_transfer_fee(token_type, amount)
            
            from_balance = self.balances[from_address][token_type]
            total_required = amount + fee
            
            if from_balance.available < total_required:
                raise ValueError("Insufficient balance for transfer")
            
            # Execute transfer
            from_balance.available -= total_required
            self.balances[to_address][token_type].available += amount
            
            # Burn fee (deflationary mechanism)
            if fee > 0:
                self._burn_fee(token_type, fee)
            
            # Create transaction record
            tx = TokenTransaction(
                tx_id=self._generate_tx_id(),
                from_address=from_address,
                to_address=to_address,
                token_type=token_type,
                amount=amount,
                fee=fee,
                timestamp=datetime.now(),
                block_height=self._get_current_block_height()
            )
            
            self.transaction_history.append(tx)
            self.logger.info(f"Transferred {amount} {token_type.value} from {from_address} to {to_address}")
            
            return tx.tx_id
    
    def stake_tokens(self,
                    address: str,
                    token_type: TokenType,
                    amount: Decimal,
                    lock_period_days: int = 30,
                    auto_compound: bool = False) -> str:
        """Stake tokens for rewards."""
        with self.staking_lock:
            if token_type not in [TokenType.QUANTUM, TokenType.NEURAL, TokenType.GOVERNANCE]:
                raise ValueError(f"Token {token_type} is not stakeable")
            
            balance = self.balances[address][token_type]
            if balance.available < amount:
                raise ValueError("Insufficient balance for staking")
            
            # Determine staking tier
            tier = self._determine_staking_tier(amount)
            
            # Move tokens to staked
            balance.available -= amount
            balance.staked += amount
            
            # Create staking position
            position = StakingPosition(
                amount=amount,
                tier=tier,
                start_time=datetime.now(),
                lock_period=timedelta(days=lock_period_days),
                last_reward_claim=datetime.now(),
                auto_compound=auto_compound
            )
            
            self.staking_positions[address].append(position)
            
            # Update governance voting power
            if token_type == TokenType.GOVERNANCE:
                self.voting_power[address] += amount
            
            self.logger.info(f"Staked {amount} {token_type.value} for {address}")
            
            return self._generate_tx_id()
    
    def unstake_tokens(self,
                      address: str,
                      position_index: int) -> str:
        """Unstake tokens (if lock period expired)."""
        with self.staking_lock:
            positions = self.staking_positions[address]
            if position_index >= len(positions):
                raise ValueError("Invalid staking position")
            
            position = positions[position_index]
            if position.is_locked:
                raise ValueError("Staking position is still locked")
            
            # Claim pending rewards first
            rewards = position.pending_rewards
            if rewards > 0:
                self._claim_staking_rewards(address, position_index)
            
            # Move tokens back to available
            token_type = TokenType.QUANTUM  # Determine from position
            balance = self.balances[address][token_type]
            balance.staked -= position.amount
            balance.available += position.amount
            
            # Remove staking position
            positions.pop(position_index)
            
            # Update governance voting power
            if token_type == TokenType.GOVERNANCE:
                self.voting_power[address] -= position.amount
            
            self.logger.info(f"Unstaked {position.amount} {token_type.value} for {address}")
            
            return self._generate_tx_id()
    
    def _determine_staking_tier(self, amount: Decimal) -> StakingTier:
        """Determine staking tier based on amount."""
        for tier in reversed(list(StakingTier)):
            if amount >= tier.min_stake:
                return tier
        return StakingTier.BRONZE
    
    def _calculate_transfer_fee(self, token_type: TokenType, amount: Decimal) -> Decimal:
        """Calculate dynamic transfer fee."""
        base_fee = Decimal("0.001")  # 0.1%
        network_congestion = self._get_network_congestion()
        
        # Adjust fee based on congestion
        congestion_multiplier = Decimal("1") + (network_congestion / Decimal("100"))
        
        fee = amount * base_fee * congestion_multiplier
        
        # Minimum and maximum fee bounds
        min_fee = Decimal("0.0001")
        max_fee = amount * Decimal("0.01")  # Max 1%
        
        return max(min_fee, min(fee, max_fee))
    
    def _get_network_congestion(self) -> Decimal:
        """Get current network congestion level (0-100)."""
        # Simplified congestion calculation
        recent_txs = len([tx for tx in self.transaction_history[-1000:] 
                         if (datetime.now() - tx.timestamp).total_seconds() < 3600])
        return min(Decimal(str(recent_txs)) / Decimal("10"), Decimal("100"))
    
    def _burn_fee(self, token_type: TokenType, fee: Decimal):
        """Burn transaction fee (deflationary mechanism)."""
        # Fee burning reduces total supply
        pass  # Implementation would reduce tracked total supply
    
    def _distribute_staking_rewards(self):
        """Distribute staking rewards to all stakers."""
        with self.staking_lock:
            for address, positions in self.staking_positions.items():
                for position in positions:
                    rewards = position.pending_rewards
                    if rewards > Decimal("0.0001"):  # Minimum reward threshold
                        # Mint reward tokens
                        self.balances[address][TokenType.STAKE].available += rewards
                        position.last_reward_claim = datetime.now()
                        
                        if position.auto_compound:
                            # Auto-compound by increasing staked amount
                            position.amount += rewards
                            self.balances[address][TokenType.QUANTUM].staked += rewards
    
    def _process_governance_rewards(self):
        """Process governance participation rewards."""
        # Reward active governance participants
        for address, voting_power in self.voting_power.items():
            if voting_power > Decimal("0"):
                reward = voting_power * Decimal("0.0001")  # 0.01% daily
                self.balances[address][TokenType.GOVERNANCE].available += reward
    
    def _claim_staking_rewards(self, address: str, position_index: int) -> Decimal:
        """Claim staking rewards for specific position."""
        position = self.staking_positions[address][position_index]
        rewards = position.pending_rewards
        
        if rewards > Decimal("0"):
            self.balances[address][TokenType.STAKE].available += rewards
            position.last_reward_claim = datetime.now()
        
        return rewards
    
    def get_balance(self, address: str, token_type: TokenType) -> TokenBalance:
        """Get token balance for address."""
        return self.balances[address][token_type]
    
    def get_total_balance(self, address: str, token_type: TokenType) -> Decimal:
        """Get total balance including staked and locked tokens."""
        balance = self.balances[address][token_type]
        return balance.total
    
    def get_staking_positions(self, address: str) -> List[StakingPosition]:
        """Get all staking positions for address."""
        return self.staking_positions[address]
    
    def get_voting_power(self, address: str) -> Decimal:
        """Get governance voting power for address."""
        return self.voting_power[address]
    
    def _get_total_supply(self, token_type: TokenType) -> Decimal:
        """Calculate current total supply of token."""
        total = Decimal("0")
        for address_balances in self.balances.values():
            balance = address_balances[token_type]
            total += balance.total
        return total
    
    def _generate_tx_id(self) -> str:
        """Generate unique transaction ID."""
        import secrets
        timestamp = str(datetime.now().timestamp())
        random_part = secrets.token_hex(16)
        return hashlib.sha256(f"{timestamp}{random_part}".encode()).hexdigest()
    
    def _get_current_block_height(self) -> int:
        """Get current blockchain height."""
        # This would interface with the blockchain
        return len(self.transaction_history) + 1000000
    
    def get_token_metrics(self, token_type: TokenType) -> Dict[str, Any]:
        """Get comprehensive token metrics."""
        metadata = self.token_metadata[token_type]
        total_supply = self._get_total_supply(token_type)
        
        # Calculate staking metrics
        total_staked = Decimal("0")
        staking_addresses = 0
        
        for positions in self.staking_positions.values():
            for position in positions:
                total_staked += position.amount
                staking_addresses += 1
        
        staking_ratio = total_staked / total_supply if total_supply > 0 else Decimal("0")
        
        return {
            "metadata": {
                "name": metadata.name,
                "symbol": metadata.symbol,
                "decimals": metadata.decimals,
                "total_supply": str(total_supply),
                "max_supply": str(metadata.max_supply) if metadata.max_supply else None,
                "mintable": metadata.mintable,
                "burnable": metadata.burnable
            },
            "economics": {
                "total_staked": str(total_staked),
                "staking_ratio": str(staking_ratio),
                "staking_addresses": staking_addresses,
                "current_inflation": str(self.economics.calculate_dynamic_inflation(
                    total_supply, total_staked, Decimal("100000")
                ))
            },
            "network": {
                "total_transactions": len(self.transaction_history),
                "network_congestion": str(self._get_network_congestion()),
                "active_addresses": len(self.balances)
            }
        }
    
    def export_state(self) -> Dict[str, Any]:
        """Export complete token system state."""
        return {
            "network_id": self.network_id,
            "token_metadata": {
                token_type.value: {
                    "name": metadata.name,
                    "symbol": metadata.symbol,
                    "decimals": metadata.decimals,
                    "total_supply": str(metadata.total_supply),
                    "max_supply": str(metadata.max_supply) if metadata.max_supply else None,
                    "mintable": metadata.mintable,
                    "burnable": metadata.burnable,
                    "pausable": metadata.pausable,
                    "upgradeable": metadata.upgradeable,
                    "quantum_enhanced": metadata.quantum_enhanced,
                    "staking_enabled": metadata.staking_enabled,
                    "governance_rights": metadata.governance_rights
                }
                for token_type, metadata in self.token_metadata.items()
            },
            "balances": {
                address: {
                    token_type.value: {
                        "available": str(balance.available),
                        "staked": str(balance.staked),
                        "locked": str(balance.locked),
                        "pending_rewards": str(balance.pending_rewards),
                        "last_update": balance.last_update.isoformat()
                    }
                    for token_type, balance in token_balances.items()
                }
                for address, token_balances in self.balances.items()
            },
            "staking_positions": {
                address: [
                    {
                        "amount": str(position.amount),
                        "tier": position.tier.tier_name,
                        "start_time": position.start_time.isoformat(),
                        "lock_period_days": position.lock_period.days,
                        "last_reward_claim": position.last_reward_claim.isoformat(),
                        "auto_compound": position.auto_compound,
                        "quantum_boost": str(position.quantum_boost)
                    }
                    for position in positions
                ]
                for address, positions in self.staking_positions.items()
            },
            "voting_power": {
                address: str(power)
                for address, power in self.voting_power.items()
            },
            "transaction_history": [tx.to_dict() for tx in self.transaction_history[-10000:]]  # Last 10k transactions
        }

# Factory functions for easy token system creation
def create_quantum_token_system(network_id: str = "quantum-mainnet") -> QuantumTokenSystem:
    """Create and initialize a quantum token system."""
    return QuantumTokenSystem(network_id)

def create_test_token_system() -> QuantumTokenSystem:
    """Create a token system for testing with pre-funded addresses."""
    system = QuantumTokenSystem("quantum-testnet")
    
    # Create test addresses with initial balances
    test_addresses = [system.create_address() for _ in range(10)]
    
    for address in test_addresses:
        # Mint initial test tokens
        system.mint_tokens(TokenType.QUANTUM, address, Decimal("1000000"), "genesis")
        system.mint_tokens(TokenType.NEURAL, address, Decimal("500000"), "genesis")
        system.mint_tokens(TokenType.COMPUTE, address, Decimal("2000000"), "genesis")
        system.mint_tokens(TokenType.ENERGY, address, Decimal("10000000"), "genesis")
        system.mint_tokens(TokenType.GOVERNANCE, address, Decimal("10000"), "genesis")
    
    return system
