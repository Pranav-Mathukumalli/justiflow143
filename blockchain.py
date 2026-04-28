"""
JustiFlow — Simulated Blockchain Module
========================================
Implements a local proof-of-work blockchain that:
  1. Records a tamper-proof fingerprint of every uploaded CSV file
  2. Logs every analysis result as an immutable block
  3. Allows verification that a file has not been modified since upload
  4. Persists the chain to disk so it survives server restarts
 
In production this would be swapped for Web3.py + Ethereum / Polygon.
The API surface is identical — only the storage backend changes.
"""
 
import hashlib
import json
import time
import os
from datetime import datetime
 
 
CHAIN_FILE = "blockchain_ledger.json"
 
 
# ── BLOCK ─────────────────────────────────────────────────────────────────────
 
class Block:
    def __init__(self, index: int, block_type: str, data: dict,
                 previous_hash: str, timestamp: float | None = None):
        self.index         = index
        self.block_type    = block_type   # "GENESIS" | "FILE_UPLOAD" | "ANALYSIS"
        self.data          = data
        self.previous_hash = previous_hash
        self.timestamp     = timestamp or time.time()
        self.nonce         = 0
        self.hash          = self._mine()
 
    # ── proof-of-work (difficulty 2 = fast for demo, still tamper-evident)
    def _mine(self) -> str:
        difficulty = 2
        prefix = "0" * difficulty
        while True:
            candidate = self._compute_hash()
            if candidate.startswith(prefix):
                return candidate
            self.nonce += 1
 
    def _compute_hash(self) -> str:
        payload = json.dumps({
            "index":         self.index,
            "block_type":    self.block_type,
            "data":          self.data,
            "previous_hash": self.previous_hash,
            "timestamp":     self.timestamp,
            "nonce":         self.nonce,
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()
 
    def to_dict(self) -> dict:
        return {
            "index":         self.index,
            "block_type":    self.block_type,
            "data":          self.data,
            "previous_hash": self.previous_hash,
            "timestamp":     self.timestamp,
            "nonce":         self.nonce,
            "hash":          self.hash,
        }
 
    @classmethod
    def from_dict(cls, d: dict) -> "Block":
        b = cls.__new__(cls)
        b.index         = d["index"]
        b.block_type    = d["block_type"]
        b.data          = d["data"]
        b.previous_hash = d["previous_hash"]
        b.timestamp     = d["timestamp"]
        b.nonce         = d["nonce"]
        b.hash          = d["hash"]
        return b
 
 
# ── BLOCKCHAIN ────────────────────────────────────────────────────────────────
 
class JustiChain:
    def __init__(self):
        self.chain: list[Block] = []
        self._load_or_create()
 
    # ── persistence ──────────────────────────────────────────────────────────
 
    def _load_or_create(self):
        if os.path.exists(CHAIN_FILE):
            try:
                with open(CHAIN_FILE, "r") as f:
                    raw = json.load(f)
                self.chain = [Block.from_dict(b) for b in raw]
                return
            except Exception:
                pass
        # Fresh chain — create genesis block
        genesis = Block(
            index         = 0,
            block_type    = "GENESIS",
            data          = {"message": "JustiFlow Blockchain Initialised",
                             "version": "1.0"},
            previous_hash = "0" * 64,
        )
        self.chain = [genesis]
        self._save()
 
    def _save(self):
        try:
            with open(CHAIN_FILE, "w") as f:
                json.dump([b.to_dict() for b in self.chain], f, indent=2)
        except Exception as e:
            print(f"[blockchain] Could not save chain: {e}")
 
    # ── public API ───────────────────────────────────────────────────────────
 
    @property
    def latest(self) -> Block:
        return self.chain[-1]
 
    def add_file_block(self, file_hash: str, filename: str,
                       file_size_bytes: int, analysis_id: str) -> Block:
        """Record that a file was uploaded and fingerprinted."""
        block = Block(
            index         = len(self.chain),
            block_type    = "FILE_UPLOAD",
            data          = {
                "file_hash":        file_hash,
                "filename":         filename,
                "file_size_bytes":  file_size_bytes,
                "analysis_id":      analysis_id,
                "uploaded_at":      datetime.utcnow().isoformat() + "Z",
            },
            previous_hash = self.latest.hash,
        )
        self.chain.append(block)
        self._save()
        return block
 
    def add_analysis_block(self, analysis_id: str, file_hash: str,
                           bias_score: float, risk_level: str,
                           protected_col: str, outcome_col: str,
                           row_count: int) -> Block:
        """Record the result of a bias analysis."""
        block = Block(
            index         = len(self.chain),
            block_type    = "ANALYSIS",
            data          = {
                "analysis_id":   analysis_id,
                "file_hash":     file_hash,
                "bias_score":    bias_score,
                "risk_level":    risk_level,
                "protected_col": protected_col,
                "outcome_col":   outcome_col,
                "row_count":     row_count,
                "analysed_at":   datetime.utcnow().isoformat() + "Z",
            },
            previous_hash = self.latest.hash,
        )
        self.chain.append(block)
        self._save()
        return block
 
    def verify_file(self, file_bytes: bytes) -> dict:
        """
        Check if a file's SHA-256 hash matches any FILE_UPLOAD block.
        Returns verification result dict.
        """
        file_hash = hashlib.sha256(file_bytes).hexdigest()
        for block in self.chain:
            if block.block_type == "FILE_UPLOAD" and \
               block.data.get("file_hash") == file_hash:
                return {
                    "verified":     True,
                    "file_hash":    file_hash,
                    "block_index":  block.index,
                    "block_hash":   block.hash,
                    "analysis_id":  block.data.get("analysis_id"),
                    "uploaded_at":  block.data.get("uploaded_at"),
                    "filename":     block.data.get("filename"),
                }
        return {
            "verified":  False,
            "file_hash": file_hash,
            "message":   "No matching record found on chain. File may be new or tampered.",
        }
 
    def is_valid(self) -> bool:
        """Full chain integrity check — detects any tampering."""
        for i in range(1, len(self.chain)):
            curr = self.chain[i]
            prev = self.chain[i - 1]
            if curr.previous_hash != prev.hash:
                return False
            # Recompute hash to catch data mutation
            if curr._compute_hash() != curr.hash:
                return False
        return True
 
    def get_analysis_history(self, limit: int = 20) -> list[dict]:
        """Return last N ANALYSIS blocks newest-first."""
        blocks = [b for b in self.chain if b.block_type == "ANALYSIS"]
        blocks.sort(key=lambda b: b.timestamp, reverse=True)
        return [b.to_dict() for b in blocks[:limit]]
 
    def get_file_record(self, analysis_id: str) -> dict | None:
        """Return the FILE_UPLOAD block for a given analysis ID."""
        for b in self.chain:
            if b.block_type == "FILE_UPLOAD" and \
               b.data.get("analysis_id") == analysis_id:
                return b.to_dict()
        return None
 
    def summary(self) -> dict:
        total      = len(self.chain)
        uploads    = sum(1 for b in self.chain if b.block_type == "FILE_UPLOAD")
        analyses   = sum(1 for b in self.chain if b.block_type == "ANALYSIS")
        valid      = self.is_valid()
        return {
            "total_blocks":    total,
            "file_uploads":    uploads,
            "analyses":        analyses,
            "chain_valid":     valid,
            "genesis_hash":    self.chain[0].hash,
            "latest_hash":     self.latest.hash,
            "latest_index":    self.latest.index,
        }
 
 
# ── FILE HASHING UTILITY ──────────────────────────────────────────────────────
 
def hash_file_bytes(file_bytes: bytes) -> str:
    """Return SHA-256 hex digest of raw file bytes."""
    return hashlib.sha256(file_bytes).hexdigest()
 
 
def hash_dataframe(df) -> str:
    """Deterministic hash of a pandas DataFrame contents."""
    import pandas as pd
    csv_bytes = df.to_csv(index=False).encode()
    return hashlib.sha256(csv_bytes).hexdigest()
 
 
# ── MODULE-LEVEL SINGLETON ────────────────────────────────────────────────────
 
justi_chain = JustiChain()