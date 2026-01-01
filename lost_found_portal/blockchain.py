import hashlib
import json
from time import time

class Block:
    def __init__(self, index, timestamp, item_id, item_data, previous_hash, nonce=0):
        self.index = index
        self.timestamp = timestamp
        self.item_id = item_id
        self.item_data = item_data  # Should be a dict or string of item details
        self.previous_hash = previous_hash
        self.nonce = nonce
        self.hash = self.calculate_hash()

    def calculate_hash(self):
        block_string = json.dumps({
            "index": self.index,
            "timestamp": self.timestamp,
            "item_id": self.item_id,
            "item_data": self.item_data,
            "previous_hash": self.previous_hash,
            "nonce": self.nonce
        }, sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()

    def mine_block(self, difficulty):
        target = "0" * difficulty
        while self.hash[:difficulty] != target:
            self.nonce += 1
            self.hash = self.calculate_hash()

class Blockchain:
    def __init__(self):
        self.chain = []
        self.difficulty = 2  # Low difficulty for fast processing in this portal
        # Genesis block
        self.create_block(item_id=0, item_data="Genesis Block", previous_hash="0")

    def create_block(self, item_id, item_data, previous_hash=None):
        if previous_hash is None:
            previous_hash = self.get_latest_block().hash
        
        new_block = Block(
            index=len(self.chain),
            timestamp=time(),
            item_id=item_id,
            item_data=item_data,
            previous_hash=previous_hash
        )
        new_block.mine_block(self.difficulty)
        self.chain.append(new_block)
        return new_block

    def get_latest_block(self):
        return self.chain[-1]

    def is_chain_valid(self):
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i-1]

            if current_block.hash != current_block.calculate_hash():
                return False
            if current_block.previous_hash != previous_block.hash:
                return False
        return True

    def to_json(self):
        return json.dumps([vars(b) for b in self.chain])

    @classmethod
    def from_json(cls, json_data):
        bc = cls()
        bc.chain = []
        data = json.loads(json_data)
        for b_data in data:
            block = Block(
                index=b_data['index'],
                timestamp=b_data['timestamp'],
                item_id=b_data['item_id'],
                item_data=b_data['item_data'],
                previous_hash=b_data['previous_hash'],
                nonce=b_data['nonce']
            )
            block.hash = b_data['hash']
            bc.chain.append(block)
        return bc
