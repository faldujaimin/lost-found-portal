import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from app import app, db, Item, User, BlockchainBlock
from blockchain import Block
import json

def test_blockchain():
    with app.app_context():
        print("--- Blockchain Verification Test ---")
        
        # 1. Create a dummy user if none exists
        user = User.query.first()
        if not user:
            print("Creating dummy user...")
            user = User(registration_no="TEST001", full_name="Test User")
            user.set_password("password")
            db.session.add(user)
            db.session.commit()

        # 2. Add an item (this should trigger blockchain recording)
        print("Reporting a new item...")
        from app import record_to_blockchain
        
        item = Item(
            item_name="Blockchain Test Phone",
            description="A phone for testing blockchain",
            lost_found_date=db.func.current_date(),
            location="Lab",
            status="Found",
            reporter=user
        )
        db.session.add(item)
        db.session.commit()
        
        record_to_blockchain(item)
        
        # 3. Verify it's valid initially
        from app import verify_item_blockchain
        valid, msg = verify_item_blockchain(item.id)
        print(f"Initial Verification: {valid} - {msg}")
        
        if not valid:
            print("Error: Initial verification failed!")
            return

        # 4. Tamper with the database
        print("\nTAMPERING with the database record...")
        item.description = "MALICIOUS CHANGE - This item is actually a gold watch"
        db.session.commit()
        
        # 5. Verify again (should fail)
        valid, msg = verify_item_blockchain(item.id)
        print(f"Verification after Tampering: {valid} - {msg}")
        
        if not valid and "tampered" in msg.lower():
            print("\nSUCCESS: Blockchain correctly detected tampering!")
        else:
            print("\nFAILURE: Blockchain did NOT detect tampering correctly.")

        # Cleanup
        db.session.delete(item)
        # Delete the block too
        block = BlockchainBlock.query.filter_by(item_id=item.id).first()
        if block:
             db.session.delete(block)
        db.session.commit()

if __name__ == "__main__":
    test_blockchain()
