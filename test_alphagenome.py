import os
import numpy as np
import sys
from dotenv import load_dotenv

load_dotenv()

# 1. Check for library availability
try:
    from alphagenome.models import dna_client
    from alphagenome.data import genome
except ImportError:
    print("❌ Error: 'alphagenome' library not installed.")
    sys.exit(1)

# 2. Get API Key
api_key = os.getenv("ALPHAGENOME_API_KEY")
if not api_key:
    print("❌ Error: ALPHAGENOME_API_KEY environment variable not set.")
    sys.exit(1)

def run_test():
    print("🔄 Connecting to AlphaGenome API...")
    
    try:
        # Initialize Client
        client = dna_client.create(api_key)
        
        # 3. Create a dummy sequence with EXACTLY 1,048,576 bases (1 MiB)
        REQUIRED_LENGTH = 1_048_576  # 2^20
        
        center_seq = "ATCG" * 100
        padding_needed = REQUIRED_LENGTH - len(center_seq)
        left_pad = padding_needed // 2
        right_pad = padding_needed - left_pad
        
        full_sequence = ("N" * left_pad) + center_seq + ("N" * right_pad)
        
        print(f"⚡ Sending request ({len(full_sequence)} bp)...")
        
        # 4. Request Prediction (Lung Tissue = UBERON:0002048)
        output = client.predict_sequence(
            sequence=full_sequence,
            requested_outputs=[dna_client.OutputType.DNASE],
            ontology_terms=["UBERON:0002048"]
        )
        
        # 5. Check Results
        dnase_values = output.dnase.values
        avg_signal = np.mean(dnase_values)
        
        print("\n✅ SUCCESS: Connection established!")
        print(f"   - Received DNase track shape: {dnase_values.shape}")
        print(f"   - Average signal strength: {avg_signal:.5f}")

    except Exception as e:
        print(f"\n❌ FAILURE: The test crashed.")
        print(f"   Error details: {e}")

if __name__ == "__main__":
    run_test()

