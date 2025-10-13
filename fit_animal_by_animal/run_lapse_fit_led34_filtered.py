"""
Run lapse fitting for LED34 batch with filtered stimuli
Animals: 45, 57, 59, 61
Output directory: led34_vanila_lapse_led34_filered
"""
import subprocess
import sys

BATCH = "LED34"
OUTPUT_DIR = "led34_vanila_lapse_led34_filered"
ANIMALS = [45, 57, 59, 61]

print("Starting lapse fitting for LED34 batch (filtered stimuli)")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Animals: {ANIMALS}")
print("=" * 50)

for animal in ANIMALS:
    print(f"\nProcessing Animal {animal}...")
    print("-" * 50)
    
    cmd = [
        "python", "lapses_fit_single_animal.py",
        "--batch", BATCH,
        "--animal", str(animal),
        "--output-dir", OUTPUT_DIR,
        "--is-stim-filtered"
    ]
    
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print(f"✓ Successfully completed Animal {animal}")
    else:
        print(f"✗ Error processing Animal {animal}")
        sys.exit(1)

print("\n" + "=" * 50)
print("All animals processed successfully!")
print(f"Results saved in: {OUTPUT_DIR}")
