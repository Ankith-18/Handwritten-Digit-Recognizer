# run_all_smart.py
import subprocess
import sys
import time
import os

print("="*50)
print("HANDWRITTEN DIGIT RECOGNIZER")
print("Running all files in sequence")
print("="*50)

# Check if models already exist
cnn_model_exists = os.path.exists('best_cnn_model.h5') or os.path.exists('cnn_digit_recognizer.h5')
mlp_model_exists = os.path.exists('mlp_digit_recognizer.h5')

print(f"\n📊 Existing models found:")
print(f"   - CNN Model: {'✅' if cnn_model_exists else '❌'}")
print(f"   - MLP Model: {'✅' if mlp_model_exists else '❌'}")

files_to_run = []

# Always run data exploration and preprocessing (they're fast)
files_to_run.extend(["1_data_exploration.py", "2_data_preprocessing.py"])

# Run MLP only if needed
if not mlp_model_exists:
    files_to_run.append("3_simple_model.py")
else:
    print("\n⏭️  Skipping MLP training (model already exists)")

# Run CNN only if needed
if not cnn_model_exists:
    files_to_run.append("4_cnn_model.py")
else:
    print("⏭️  Skipping CNN training (model already exists)")

print("\n" + "="*50)
print(f"Files to run: {len(files_to_run)}")
print("="*50)

for i, file in enumerate(files_to_run, 1):
    print(f"\n[{i}/{len(files_to_run)}] Running {file}...")
    print("-"*30)
    
    result = subprocess.run([sys.executable, file])
    
    if result.returncode != 0:
        print(f"❌ Error in {file}")
        break
    else:
        print(f"✅ {file} completed successfully")
    
    time.sleep(2)

print("\n" + "="*50)
print("✅ SETUP COMPLETE!")
print("="*50)
print("\n🎯 Your models are ready! Run one of these:")
print("   python 5_prediction_interface_fixed.py  - For GUI")
print("   streamlit run 6_web_interface.py        - For Web App")