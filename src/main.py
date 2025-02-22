import subprocess
import os

def run_edge_detection_script():
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    edge_script = os.path.join(current_dir, "coin_edge.py")
    
    result = subprocess.run(["python", edge_script], capture_output=True, text=True)
    
    coin_count = result.stdout.strip()
    return coin_count

def run_segmentation_script():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(current_dir, "coin_segmentation.py")
    
    result = subprocess.run(["python", script_path], capture_output=True, text=True)
    
    output = result.stdout.strip()
    return output

def main():
    method = 0 #0 for edge detection and 1 for segementation
    if(method):
        coin_count = run_segmentation_script()
        print("Segmentation-based detection found", coin_count, "coins.")
    else:
        coin_count = run_edge_detection_script()
        print("Edge-based detection found", coin_count, "coins.")

if __name__ == "__main__":
    main()