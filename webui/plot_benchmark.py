import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless servers
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

def plot_benchmark(json_file):
    """Create plots from benchmark JSON file"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    results = data['results']
    model_name = data['model']
    video_name = data['video']
    timestamp = data['timestamp']
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Benchmark Results: {model_name} on {video_name}\n{timestamp}', 
                 fontsize=16, fontweight='bold')
    
    # 1. Inference Time Over Frames
    ax = axes[0, 0]
    frames = range(len(results['inference_times_ms']))
    ax.plot(frames, results['inference_times_ms'], linewidth=0.5, alpha=0.7, color='#667eea')
    ax.axhline(results['avg_inference_time_ms'], color='red', linestyle='--', 
               label=f"Avg: {results['avg_inference_time_ms']:.2f} ms")
    ax.set_xlabel('Frame Number')
    ax.set_ylabel('Inference Time (ms)')
    ax.set_title('Inference Time per Frame')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Inference Time Distribution
    ax = axes[0, 1]
    ax.hist(results['inference_times_ms'], bins=50, color='#764ba2', alpha=0.7, edgecolor='black')
    ax.axvline(results['avg_inference_time_ms'], color='red', linestyle='--', 
               label=f"Mean: {results['avg_inference_time_ms']:.2f} ms")
    ax.axvline(np.median(results['inference_times_ms']), color='green', linestyle='--', 
               label=f"Median: {np.median(results['inference_times_ms']):.2f} ms")
    ax.set_xlabel('Inference Time (ms)')
    ax.set_ylabel('Frequency')
    ax.set_title('Inference Time Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Memory Usage Over Time
    ax = axes[0, 2]
    ax.plot(results['memory_usage_MiB'], color='#a855f7', linewidth=1.5)
    ax.axhline(results['avg_memory_usage_MiB'], color='red', linestyle='--', 
               label=f"Avg: {results['avg_memory_usage_MiB']:.2f} MiB")
    ax.set_xlabel('Frame Number')
    ax.set_ylabel('Memory Usage (MiB)')
    ax.set_title('Memory Usage Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.fill_between(range(len(results['memory_usage_MiB'])), 
                      results['memory_usage_MiB'], alpha=0.3, color='#a855f7')
    
    # 4. CPU Usage Over Time
    ax = axes[1, 0]
    ax.plot(results['cpu_usage_percent'], color='#ef4444', linewidth=1.5)
    ax.axhline(results['avg_cpu_usage_percent'], color='darkred', linestyle='--', 
               label=f"Avg: {results['avg_cpu_usage_percent']:.2f}%")
    ax.set_xlabel('Frame Number')
    ax.set_ylabel('CPU Usage (%)')
    ax.set_title('CPU Usage Over Time')
    ax.set_ylim([0, 100])
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.fill_between(range(len(results['cpu_usage_percent'])), 
                      results['cpu_usage_percent'], alpha=0.3, color='#ef4444')
    
    # 5. Energy Consumption Over Time
    ax = axes[1, 1]
    ax.plot(results['energy_consumption_W'], color='#22c55e', linewidth=1.5)
    ax.axhline(results['avg_energy_consumption_W'], color='darkgreen', linestyle='--', 
               label=f"Avg: {results['avg_energy_consumption_W']:.2f} W")
    ax.set_xlabel('Frame Number')
    ax.set_ylabel('Energy Consumption (W)')
    ax.set_title('Energy Consumption Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.fill_between(range(len(results['energy_consumption_W'])), 
                      results['energy_consumption_W'], alpha=0.3, color='#22c55e')
    
    # 6. Summary Statistics
    ax = axes[1, 2]
    ax.axis('off')
    
    summary_text = f"""
    Summary Statistics
    {'='*40}
    
    Total Frames: {results['total_frames']}
    Detections: {data['results']['frames_with_detections']}
    Detection Rate: {data['results']['frames_with_detections']/results['total_frames']*100:.1f}%
    
    Inference Time:
      • Average: {results['avg_inference_time_ms']:.2f} ms
      • Min: {results['min_inference_time_ms']:.2f} ms
      • Max: {results['max_inference_time_ms']:.2f} ms
      • Std Dev: {results['std_inference_time_ms']:.2f} ms
      • FPS: {1000/results['avg_inference_time_ms']:.2f}
    
    Memory Usage:
      • Average: {results['avg_memory_usage_MiB']:.2f} MiB
    
    CPU Usage:
      • Average: {results['avg_cpu_usage_percent']:.2f}%
    
    Energy:
      • Average: {results['avg_energy_consumption_W']:.2f} W
      • Total Energy: {results['avg_energy_consumption_W'] * results['total_time_s']:.2f} J
    """
    
    if 'avg_temperature_C' in results:
        summary_text += f"\n    Temperature:\n      • Average: {results['avg_temperature_C']:.2f}°C"
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, 
            fontfamily='monospace', fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    # Save plot
    output_dir = Path(json_file).parent.parent / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_file = output_dir / f"benchmark_{model_name}_{video_name}_{timestamp}.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {plot_file}")
    
    return str(plot_file)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
        plot_benchmark(json_file)
    else:
        print("Usage: python plot_benchmark.py <benchmark_json_file>")
