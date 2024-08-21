import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from matplotlib.animation import FuncAnimation

def plot_protein(ax, coords):
    ax.clear()
    ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], '-o', markersize=2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Sampled Protein Structure')

def create_protein_animation(sample_dir, output_file):
    sample_files = sorted([f for f in os.listdir(sample_dir) if f.endswith('.npy')], key=lambda x:int(x.split('_')[2].split('.')[0]))
    print(sample_files)
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    def update(frame):
        coords = np.load(os.path.join(sample_dir, sample_files[frame]))
        plot_protein(ax, coords)
        ax.set_title(f'Sampled Protein Structure (Epoch {(frame+1)})')
    
    anim = FuncAnimation(fig, update, frames=len(sample_files), interval=50, repeat_delay=1000)
    anim.save(output_file, writer='pillow', fps=2)
    plt.close(fig)

def visualize_live(sample_dir):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    while True:
        sample_files = sorted([f for f in os.listdir(sample_dir) if f.endswith('.npy')], key=lambda x:int(x.split('_')[2].split('.')[0]))
        print(sample_files)
        if sample_files:
            latest_sample = sample_files[-1]
            print(f"{latest_sample=}")
            coords = np.load(os.path.join(sample_dir, latest_sample))
            plot_protein(ax, coords)
            epoch = int(latest_sample.split('_')[2].split('.')[0])
            ax.set_title(f'Sampled Protein Structure (Epoch {epoch})')
            plt.pause(5)  # Update every 5 seconds
        else:
            plt.pause(1)

if __name__ == "__main__":
    sample_dir = 'samples'
    
    # Uncomment one of the following lines based on your preference:
    
    # For live visualization during training:
    #visualize_live(sample_dir)
    
    # For creating an animation after training:
    create_protein_animation(sample_dir, 'protein_evolution.gif')