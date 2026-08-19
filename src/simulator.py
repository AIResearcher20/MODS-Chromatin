import numpy as np
from tqdm import tqdm
import time

class ChromatinSimulator:
    def __init__(self, num_beads=100):
        self.num_beads = num_beads
        
    def generate_sample(self):
        spring_k = np.random.uniform(0.5, 2.0)
        attraction = np.random.uniform(0.1, 1.0)
        noise = np.random.uniform(0.01, 0.15)
        compaction = np.random.uniform(0.3, 0.8)
        
        positions = np.random.randn(self.num_beads, 3) * compaction
        
        for _ in range(30):
            diff = positions[1:] - positions[:-1]
            forces = spring_k * diff
            positions[1:] += 0.01 * forces
            positions[:-1] -= 0.01 * forces
            
            for i in range(self.num_beads):
                for j in range(i+2, min(i+15, self.num_beads)):
                    diff_vec = positions[j] - positions[i]
                    dist = np.linalg.norm(diff_vec)
                    if 0.1 < dist < 2.5:
                        force = attraction * diff_vec / (dist ** 2)
                        positions[i] += 0.005 * force
                        positions[j] -= 0.005 * force
            
            positions += noise * np.random.randn(self.num_beads, 3)
        
        hic = np.zeros((self.num_beads, self.num_beads))
        for i in range(self.num_beads):
            diff = positions - positions[i]
            dist = np.linalg.norm(diff, axis=1)
            contacts = np.exp(-dist ** 2 / 2)
            contacts[i] = 0
            hic[i] = contacts
        hic = (hic + hic.T) / 2
        hic = hic / hic.max() if hic.max() > 0 else hic
        
        i = np.arange(self.num_beads)
        chip = np.zeros((self.num_beads, 3))
        chip[:, 0] = np.random.exponential(0.5, self.num_beads) * (1 + 0.5 * np.sin(i/10))
        chip[:, 1] = np.random.exponential(0.3, self.num_beads) * (1 + 0.5 * np.cos(i/15))
        chip[:, 2] = np.random.exponential(0.4, self.num_beads) * (1 + 0.5 * np.sin(i/20 + 1))
        
        active_score = chip[:, 0] + chip[:, 1] * 0.5
        rna = np.random.exponential(0.5, self.num_beads) * (1 + 0.8 * active_score)
        
        return {
            'positions': positions,
            'hic_matrix': hic,
            'chip_signals': chip,
            'rna_expression': rna,
            'ground_truth': {
                'spring_k': spring_k,
                'attraction': attraction,
                'noise': noise,
                'compaction': compaction
            }
        }
    
    def generate_dataset(self, n_samples):
        samples = []
        start = time.time()
        for i in tqdm(range(n_samples)):
            samples.append(self.generate_sample())
            if (i + 1) % 100 == 0:
                elapsed = time.time() - start
                speed = (i + 1) / elapsed
                remaining = (n_samples - i - 1) / speed
                print(f"   {i+1}/{n_samples} | Speed: {speed:.1f} samples/sec | ETA: {remaining/60:.1f} min")
        print(f" Done! Total time: {(time.time()-start)/60:.2f} minutes")
        return samples
