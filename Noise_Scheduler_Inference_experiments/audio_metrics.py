import os
import numpy as np
import torch
import torchaudio
import librosa
import glob
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr
from tqdm import tqdm

class AudioMetrics:
    def __init__(self, audio_dir, sample_rate=22050):
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def _load_audio(self, file_path):
        """Load audio file and convert to torch tensor"""
        waveform, sr = torchaudio.load(file_path)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        return waveform.to(self.device).squeeze(0)  # Convert to mono if stereo
    
    def compute_lsd(self, original, inpainted, win_size=2048, hop_size=512):
        """
        Compute Log-Spectral Distance (LSD) between original and inpainted audio
        
        Args:
            original (torch.Tensor): Original audio signal
            inpainted (torch.Tensor): Inpainted audio signal
            win_size (int): STFT window size
            hop_size (int): STFT hop size
        
        Returns:
            float: LSD value
        """
        # Ensure the tensors are on the same device
        original = original.to(self.device)
        inpainted = inpainted.to(self.device)
        
        # Make sure both signals have the same length
        min_len = min(original.shape[0], inpainted.shape[0])
        original = original[:min_len]
        inpainted = inpainted[:min_len]
        
        # Compute STFT
        window = torch.hann_window(win_size).to(self.device)
        stft_orig = torch.stft(
            original, 
            n_fft=win_size, 
            hop_length=hop_size,
            window=window,
            return_complex=True
        )
        
        stft_inpainted = torch.stft(
            inpainted, 
            n_fft=win_size, 
            hop_length=hop_size,
            window=window,
            return_complex=True
        )
        
        # Compute magnitude spectra
        spec_orig = torch.abs(stft_orig)
        spec_inpainted = torch.abs(stft_inpainted)
        
        # Compute log spectra
        log_spec_orig = torch.log10(spec_orig + 1e-8)
        log_spec_inpainted = torch.log10(spec_inpainted + 1e-8)
        
        # Compute LSD
        lsd = torch.mean((log_spec_orig - log_spec_inpainted) ** 2, dim=0)
        lsd = torch.sqrt(lsd)
        lsd = torch.mean(lsd).item()
        
        return lsd
    
    def compute_fad(self, original, inpainted, sr=22050):
        """
        Compute Fréchet Audio Distance (FAD) between original and inpainted audio
        Implementation based on the paper "FAD: A MEASURE OF ESTIMATING PERCEPTUAL AUDIO QUALITY"
        
        Args:
            original (torch.Tensor): Original audio signal
            inpainted (torch.Tensor): Inpainted audio signal
            sr (int): Sample rate
            
        Returns:
            float: FAD value
        """
        # Convert torch tensors to numpy for librosa
        if torch.is_tensor(original):
            original = original.detach().cpu().numpy()
        if torch.is_tensor(inpainted):
            inpainted = inpainted.detach().cpu().numpy()
        
        # Make sure both signals have the same length
        min_len = min(len(original), len(inpainted))
        original = original[:min_len]
        inpainted = inpainted[:min_len]
        
        # Extract MFCCs for both signals (VGGish-like features)
        mfcc_orig = librosa.feature.mfcc(y=original, sr=sr, n_mfcc=13)
        mfcc_inpainted = librosa.feature.mfcc(y=inpainted, sr=sr, n_mfcc=13)
        
        # Compute mean and covariance for both MFCCs
        mu_orig = np.mean(mfcc_orig, axis=1)
        cov_orig = np.cov(mfcc_orig)
        
        mu_inpainted = np.mean(mfcc_inpainted, axis=1)
        cov_inpainted = np.cov(mfcc_inpainted)
        
        # Compute the squared difference between means
        mean_diff = np.sum((mu_orig - mu_inpainted) ** 2)
        
        # Compute the Fréchet distance
        # sqrt(tr(cov_orig + cov_inpainted - 2*sqrt(cov_orig*cov_inpainted)) + mean_diff)
        
        # For numerical stability
        eps = 1e-6
        cov_prod = np.sqrt(cov_orig @ cov_inpainted + eps * np.eye(cov_orig.shape[0]))
        cov_sum = cov_orig + cov_inpainted
        trace_term = np.trace(cov_sum - 2 * cov_prod)
        
        # The Fréchet distance
        fad = mean_diff + trace_term
        
        return fad
    
    def compute_spectral_convergence(self, original, inpainted):
        """
        Compute spectral convergence metric that's less dependent on gap length
        It measures how well the spectral content converges regardless of gap size
        
        Args:
            original (torch.Tensor): Original audio signal
            inpainted (torch.Tensor): Inpainted audio signal
            
        Returns:
            float: Spectral convergence value (lower is better)
        """
        # Ensure the tensors are on the same device
        original = original.to(self.device)
        inpainted = inpainted.to(self.device)
        
        # Make sure both signals have the same length
        min_len = min(original.shape[0], inpainted.shape[0])
        original = original[:min_len]
        inpainted = inpainted[:min_len]
        
        # Compute spectral centroid features
        spec_centroid_orig = librosa.feature.spectral_centroid(
            y=original.cpu().numpy(), sr=self.sample_rate
        )[0]
        
        spec_centroid_inp = librosa.feature.spectral_centroid(
            y=inpainted.cpu().numpy(), sr=self.sample_rate
        )[0]
        
        # Compute spectral contrast features
        spec_contrast_orig = librosa.feature.spectral_contrast(
            y=original.cpu().numpy(), sr=self.sample_rate
        )
        
        spec_contrast_inp = librosa.feature.spectral_contrast(
            y=inpainted.cpu().numpy(), sr=self.sample_rate
        )
        
        # Compute correlation coefficients for spectral features
        # Higher correlation means better preservation of spectral characteristics
        centroid_corr, _ = pearsonr(spec_centroid_orig, spec_centroid_inp)
        
        # For spectral contrast, compute correlation for each band and average
        contrast_corrs = []
        for i in range(spec_contrast_orig.shape[0]):
            corr, _ = pearsonr(spec_contrast_orig[i], spec_contrast_inp[i])
            contrast_corrs.append(corr)
        
        avg_contrast_corr = np.mean(contrast_corrs)
        
        # Combine correlations (transform to [0, 1] where 1 is best)
        centroid_corr = (centroid_corr + 1) / 2  # Transform from [-1, 1] to [0, 1]
        avg_contrast_corr = (avg_contrast_corr + 1) / 2
        
        # Final metric: weighted average of correlations (higher is better)
        spectral_convergence = (centroid_corr + avg_contrast_corr) / 2
        
        return spectral_convergence
    
    def compute_normalized_harmonicity(self, original, inpainted):
        """
        Compute harmonicity similarity that's normalized against gap length
        
        Args:
            original (torch.Tensor): Original audio signal
            inpainted (torch.Tensor): Inpainted audio signal
            
        Returns:
            float: Normalized harmonicity similarity (higher is better)
        """
        # Convert torch tensors to numpy for librosa
        if torch.is_tensor(original):
            original = original.detach().cpu().numpy()
        if torch.is_tensor(inpainted):
            inpainted = inpainted.detach().cpu().numpy()
        
        # Make sure both signals have the same length
        min_len = min(len(original), len(inpainted))
        original = original[:min_len]
        inpainted = inpainted[:min_len]
        
        # Compute harmonicity features
        harmonic_orig, percussive_orig = librosa.effects.hpss(original)
        harmonic_inp, percussive_inp = librosa.effects.hpss(inpainted)
        
        # Compute energy ratios
        energy_orig = np.sum(harmonic_orig**2) / (np.sum(percussive_orig**2) + 1e-8)
        energy_inp = np.sum(harmonic_inp**2) / (np.sum(percussive_inp**2) + 1e-8)
        
        # Compute relative error
        relative_error = np.abs(energy_orig - energy_inp) / (energy_orig + 1e-8)
        
        # Normalize to [0, 1] where 1 means perfect match
        norm_harmonicity = np.exp(-relative_error)
        
        return norm_harmonicity
    
    def analyze_inpainting_results(self):
        """
        Analyze all audio files in the inpainting directory
        Compute LSD, FAD, and gap-independent metrics
        """
        # Find all gap lengths
        gap_lengths = [100, 200, 400]  # Based on your folder structure
        schedulers = ['sigmoid', 'cosine', 'power']  # Based on your folder structure
        
        # Dictionary to store results
        results = {
            'gap_length': [],
            'scheduler': [],
            'sample_idx': [],
            'lsd': [],
            'fad': [],
            'spectral_convergence': [],
            'normalized_harmonicity': []
        }
        
        # Process each gap length
        for gap_length in gap_lengths:
            # Process inpainting results for each scheduler
            for scheduler in schedulers:
                # Find all sample indices for this gap and scheduler
                pattern = f'inpainted_gap{gap_length}ms_{scheduler}_sample_*.wav'
                inpainted_files = glob.glob(os.path.join(self.audio_dir, pattern))
                
                for inpainted_file in tqdm(inpainted_files, desc=f"Gap {gap_length}ms, {scheduler}"):
                    # Extract sample index
                    file_name = os.path.basename(inpainted_file)
                    sample_idx = int(file_name.split('_')[-1].split('.')[0])
                    
                    # Load corresponding original and inpainted audio
                    original_file = os.path.join(self.audio_dir, f'Original_{gap_length}ms_sample_{sample_idx}.wav')
                    
                    if not os.path.exists(original_file):
                        print(f"Warning: Original file not found: {original_file}")
                        continue
                    
                    # Load audio
                    original = self._load_audio(original_file)
                    inpainted = self._load_audio(inpainted_file)
                    
                    # Compute metrics
                    lsd = self.compute_lsd(original, inpainted)
                    fad = self.compute_fad(original, inpainted)
                    spectral_convergence = self.compute_spectral_convergence(original, inpainted)
                    normalized_harmonicity = self.compute_normalized_harmonicity(original, inpainted)
                    
                    # Store results
                    results['gap_length'].append(gap_length)
                    results['scheduler'].append(scheduler)
                    results['sample_idx'].append(sample_idx)
                    results['lsd'].append(lsd)
                    results['fad'].append(fad)
                    results['spectral_convergence'].append(spectral_convergence)
                    results['normalized_harmonicity'].append(normalized_harmonicity)
            
            # Process masked audio (no inpainting) - COMMENTED OUT
            """
            masked_files = glob.glob(os.path.join(self.audio_dir, f'Masked_{gap_length}ms_sample_*.wav'))
            
            for masked_file in tqdm(masked_files, desc=f"Gap {gap_length}ms, masked"):
                # Extract sample index
                file_name = os.path.basename(masked_file)
                sample_idx = int(file_name.split('_')[-1].split('.')[0])
                
                # Load corresponding original and masked audio
                original_file = os.path.join(self.audio_dir, f'Original_{gap_length}ms_sample_{sample_idx}.wav')
                
                if not os.path.exists(original_file):
                    print(f"Warning: Original file not found: {original_file}")
                    continue
                
                # Load audio
                original = self._load_audio(original_file)
                masked = self._load_audio(masked_file)
                
                # Compute metrics
                lsd = self.compute_lsd(original, masked)
                fad = self.compute_fad(original, masked)
                spectral_convergence = self.compute_spectral_convergence(original, masked)
                normalized_harmonicity = self.compute_normalized_harmonicity(original, masked)
                
                # Store results
                results['gap_length'].append(gap_length)
                results['scheduler'].append('masked')  # This is for the masked audio (no inpainting)
                results['sample_idx'].append(sample_idx)
                results['lsd'].append(lsd)
                results['fad'].append(fad)
                results['spectral_convergence'].append(spectral_convergence)
                results['normalized_harmonicity'].append(normalized_harmonicity)
            """
        
        # Convert to DataFrame for easier analysis
        results_df = pd.DataFrame(results)
        
        # Group by gap_length and scheduler to calculate mean and std
        grouped = results_df.groupby(['gap_length', 'scheduler'])
        mean_results = grouped.mean().reset_index()
        std_results = grouped.std().reset_index()
        
        # Save results to CSV
        results_df.to_csv(os.path.join(self.audio_dir, 'inpainting_metrics_detailed.csv'), index=False)
        mean_results.to_csv(os.path.join(self.audio_dir, 'inpainting_metrics_mean.csv'), index=False)
        std_results.to_csv(os.path.join(self.audio_dir, 'inpainting_metrics_std.csv'), index=False)
        
        # Create visualizations
        self._plot_metrics_with_error_bars(mean_results, std_results)
        
        return results_df, mean_results, std_results
    
    def _plot_metrics_with_error_bars(self, mean_results, std_results):
        """Create visualizations of the metrics with error bars showing standard deviation"""
        # Set up the figure
        fig, axs = plt.subplots(2, 2, figsize=(15, 12))
        
        metrics = [
            ('lsd', 'Log-Spectral Distance (lower is better)'),
            ('fad', 'Fréchet Audio Distance (lower is better)'),
            ('spectral_convergence', 'Spectral Convergence (higher is better)'),
            ('normalized_harmonicity', 'Normalized Harmonicity (higher is better)')
        ]
        
        # Get unique gap lengths and schedulers
        gap_lengths = sorted(mean_results['gap_length'].unique())
        schedulers = sorted(mean_results['scheduler'].unique())
        
        # Comment out adding masked to schedulers
        """
        # Add 'masked' to schedulers if not already present
        if 'masked' not in schedulers:
            schedulers.append('masked')
        """
        
        # Colors and markers for different schedulers
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', 'k']  # Blue, Orange, Green, Black
        markers = ['o', 's', '^', 'x']
        
        # Plot each metric
        for i, (metric, title) in enumerate(metrics):
            row, col = i // 2, i % 2
            ax = axs[row, col]
            
            for j, scheduler in enumerate(schedulers):
                # Skip masked data
                if scheduler == 'masked':
                    continue
                
                # Get data for this scheduler
                mean_data = mean_results[mean_results['scheduler'] == scheduler]
                std_data = std_results[std_results['scheduler'] == scheduler]
                
                if len(mean_data) == 0:
                    print(f"No data for scheduler {scheduler}")
                    continue
                
                # Sort by gap length
                mean_data = mean_data.sort_values('gap_length')
                std_data = std_data.sort_values('gap_length')
                
                # Extract mean and std values for this metric
                x = mean_data['gap_length']
                y = mean_data[metric]
                yerr = std_data[metric] if len(std_data) > 0 else None
                
                # Plot with error bars
                ax.errorbar(
                    x, y, yerr=yerr, 
                    marker=markers[j % len(markers)], 
                    color=colors[j % len(colors)], 
                    label=scheduler,
                    capsize=5,
                    markersize=8,
                    linewidth=2,
                    elinewidth=1
                )
            
            ax.set_title(title, fontsize=14)
            ax.set_xlabel('Gap Length (ms)', fontsize=12)
            ax.set_ylabel('Value', fontsize=12)
            ax.set_xticks(gap_lengths)
            ax.legend(fontsize=10)
            
            # For metrics where higher is better, ensure y-axis is properly scaled
            if 'higher' in title:
                bottom, top = ax.get_ylim()
                ax.set_ylim(bottom, min(1.0, top * 1.1))  # Cap at 1.0 for normalized metrics
            
            # Add grid for better readability
            ax.grid(True, linestyle='--', alpha=0.7)
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.audio_dir, 'inpainting_metrics_comparison.svg'), format='svg')
        
        # Also save as PNG for easy viewing
        plt.savefig(os.path.join(self.audio_dir, 'inpainting_metrics_comparison.png'), dpi=300)
        plt.close()

if __name__ == "__main__":
    # Directory containing the audio files
    audio_dir = "E:\\Class\\ECE661\\Diffusion_Audio_Inpainting\\inpainting_length_scheduler_comparison"
    
    # Create metrics calculator
    metrics = AudioMetrics(audio_dir)
    
    # Analyze results
    results_df, mean_results, std_results = metrics.analyze_inpainting_results()
    
    print("Analysis complete. Results saved to:")
    print(f"- {os.path.join(audio_dir, 'inpainting_metrics_detailed.csv')}")
    print(f"- {os.path.join(audio_dir, 'inpainting_metrics_mean.csv')}")
    print(f"- {os.path.join(audio_dir, 'inpainting_metrics_std.csv')}")
    print(f"- {os.path.join(audio_dir, 'inpainting_metrics_comparison.svg')}")
    print(f"- {os.path.join(audio_dir, 'inpainting_metrics_comparison.png')}") 