import os
import time
import math
import numpy as np
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
import matplotlib.animation as animation  # Keep this for animation if still desired

try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x, **kw: x  # fallback: identity iterator

def extract_frequency(input_video_path, input_audio_path, output_path, verbose=False):
    # A-weighting (numerically safe) - (keep as is)
    def a_weighting_db(freqs_hz):
        """Return A-weighting in dB for array freqs_hz (numerically safe)."""
        f = np.asarray(freqs_hz, dtype=float)
        f_safe = np.where(f == 0.0, 1e-12, f)
        f2 = f_safe * f_safe
        ra_num = (12194.0 ** 2) * (f2 ** 2)
        ra_den = ((f2 + 20.6 ** 2) *
                (f2 + 12194.0 ** 2) *
                np.sqrt((f2 + 107.7 ** 2) * (f2 + 737.9 ** 2)))
        with np.errstate(divide='ignore', invalid='ignore'):
            ra = ra_num / ra_den
            A = 20.0 * np.log10(ra) + 2.0
        A = np.where(np.isfinite(A), A, -120.0)
        return A


    # Core: build time-sliced heatmaps (vectorized STFT decoding) - (modified section)
    def build_heatmaps_over_time(W, X, Y, Z, sr,
                                n_fft=2048, hop_length=512,
                                az_res=72, el_res=36,
                                time_step=1.0,
                                normalize_global=True,
                                save_dir=None,
                                save_pngs=False,
                                save_mp4_raw_heatmap=False,
                                save_mp4_with_graph=False,
                                video_fps=30   # ← NEW PARAM
                                ):
        """
        Build heatmaps for each time slice, and duplicate them to match a video framerate.
        """

        # -----------------------
        # 1. Compute STFTs
        # -----------------------
        print("Computing STFTs (one-time)...")
        t0 = time.time()
        SW = librosa.stft(W, n_fft=n_fft, hop_length=hop_length)
        SX = librosa.stft(X, n_fft=n_fft, hop_length=hop_length)
        SY = librosa.stft(Y, n_fft=n_fft, hop_length=hop_length)
        SZ = librosa.stft(Z, n_fft=n_fft, hop_length=hop_length)
        n_bins, n_frames = SW.shape
        frame_duration = hop_length / sr
        print(f" STFT frames: {n_frames}, freq bins: {n_bins}, frame dur: {frame_duration:.4f}s")
        print("STFTs done in {:.1f}s".format(time.time() - t0))

        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        A_db = a_weighting_db(freqs)
        A_lin = (10.0 ** (A_db / 10.0))[:, None]

        azimuths = np.radians(np.linspace(0.0, 360.0, az_res, endpoint=False))
        elevations = np.radians(np.linspace(-90.0, 90.0, el_res))

        frames_per_map = max(1, int(round(time_step / frame_duration)))
        n_maps = math.ceil(n_frames / frames_per_map)
        print(f"Building {n_maps} heatmap(s) (time_step={time_step}s; frames_per_map={frames_per_map})")

        heatmaps = []

        # -----------------------
        # 2. Build base heatmaps
        # -----------------------
        for map_idx in tqdm(range(n_maps), desc="time-slices"):
            start = map_idx * frames_per_map
            end = min(start + frames_per_map, n_frames)

            SWs = SW[:, start:end]
            SXs = SX[:, start:end]
            SYs = SY[:, start:end]
            SZs = SZ[:, start:end]

            heat = np.zeros((len(elevations), len(azimuths)), dtype=float)

            for i_phi, phi in enumerate(elevations):
                sinphi = math.sin(phi)
                cosphi = math.cos(phi)
                for j_theta, theta in enumerate(azimuths):
                    cost = math.cos(theta)
                    sint = math.sin(theta)

                    Sdir = (SWs +
                            SXs * (cost * cosphi) +
                            SYs * (sint * cosphi) +
                            SZs * sinphi)

                    P = np.abs(Sdir) ** 2
                    Pw = P * A_lin
                    heat[i_phi, j_theta] = 10.0 * np.log10(np.sum(Pw) + 1e-20)

                    inverse_scale = 20      #if you want the maps to be smaller/more accurate increase this number (10 was too low so I upped it to 20)
                    heat[heat < (heat.max() - heat.max() / inverse_scale)] = 0

            heatmaps.append(heat)

        # -----------------------------------
        # 3. DUPLICATE HEATMAPS TO MATCH FPS
        # -----------------------------------
        repeats = max(1, int(round(video_fps * time_step)))
        print(f"Duplicating each heatmap {repeats}× to match {video_fps} FPS")

        expanded_heatmaps = []
        for h in heatmaps:
            for _ in range(repeats):
                expanded_heatmaps.append(h.copy())

        heatmaps = expanded_heatmaps  # overwrite with expanded version

        # -----------------------
        # 4. NORMALIZATION
        # -----------------------
        if normalize_global:
            all_max = max(np.nanmax(h) for h in heatmaps)
            all_min = min(np.nanmin(h) for h in heatmaps)
        else:
            all_max = all_min = None

        # -----------------------
        # 5. Saving logic
        # -----------------------
        if save_dir is not None and (save_pngs or save_mp4_raw_heatmap or save_mp4_with_graph):
            os.makedirs(save_dir, exist_ok=True)

            # ----- Save RAW PNGs -----
            if save_pngs or save_mp4_raw_heatmap:
                print("Saving raw heatmap images...")
                raw_png_files = []
                for idx, heat in enumerate(tqdm(heatmaps, desc="save-raw-images")):
                    outname = os.path.join(save_dir, f"raw_heat_{idx:04d}.png")
                    vmin, vmax = (all_min, all_max) if all_max else (np.nanmin(heat), np.nanmax(heat))
                    plt.imsave(outname, heat, origin='lower', cmap='inferno', vmin=vmin, vmax=vmax)
                    raw_png_files.append(outname)

                # Create MP4
                if save_mp4_raw_heatmap:
                    try:
                        #mp4_path = os.path.join(save_dir, "raw_heatmaps.mp4")
                        cmd = f"ffmpeg -y -r {video_fps} -i '{save_dir}/raw_heat_%04d.png' -c:v libx264 -pix_fmt yuv420p '{output_path}'"
                        print(f"Running ffmpeg:\n{cmd}")
                        os.system(cmd)
                        print("Saved MP4:", output_path)

                    except Exception as e:
                        print("MP4 export failed:", e)

            # ----- Save GRAPH PNGs or MP4 with graph -----
            if save_pngs or save_mp4_with_graph:
                print("Saving graph-enhanced heatmaps...")
                graph_png_files = []
                for idx, heat in enumerate(tqdm(heatmaps, desc="save-graph-images")):
                    outname = os.path.join(save_dir, f"heat_graph_{idx:04d}.png")
                    plt.figure(figsize=(8, 4))
                    vmin, vmax = (all_min, all_max) if all_max else (np.nanmin(heat), np.nanmax(heat))
                    plt.imshow(heat, origin='lower', aspect='auto', extent=[0, 360, -90, 90],
                            cmap='inferno', vmin=vmin, vmax=vmax)
                    plt.title(f"Heatmap (frame {idx})")
                    plt.colorbar(label="dB(A)")
                    plt.xlabel("Azimuth (deg)")
                    plt.ylabel("Elevation (deg)")
                    plt.tight_layout()
                    plt.savefig(outname, dpi=150)
                    plt.close()
                    graph_png_files.append(outname)

                if save_mp4_with_graph:
                    try:
                        mp4_graph_path = os.path.join(save_dir, "heatmaps_with_graph.mp4")
                        cmd = f"ffmpeg -y -r {video_fps} -i {save_dir}/heat_graph_%04d.png -c:v libx264 -pix_fmt yuv420p {mp4_graph_path}"
                        print(f"Running ffmpeg:\n{cmd}")
                        os.system(cmd)
                        print("Saved MP4 with graph:", mp4_graph_path)

                    except Exception as e:
                        print("Graph MP4 export failed:", e)

        return heatmaps, np.degrees(azimuths), np.degrees(elevations)



    # ---------------------------
    # Main runnable example
    # ---------------------------
    infile = input_audio_path
    if not os.path.exists(infile):
        # Create a dummy ambi.wav if it doesn't exist for testing purposes
        raise IOError(f"{infile} not found.")
        #print(f"{infile} not found. Creating a dummy 4-channel WAV for demonstration.")
        # sr = 44100
        # duration = 5  # seconds
        # t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        # # Simple test signal: a sweep primarily in one direction (e.g., front-left)
        # # W (omni), X (front-back), Y (left-right), Z (up-down)
        # W = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.randn(len(t))  # Base sound
        # X = 0.4 * np.sin(2 * np.pi * 880 * t)  # Front emphasis
        # Y = 0.3 * np.sin(2 * np.pi * 1760 * t)  # Left emphasis
        # Z = 0.05 * np.sin(2 * np.pi * 220 * t)  # Minor vertical component
        # data = np.vstack([W, X, Y, Z]).T
        # sf.write(infile, data.astype(np.float32), sr)

    print("Loading:", infile)
    data, sr = sf.read(infile)
    if data.ndim != 2 or data.shape[1] != 4:
        raise ValueError("Expected a 4-channel ambisonic WAV (W,X,Y,Z).")

    W, X, Y, Z = data.T
    print(f"Loaded {infile} @ {sr} Hz, {data.shape[0]} samples, channels: {data.shape[1]}")

    # ---------- USER CONFIG ----------
    n_fft = 2048
    hop_length = 512
    az_res = 72  # azimuth resolution (try 72 for speed; 360 for full 1°)
    el_res = 36  # elevation resolution (try 36; 180 for full 1°)
    time_step = 1.0  # 1.0 = produce one heatmap-per-second; set 2.0 for every 2 seconds
    save_dir = os.path.join(os.path.dirname(output_path), "heat_outputs")
    save_pngs = False  # Set to True to save individual raw heatmap PNGs
    save_mp4_raw_heatmap = True  # NEW: Set True to save MP4 of raw heatmaps
    save_mp4_with_graph = False  # ORIGINAL: Set True to save MP4 with graph elements
    normalize_global = True
    # -------------------------------

    start_all = time.time()
    heatmaps, az_deg, el_deg = build_heatmaps_over_time(
        W, X, Y, Z, sr,
        n_fft=n_fft, hop_length=hop_length,
        az_res=az_res, el_res=el_res,
        time_step=time_step,
        normalize_global=normalize_global,
        save_dir=save_dir,
        save_pngs=save_pngs,
        save_mp4_raw_heatmap=save_mp4_raw_heatmap,
        save_mp4_with_graph=save_mp4_with_graph
    )
    print(f"Generated {len(heatmaps)} heatmaps in {time.time() - start_all:.1f}s")
    print("Example: heatmaps[0] shape:", heatmaps[0].shape)
    print("Saved outputs to:", save_dir)

    # quick show of the first heatmap (still includes graph elements for quick preview)
    plt.figure(figsize=(10, 4))
    plt.imshow(heatmaps[0], origin='lower', aspect='auto', extent=[0, 360, -90, 90], cmap='inferno')
    plt.colorbar(label="dB(A)")
    plt.title("First heatmap (t=0) - (with graph for preview)")
    plt.xlabel("Azimuth (deg)")
    plt.ylabel("Elevation (deg)")
    plt.tight_layout()
    #plt.show()




