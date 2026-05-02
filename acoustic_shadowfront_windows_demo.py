"""
Acoustic Shadowfront / Nullfront Demo for Windows Laptops
=========================================================

Purpose
-------
Use stereo speakers and a stereo microphone input to test whether an acoustic
"absence event" or tone-boundary appears to traverse the microphone baseline
faster than the local speed of sound.

This script plays a stereo acoustic stimulus and simultaneously records two
microphone channels. It then detects the time at which each microphone sees the
signal appear or disappear, computes the inter-channel time offset, and estimates
an apparent boundary speed:

    v_app = mic_spacing_m / abs(delta_t)

If v_app > speed_of_sound_mps, the measured event boundary is supersonic across
the detector baseline. This does NOT mean sound energy traveled faster than
sound. It means the event/front intersection with the detector manifold traversed
faster than the local wave speed.

Hardware
--------
Minimum:
    - Windows laptop
    - Stereo output device driving left/right speakers
    - Stereo microphone input, or USB stereo audio interface
    - Two microphones separated by a known distance

Recommended:
    - External USB audio interface with 2 input channels
    - Two matched microphones
    - Two speakers separated by 0.5-2 meters
    - Microphones separated by 0.25-1 meter

Install
-------
From PowerShell or Command Prompt:

    py -m pip install numpy sounddevice scipy matplotlib

Run
---
Basic run:

    py acoustic_shadowfront_windows_demo.py --mic-spacing 0.50

List audio devices:

    py acoustic_shadowfront_windows_demo.py --list-devices

Specify devices by index:

    py acoustic_shadowfront_windows_demo.py --output-device 5 --input-device 2 --mic-spacing 0.50

Modes
-----
1. gated_tone
   Plays a tone, then cuts it off. The disappearance boundary is detected.
   This is the most robust first test.

2. panned_edge
   Creates a fast stereo pan/edge between left and right speakers. This is useful
   for generating a clean projected timing difference across two microphones.

3. phase_null
   Plays stereo tones with a programmed phase flip. Depending on speaker/mic
   geometry, this can create an interference/null transition. This is more
   sensitive to room reflections and placement.

Example:

    py acoustic_shadowfront_windows_demo.py --mode gated_tone --mic-spacing 0.50 --frequency 1200

Safety
------
Start with low volume. Sustained tones can be unpleasant. Keep speaker volume
comfortable and avoid high levels.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional

import numpy as np

try:
    import sounddevice as sd
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: sounddevice. Install with:\n"
        "    py -m pip install sounddevice numpy scipy matplotlib"
    ) from exc

try:
    from scipy.signal import butter, filtfilt, hilbert
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: scipy. Install with:\n"
        "    py -m pip install scipy"
    ) from exc

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


@dataclass
class DetectionResult:
    left_time_s: float
    right_time_s: float
    delta_t_s: float
    apparent_speed_mps: float
    mach_number: float
    event_kind: str
    threshold_left: float
    threshold_right: float


def list_devices() -> None:
    print(sd.query_devices())


def speed_of_sound_from_temp(temp_c: float) -> float:
    """Approximate speed of sound in dry air in m/s."""
    return 331.3 + 0.606 * temp_c


def raised_cosine_edge(n: int) -> np.ndarray:
    """Smooth 0..1 transition to avoid speaker clicks."""
    if n <= 1:
        return np.ones(max(n, 1))
    x = np.linspace(0.0, math.pi, n)
    return 0.5 - 0.5 * np.cos(x)


def make_envelope(total_n: int, sr: int, attack_s: float, sustain_s: float, release_s: float) -> np.ndarray:
    attack_n = int(round(attack_s * sr))
    sustain_n = int(round(sustain_s * sr))
    release_n = int(round(release_s * sr))

    env = np.zeros(total_n, dtype=np.float64)
    idx = 0

    if attack_n > 0:
        end = min(idx + attack_n, total_n)
        env[idx:end] = raised_cosine_edge(end - idx)
        idx = end

    if sustain_n > 0 and idx < total_n:
        end = min(idx + sustain_n, total_n)
        env[idx:end] = 1.0
        idx = end

    if release_n > 0 and idx < total_n:
        end = min(idx + release_n, total_n)
        env[idx:end] = raised_cosine_edge(end - idx)[::-1]
        idx = end

    return env


def stimulus_gated_tone(
    sr: int,
    frequency: float,
    duration_s: float,
    amplitude: float,
    attack_s: float,
    sustain_s: float,
    release_s: float,
    stereo_delay_s: float,
) -> Tuple[np.ndarray, str, float]:
    """
    Tone appears, sustains, then disappears. Optional inter-channel delay can
    create a deliberately swept projected edge from the stereo speakers.
    """
    n = int(round(duration_s * sr))
    t = np.arange(n) / sr
    carrier = np.sin(2.0 * np.pi * frequency * t)
    env = make_envelope(n, sr, attack_s, sustain_s, release_s)
    mono = amplitude * carrier * env

    delay_n = int(round(stereo_delay_s * sr))
    left = mono.copy()
    right = np.zeros_like(mono)

    if delay_n >= 0:
        if delay_n < n:
            right[delay_n:] = mono[: n - delay_n]
    else:
        d = abs(delay_n)
        if d < n:
            left[d:] = mono[: n - d]
            right = mono.copy()

    stim = np.column_stack([left, right])
    expected_event_time = attack_s + sustain_s
    return stim, "disappearance", expected_event_time


def stimulus_panned_edge(
    sr: int,
    frequency: float,
    duration_s: float,
    amplitude: float,
    edge_start_s: float,
    edge_duration_s: float,
) -> Tuple[np.ndarray, str, float]:
    """
    A continuous tone whose amplitude weighting moves from left speaker to right
    speaker. This often gives a clear differential crossing between two mics.
    """
    n = int(round(duration_s * sr))
    t = np.arange(n) / sr
    carrier = np.sin(2.0 * np.pi * frequency * t)

    pan = np.zeros(n, dtype=np.float64)
    start = int(round(edge_start_s * sr))
    dur = max(1, int(round(edge_duration_s * sr)))
    end = min(n, start + dur)
    if start > 0:
        pan[:start] = 0.0
    if end > start:
        pan[start:end] = raised_cosine_edge(end - start)
    if end < n:
        pan[end:] = 1.0

    left = amplitude * carrier * np.cos(0.5 * math.pi * pan)
    right = amplitude * carrier * np.sin(0.5 * math.pi * pan)
    stim = np.column_stack([left, right])
    expected_event_time = edge_start_s + 0.5 * edge_duration_s
    return stim, "transition", expected_event_time


def stimulus_phase_null(
    sr: int,
    frequency: float,
    duration_s: float,
    amplitude: float,
    flip_start_s: float,
    flip_duration_s: float,
) -> Tuple[np.ndarray, str, float]:
    """
    Two speakers emit same-frequency tones. The right speaker smoothly phase-flips
    by pi radians. Depending on geometry, this can sweep constructive/destructive
    interference regions across the microphones.
    """
    n = int(round(duration_s * sr))
    t = np.arange(n) / sr

    phase = np.zeros(n, dtype=np.float64)
    start = int(round(flip_start_s * sr))
    dur = max(1, int(round(flip_duration_s * sr)))
    end = min(n, start + dur)
    if end > start:
        phase[start:end] = math.pi * raised_cosine_edge(end - start)
    if end < n:
        phase[end:] = math.pi

    left = amplitude * np.sin(2.0 * np.pi * frequency * t)
    right = amplitude * np.sin(2.0 * np.pi * frequency * t + phase)
    stim = np.column_stack([left, right])
    expected_event_time = flip_start_s + 0.5 * flip_duration_s
    return stim, "transition", expected_event_time


def bandpass(data: np.ndarray, sr: int, frequency: float, bandwidth_hz: float) -> np.ndarray:
    nyq = 0.5 * sr
    low = max(20.0, frequency - 0.5 * bandwidth_hz) / nyq
    high = min(0.95 * nyq, frequency + 0.5 * bandwidth_hz) / nyq
    if not (0.0 < low < high < 1.0):
        return data
    b, a = butter(4, [low, high], btype="band")
    return filtfilt(b, a, data, axis=0)


def envelope_from_recording(recording: np.ndarray, sr: int, frequency: float, bandwidth_hz: float) -> np.ndarray:
    filtered = bandpass(recording, sr, frequency, bandwidth_hz)
    analytic = hilbert(filtered, axis=0)
    env = np.abs(analytic)

    # Smooth envelope over about 5 ms.
    win = max(3, int(round(0.005 * sr)))
    kernel = np.ones(win, dtype=np.float64) / win
    smoothed = np.zeros_like(env)
    for ch in range(env.shape[1]):
        smoothed[:, ch] = np.convolve(env[:, ch], kernel, mode="same")
    return smoothed


def first_crossing_time(
    env: np.ndarray,
    sr: int,
    expected_event_time: float,
    event_kind: str,
    search_window_s: float,
    threshold_fraction: float,
) -> Tuple[float, float]:
    """Find threshold crossing near expected event time for one channel."""
    n = len(env)
    center = int(round(expected_event_time * sr))
    half = int(round(0.5 * search_window_s * sr))
    lo = max(0, center - half)
    hi = min(n, center + half)

    if hi <= lo + 4:
        raise ValueError("Search window too small or outside recording.")

    segment = env[lo:hi]
    pre_lo = max(0, lo - int(round(0.25 * sr)))
    pre_hi = lo
    post_lo = hi
    post_hi = min(n, hi + int(round(0.25 * sr)))

    high_level = np.percentile(env[pre_lo:pre_hi] if pre_hi > pre_lo else segment, 90)
    low_level = np.percentile(env[post_lo:post_hi] if post_hi > post_lo else segment, 10)

    # For transition modes, use robust high/low estimate inside the search window.
    if event_kind == "transition":
        high_level = np.percentile(segment, 85)
        low_level = np.percentile(segment, 15)

    threshold = low_level + threshold_fraction * (high_level - low_level)

    if event_kind == "disappearance":
        idxs = np.where(segment < threshold)[0]
    else:
        # Find the largest local slope event by threshold crossing either direction.
        above = segment > threshold
        changes = np.where(np.diff(above.astype(int)) != 0)[0]
        idxs = changes + 1

    if len(idxs) == 0:
        # Fallback: choose steepest slope magnitude.
        slope = np.abs(np.gradient(segment))
        idx = int(np.argmax(slope))
    else:
        # Prefer crossing closest to expected center.
        idx = int(idxs[np.argmin(np.abs((lo + idxs) - center))])

    # Parabolic-ish refinement using linear interpolation around threshold.
    global_idx = lo + idx
    if 1 <= global_idx < n:
        y0 = env[global_idx - 1]
        y1 = env[global_idx]
        if y1 != y0:
            frac = float((threshold - y0) / (y1 - y0))
            if 0.0 <= frac <= 1.0:
                return (global_idx - 1 + frac) / sr, float(threshold)

    return global_idx / sr, float(threshold)


def detect_event_times(
    recording: np.ndarray,
    sr: int,
    frequency: float,
    bandwidth_hz: float,
    expected_event_time: float,
    event_kind: str,
    mic_spacing_m: float,
    speed_of_sound_mps: float,
    search_window_s: float,
    threshold_fraction: float,
) -> Tuple[DetectionResult, np.ndarray]:
    if recording.ndim != 2 or recording.shape[1] < 2:
        raise ValueError("Recording must have at least two channels.")

    recording = recording[:, :2]
    env = envelope_from_recording(recording, sr, frequency, bandwidth_hz)

    left_t, left_thr = first_crossing_time(
        env[:, 0], sr, expected_event_time, event_kind, search_window_s, threshold_fraction
    )
    right_t, right_thr = first_crossing_time(
        env[:, 1], sr, expected_event_time, event_kind, search_window_s, threshold_fraction
    )

    delta_t = right_t - left_t
    if abs(delta_t) < 1e-9:
        apparent_speed = float("inf")
        mach = float("inf")
    else:
        apparent_speed = mic_spacing_m / abs(delta_t)
        mach = apparent_speed / speed_of_sound_mps

    return DetectionResult(
        left_time_s=left_t,
        right_time_s=right_t,
        delta_t_s=delta_t,
        apparent_speed_mps=apparent_speed,
        mach_number=mach,
        event_kind=event_kind,
        threshold_left=left_thr,
        threshold_right=right_thr,
    ), env


def play_and_record(
    stim: np.ndarray,
    sr: int,
    input_device: Optional[int],
    output_device: Optional[int],
    input_channels: int,
    pre_roll_s: float,
    post_roll_s: float,
) -> np.ndarray:
    pre = np.zeros((int(round(pre_roll_s * sr)), 2), dtype=np.float32)
    post = np.zeros((int(round(post_roll_s * sr)), 2), dtype=np.float32)
    playback = np.vstack([pre, stim.astype(np.float32), post])

    rec_channels = max(2, input_channels)
    device = None
    if input_device is not None or output_device is not None:
        device = (input_device, output_device)

    print("Recording and playing stimulus...")
    recording = sd.playrec(
        playback,
        samplerate=sr,
        channels=rec_channels,
        dtype="float32",
        device=device,
        blocking=True,
    )
    return np.asarray(recording[:, :2], dtype=np.float64)


def save_plot(
    path: Path,
    recording: np.ndarray,
    env: np.ndarray,
    sr: int,
    result: DetectionResult,
) -> None:
    if plt is None:
        print("matplotlib not installed; skipping plot.")
        return

    t = np.arange(recording.shape[0]) / sr

    fig = plt.figure(figsize=(12, 7))

    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(t, recording[:, 0], label="Mic L raw", linewidth=0.8)
    ax1.plot(t, recording[:, 1], label="Mic R raw", linewidth=0.8, alpha=0.8)
    ax1.axvline(result.left_time_s, linestyle="--", label="L event")
    ax1.axvline(result.right_time_s, linestyle=":", label="R event")
    ax1.set_title("Recorded waveform")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")
    ax1.legend(loc="upper right")

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(t, env[:, 0], label="Mic L envelope", linewidth=1.0)
    ax2.plot(t, env[:, 1], label="Mic R envelope", linewidth=1.0)
    ax2.axhline(result.threshold_left, linestyle="--", linewidth=0.8, label="L threshold")
    ax2.axhline(result.threshold_right, linestyle=":", linewidth=0.8, label="R threshold")
    ax2.axvline(result.left_time_s, linestyle="--")
    ax2.axvline(result.right_time_s, linestyle=":")
    ax2.set_title("Bandpassed amplitude envelope and detected event times")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Envelope")
    ax2.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"Saved plot: {path}")


def build_stimulus(args: argparse.Namespace) -> Tuple[np.ndarray, str, float]:
    if args.mode == "gated_tone":
        return stimulus_gated_tone(
            sr=args.sample_rate,
            frequency=args.frequency,
            duration_s=args.duration,
            amplitude=args.amplitude,
            attack_s=args.attack,
            sustain_s=args.sustain,
            release_s=args.release,
            stereo_delay_s=args.stereo_delay,
        )

    if args.mode == "panned_edge":
        return stimulus_panned_edge(
            sr=args.sample_rate,
            frequency=args.frequency,
            duration_s=args.duration,
            amplitude=args.amplitude,
            edge_start_s=args.edge_start,
            edge_duration_s=args.edge_duration,
        )

    if args.mode == "phase_null":
        return stimulus_phase_null(
            sr=args.sample_rate,
            frequency=args.frequency,
            duration_s=args.duration,
            amplitude=args.amplitude,
            flip_start_s=args.edge_start,
            flip_duration_s=args.edge_duration,
        )

    raise ValueError(f"Unknown mode: {args.mode}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Windows stereo speaker/microphone acoustic shadowfront demo."
    )
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit.")
    parser.add_argument("--input-device", type=int, default=None, help="Input device index from --list-devices.")
    parser.add_argument("--output-device", type=int, default=None, help="Output device index from --list-devices.")
    parser.add_argument("--input-channels", type=int, default=2, help="Number of input channels to request.")

    parser.add_argument("--mode", choices=["gated_tone", "panned_edge", "phase_null"], default="gated_tone")
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument("--frequency", type=float, default=1200.0, help="Tone frequency in Hz.")
    parser.add_argument("--bandwidth", type=float, default=400.0, help="Bandpass width around tone frequency in Hz.")
    parser.add_argument("--duration", type=float, default=2.0, help="Stimulus duration in seconds, excluding pre/post roll.")
    parser.add_argument("--amplitude", type=float, default=0.15, help="Playback amplitude, 0.0 to 1.0. Start low.")

    parser.add_argument("--attack", type=float, default=0.15, help="Gated-tone attack duration in seconds.")
    parser.add_argument("--sustain", type=float, default=0.85, help="Gated-tone sustain duration in seconds.")
    parser.add_argument("--release", type=float, default=0.01, help="Gated-tone release duration in seconds.")
    parser.add_argument(
        "--stereo-delay",
        type=float,
        default=0.000,
        help="Optional delay between L/R speaker channels in seconds for gated_tone.",
    )

    parser.add_argument("--edge-start", type=float, default=0.8, help="Start time for panned/phase transition.")
    parser.add_argument("--edge-duration", type=float, default=0.02, help="Duration of panned/phase transition.")

    parser.add_argument("--pre-roll", type=float, default=0.25)
    parser.add_argument("--post-roll", type=float, default=0.50)
    parser.add_argument("--mic-spacing", type=float, required=False, default=0.50, help="Distance between microphones in meters.")
    parser.add_argument("--temperature-c", type=float, default=20.0)
    parser.add_argument("--search-window", type=float, default=0.50, help="Seconds around expected event to search.")
    parser.add_argument("--threshold-fraction", type=float, default=0.50, help="Envelope threshold fraction between low/high.")
    parser.add_argument("--plot", type=str, default="acoustic_shadowfront_result.png")
    parser.add_argument("--save-npz", type=str, default="acoustic_shadowfront_recording.npz")

    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    if args.list_devices:
        list_devices()
        return 0

    if not (0.0 < args.amplitude <= 1.0):
        raise SystemExit("--amplitude must be between 0 and 1. Start around 0.10 to 0.20.")

    c_s = speed_of_sound_from_temp(args.temperature_c)
    print(f"Using speed of sound estimate: {c_s:.2f} m/s at {args.temperature_c:.1f} C")
    print(f"Microphone spacing: {args.mic_spacing:.4f} m")
    print(f"Mode: {args.mode}")

    stim, event_kind, expected_event_time = build_stimulus(args)
    expected_event_time_with_preroll = args.pre_roll + expected_event_time

    recording = play_and_record(
        stim=stim,
        sr=args.sample_rate,
        input_device=args.input_device,
        output_device=args.output_device,
        input_channels=args.input_channels,
        pre_roll_s=args.pre_roll,
        post_roll_s=args.post_roll,
    )

    result, env = detect_event_times(
        recording=recording,
        sr=args.sample_rate,
        frequency=args.frequency,
        bandwidth_hz=args.bandwidth,
        expected_event_time=expected_event_time_with_preroll,
        event_kind=event_kind,
        mic_spacing_m=args.mic_spacing,
        speed_of_sound_mps=c_s,
        search_window_s=args.search_window,
        threshold_fraction=args.threshold_fraction,
    )

    print("\n=== Acoustic Shadowfront Result ===")
    print(f"Detected event kind:      {result.event_kind}")
    print(f"Left mic event time:      {result.left_time_s:.9f} s")
    print(f"Right mic event time:     {result.right_time_s:.9f} s")
    print(f"Delta t, R - L:           {result.delta_t_s * 1e6:.3f} microseconds")
    print(f"Apparent boundary speed:  {result.apparent_speed_mps:.3f} m/s")
    print(f"Mach number vs sound:     {result.mach_number:.3f}")

    if result.mach_number > 1.0:
        print("\nInterpretation: apparent event-boundary traversal is SUPERSONIC across the mic baseline.")
        print("This supports the projected-front/absence-front concept, not faster-than-sound signaling.")
    else:
        print("\nInterpretation: apparent event-boundary traversal is subsonic for this geometry/run.")
        print("Try changing speaker angle, mic spacing, stereo delay, or mode.")

    np.savez(
        args.save_npz,
        recording=recording,
        envelope=env,
        sample_rate=args.sample_rate,
        mic_spacing_m=args.mic_spacing,
        speed_of_sound_mps=c_s,
        left_time_s=result.left_time_s,
        right_time_s=result.right_time_s,
        delta_t_s=result.delta_t_s,
        apparent_speed_mps=result.apparent_speed_mps,
        mach_number=result.mach_number,
        mode=args.mode,
        frequency=args.frequency,
    )
    print(f"Saved data: {args.save_npz}")

    if args.plot:
        save_plot(Path(args.plot), recording, env, args.sample_rate, result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
