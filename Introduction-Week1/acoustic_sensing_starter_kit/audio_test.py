#!/usr/bin/env python3
"""
Audio Setup Test for Acoustic Sensing Kit
Run this before collecting training data to verify your audio setup.
"""
import numpy as np
import time
from jacktools.jacksignal import JackSignal
from matplotlib import pyplot as plt


def test_audio_setup():
    print("🔊 Testing Audio Setup for Acoustic Sensing")
    print("=" * 50)

    # Test JACK connection
    try:
        J = JackSignal("TestJS")
        print("✅ JACK connection successful")
        name, sr, period = J.get_jack_info()
        print(f"   Sample rate: {sr} Hz")
        print(f"   Period size: {period} samples")
    except Exception as e:
        print(f"❌ JACK connection failed: {e}")
        print("   Make sure QjackCtl is running and JACK server is started")
        return False

        # Test audio I/O
    try:
        # Create test signal (short beep)
        duration = 0.1  # 100ms
        t = np.linspace(0, duration, int(sr * duration), False)
        test_signal = 0.3 * np.sin(2 * np.pi * 1000 * t).astype(
            np.float32
        )  # 1kHz tone, ensure float32

        # Setup channels
        J.create_output(0, "test_out")
        J.create_input(0, "test_in")
        J.connect_input(0, "system:capture_1")
        J.connect_output(0, "system:playback_1")

        # Record background noise first
        print("🎤 Recording background noise (2 seconds)...")
        background = np.zeros(int(sr * 2), dtype=np.float32)
        J.set_input_data(0, background)
        J.process()
        J.wait()

        # Play test signal and record
        print("🔊 Playing test signal and recording...")
        recorded = np.zeros_like(test_signal, dtype=np.float32)
        J.set_output_data(0, test_signal)
        J.set_input_data(0, recorded)
        J.process()
        J.wait()

        # Analyze recording
        background_rms = np.sqrt(np.mean(background**2))
        signal_rms = np.sqrt(np.mean(recorded**2))

        print(".4f")
        print(".4f")

        if signal_rms > background_rms * 2:
            print("✅ Microphone is detecting audio signals")
        else:
            print("⚠️  Microphone might not be working properly")
            print("   Check microphone connection and levels in QjackCtl")

        # Plot results
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.plot(test_signal)
        plt.title("Test Signal Sent")
        plt.xlabel("Samples")
        plt.ylabel("Amplitude")

        plt.subplot(1, 3, 2)
        plt.plot(recorded)
        plt.title("Signal Recorded")
        plt.xlabel("Samples")
        plt.ylabel("Amplitude")

        plt.subplot(1, 3, 3)
        # Simple spectrum analysis
        from scipy import signal

        freqs, psd = signal.welch(recorded, fs=sr, nperseg=1024)
        plt.semilogy(freqs, psd)
        plt.title("Frequency Spectrum")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power")
        plt.xlim(20, 20000)

        plt.tight_layout()
        plt.show()

        J.cleanup()
        return True

    except Exception as e:
        print(f"❌ Audio test failed: {e}")
        return False


if __name__ == "__main__":
    success = test_audio_setup()
    if success:
        print("\n🎉 Audio setup looks good! You can now run A_record.py")
        print("💡 Tips for data collection:")
        print("   - Tap firmly but consistently on each location")
        print("   - Keep the same tapping force and finger")
        print("   - Make sure the laptop surface is clean")
        print("   - Record in a quiet environment")
    else:
        print("\n❌ Please fix the audio setup issues before collecting data")
