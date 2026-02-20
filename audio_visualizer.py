"""
Real-time Audio Visualizer — WMP-style geometric visualization.
Renderer: tkinter Canvas (works on macOS from any terminal, no pygame needed)

Dependencies:
    pip install sounddevice soundfile numpy
    (tkinter is built-in with Python)

Usage:
    python audio_visualizer.py                      # opens file picker dialog
    python audio_visualizer.py path/to/audio.wav    # loads file directly
    Supports: .wav  .flac  .ogg  .aiff  (not .mp3)
"""

import colorsys
import math
import queue
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
import sounddevice as sd
import soundfile as sf

# ── Constants ──────────────────────────────────────────────────────────────
WIDTH, HEIGHT = 900, 700
CTRL_H = 52              # height of the control bar below the canvas
FRAME_MS = 16            # ~60 fps
CHUNK = 2048             # audio samples per callback
N_BARS = 120             # radial frequency bars
INNER_R = 90             # inner circle radius (px)
MAX_BAR = 180            # max bar length (px)
SYMMETRY = 3             # kaleidoscope fold count
N_PARTICLES = 50         # orbiting particles
N_STARS = 120            # background star count
SMOOTH = 0.75            # IIR smoothing coefficient
FADE_FRAMES = 1024       # samples over which to fade in/out (~23 ms at 44 kHz)
# ──────────────────────────────────────────────────────────────────────────


def hsv_hex(h: float, s: float, v: float) -> str:
    """Return a tkinter-compatible '#rrggbb' color from HSV (all 0–1)."""
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"


class AudioStreamer:
    """
    Plays an audio file through a single always-running OutputStream.
    Pause/resume are handled entirely inside the callback via a state machine
    so the hardware stream is never interrupted — eliminating all glitches.

    States: 'playing' → 'fading_out' → 'paused' → 'fading_in' → 'playing'
    """

    _FADE_SAMPLES = FADE_FRAMES  # samples for each fade ramp

    def __init__(self, path: str):
        self.q: queue.Queue = queue.Queue(maxsize=30)
        self.done = threading.Event()
        self._data, self._sr = sf.read(path, dtype="float32", always_2d=True)
        self._mono = self._data.mean(axis=1) if self._data.shape[1] > 1 else self._data[:, 0]
        self._pos = 0
        self._nch = self._data.shape[1]

        # State machine — only written from main thread, read in callback
        self._state = "playing"   # playing | fading_out | paused | fading_in
        self._fade_pos = 0        # samples elapsed in current fade

    def _callback(self, outdata, frames, time_info, status):
        """Audio callback — runs on a real-time thread, no blocking allowed."""
        remaining = len(self._data) - self._pos

        if self._state == "paused":
            outdata[:] = 0
            return

        if self._state == "playing":
            if remaining <= 0:
                outdata[:] = 0
                self.done.set()
                raise sd.CallbackStop
            n = min(frames, remaining)
            outdata[:n] = self._data[self._pos:self._pos + n]
            if n < frames:
                outdata[n:] = 0
            try:
                self.q.put_nowait(self._mono[self._pos:self._pos + n].copy())
            except queue.Full:
                pass
            self._pos += n
            return

        if self._state == "fading_out":
            # Fill with real audio first, then apply fade-out ramp
            n = min(frames, max(0, remaining))
            if n:
                outdata[:n] = self._data[self._pos:self._pos + n]
                self._pos += n
            if n < frames:
                outdata[n:] = 0

            # Ramp: goes from where we are in the fade down to 0
            start_gain = 1.0 - self._fade_pos / self._FADE_SAMPLES
            end_gain   = 1.0 - (self._fade_pos + frames) / self._FADE_SAMPLES
            start_gain = max(0.0, start_gain)
            end_gain   = max(0.0, end_gain)
            ramp = np.linspace(start_gain, end_gain, frames,
                               dtype=np.float32).reshape(-1, 1)
            outdata[:] *= ramp
            self._fade_pos += frames
            if self._fade_pos >= self._FADE_SAMPLES:
                outdata[:] = 0
                self._fade_pos = 0
                self._state = "paused"
            return

        if self._state == "fading_in":
            n = min(frames, max(0, remaining))
            if n:
                outdata[:n] = self._data[self._pos:self._pos + n]
                self._pos += n
            if n < frames:
                outdata[n:] = 0

            start_gain = self._fade_pos / self._FADE_SAMPLES
            end_gain   = (self._fade_pos + frames) / self._FADE_SAMPLES
            start_gain = min(1.0, start_gain)
            end_gain   = min(1.0, end_gain)
            ramp = np.linspace(start_gain, end_gain, frames,
                               dtype=np.float32).reshape(-1, 1)
            outdata[:] *= ramp
            self._fade_pos += frames
            if self._fade_pos >= self._FADE_SAMPLES:
                self._fade_pos = 0
                self._state = "playing"
            return

    def start(self):
        self._stream = sd.OutputStream(
            samplerate=self._sr,
            channels=self._nch,
            blocksize=CHUNK,
            dtype="float32",
            callback=self._callback,
            finished_callback=lambda: None,  # done.set() handled in callback
        )
        self._stream.start()

    def pause(self):
        if self._state == "playing":
            self._fade_pos = 0
            self._state = "fading_out"

    def resume(self):
        if self._state == "paused":
            self._fade_pos = 0
            self._state = "fading_in"

    def stop(self):
        if hasattr(self, "_stream"):
            self._stream.stop()
            self._stream.close()


class VisualizerApp:
    def __init__(self, root: tk.Tk, file_path: str):
        self.root = root
        self.file_path = file_path
        self.hue = 0.0
        self.fft_smooth = np.zeros(N_BARS)
        self.waveform = np.zeros(256)
        self.cx = WIDTH // 2
        self.cy = HEIGHT // 2
        self.paused = False
        self.angle_offset = 0.0     # slow global rotation
        self.beat_flash = 0.0       # decays each frame, spikes on bass hit
        self.prev_bass = 0.0

        # Pre-generate fixed star positions
        rng = np.random.default_rng(7)
        self._stars = list(zip(
            rng.integers(0, WIDTH,  N_STARS).tolist(),
            rng.integers(0, HEIGHT, N_STARS).tolist(),
            rng.uniform(0, 1, N_STARS).tolist(),
            rng.uniform(0.2, 0.6, N_STARS).tolist(),
        ))

        # Pre-generate orbiting particles
        rng2 = np.random.default_rng(42)
        self._particles = [
            {
                "angle":  rng2.uniform(0, 2 * math.pi),
                "radius": rng2.uniform(INNER_R + MAX_BAR * 0.55,
                                       INNER_R + MAX_BAR * 0.92),
                "speed":  rng2.uniform(0.005, 0.018)
                          * rng2.choice([-1, 1]),
                "hoff":   rng2.uniform(0, 1),
                "size":   int(rng2.integers(2, 6)),
            }
            for _ in range(N_PARTICLES)
        ]

        root.title("Audio Visualizer")
        root.resizable(False, False)
        root.configure(bg="black")

        # ── Visualizer canvas ────────────────────────────────────────────
        self.canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT,
                                bg="black", highlightthickness=0)
        self.canvas.pack()

        # ── Control bar ──────────────────────────────────────────────────
        ctrl = tk.Frame(root, bg="#111111", height=CTRL_H)
        ctrl.pack(fill=tk.X)
        ctrl.pack_propagate(False)

        self.btn = tk.Button(
            ctrl, text="⏸  Pause",
            command=self._toggle_pause,
            bg="#222222", fg="white",
            activebackground="#333333", activeforeground="white",
            relief=tk.FLAT, font=("Arial", 13, "bold"),
            padx=18, pady=6, cursor="hand2",
        )
        self.btn.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        root.bind("<Escape>", lambda e: self._quit())
        root.bind("<space>", lambda e: self._toggle_pause())
        root.protocol("WM_DELETE_WINDOW", self._quit)

        # Start audio
        self.streamer = AudioStreamer(file_path)
        self.streamer.start()

        # Kick off animation loop
        self.root.after(FRAME_MS, self._frame)

    def _quit(self):
        self.streamer.stop()
        self.root.destroy()

    def _toggle_pause(self):
        if self.paused:
            self.streamer.resume()
            self.btn.config(text="⏸  Pause")
            self.paused = False
            self.root.after(FRAME_MS, self._frame)   # restart animation loop
        else:
            self.streamer.pause()
            self.btn.config(text="▶  Play")
            self.paused = True

    def _frame(self):
        # Pull latest audio from queue
        latest = None
        while not self.streamer.q.empty():
            try:
                latest = self.streamer.q.get_nowait()
            except queue.Empty:
                break

        if latest is not None and len(latest) > 0:
            # FFT → N_BARS linearly binned
            win = np.hanning(len(latest))
            spectrum = np.abs(np.fft.rfft(latest * win))
            edges = np.linspace(0, len(spectrum) - 1, N_BARS + 1).astype(int)
            binned = np.array([spectrum[edges[i]:edges[i+1]].mean()
                               for i in range(N_BARS)])
            binned /= binned.max() + 1e-9
            self.fft_smooth = SMOOTH * self.fft_smooth + (1 - SMOOTH) * binned

            # Waveform (256 pts)
            wlen = min(len(latest), 1024)
            self.waveform = np.interp(
                np.linspace(0, wlen - 1, 256),
                np.arange(wlen),
                latest[:wlen],
            )

        self._draw()
        self.hue = (self.hue + 0.005) % 1.0
        self.angle_offset = (self.angle_offset + 0.009) % (2 * math.pi)

        # Beat detection — spike flash on sudden bass hit
        bass_now = float(np.mean(self.fft_smooth[:6]))
        if bass_now - self.prev_bass > 0.12:
            self.beat_flash = min(1.0, self.beat_flash + 0.7)
        self.prev_bass = bass_now
        self.beat_flash *= 0.80   # decay

        # Advance particle orbits
        for p in self._particles:
            p["angle"] = (p["angle"] + p["speed"]) % (2 * math.pi)

        # Reschedule unless paused or audio is finished
        if self.paused:
            return
        if not (self.streamer.done.is_set() and self.streamer.q.empty()):
            self.root.after(FRAME_MS, self._frame)
        else:
            self.root.after(800, self._quit)

    def _draw_star(self, canvas, cx, cy, r, points=5, angle=0.0, color="white"):
        """Draw a filled star polygon."""
        pts = []
        inner_r = r * 0.42
        for i in range(points * 2):
            a = math.pi * i / points + angle - math.pi / 2
            rad = r if i % 2 == 0 else inner_r
            pts.extend([cx + rad * math.cos(a), cy + rad * math.sin(a)])
        if len(pts) >= 6:
            canvas.create_polygon(pts, fill=color, outline="", smooth=False)

    def _draw(self):
        canvas = self.canvas
        canvas.delete("all")
        cx, cy = self.cx, self.cy
        hue = self.hue
        ao = self.angle_offset

        # ── Beat-flash background ─────────────────────────────────────
        if self.beat_flash > 0.04:
            flash = hsv_hex(hue, 1.0, self.beat_flash * 0.28)
            canvas.create_rectangle(0, 0, WIDTH, HEIGHT, fill=flash, outline="")
        else:
            canvas.create_rectangle(0, 0, WIDTH, HEIGHT, fill="black", outline="")

        # ── Background star field ──────────────────────────────────────
        for sx, sy, sh, sv in self._stars:
            c = hsv_hex((hue + sh) % 1.0, 0.6, sv + self.beat_flash * 0.3)
            canvas.create_oval(sx - 1, sy - 1, sx + 1, sy + 1,
                               fill=c, outline="")

        # ── Spinning outer hexagon ─────────────────────────────────────
        treble = float(np.mean(self.fft_smooth[-20:]))
        for ring_i, (sides, extra_r, lw) in enumerate([
            (6, 55 + treble * 50, 2),
            (8, 35 + treble * 30, 1),
        ]):
            poly_pts = []
            for i in range(sides):
                a = 2 * math.pi * i / sides + ao * (1 + ring_i * 0.5)
                pr = INNER_R + MAX_BAR + extra_r
                poly_pts.extend([cx + pr * math.cos(a), cy + pr * math.sin(a)])
            pc = hsv_hex((hue + 0.25 + ring_i * 0.15) % 1.0, 1.0, 0.95)
            canvas.create_polygon(poly_pts, outline=pc, fill="",
                                  width=lw, smooth=False)

        # ── Kaleidoscope radial bars (SYMMETRY-fold) ───────────────────
        for sym in range(SYMMETRY):
            sym_rot = 2 * math.pi * sym / SYMMETRY
            for i, val in enumerate(self.fft_smooth):
                bar_len = int(val * MAX_BAR) + 2
                angle = (2 * math.pi * i / N_BARS
                         - math.pi / 2 + sym_rot + ao * 0.25)
                # Each symmetry arm gets a hue offset → rainbow kaleidoscope
                h = (hue + i / N_BARS * 0.8 + sym / SYMMETRY * 0.35) % 1.0
                sat = 0.75 + val * 0.25
                color = hsv_hex(h, sat, 1.0)
                cos_a, sin_a = math.cos(angle), math.sin(angle)
                # Outer bar
                x1 = cx + INNER_R * cos_a
                y1 = cy + INNER_R * sin_a
                x2 = cx + (INNER_R + bar_len) * cos_a
                y2 = cy + (INNER_R + bar_len) * sin_a
                # Inner mirror bar (shorter)
                inward = min(bar_len * 0.35, INNER_R - 12)
                x3 = cx + (INNER_R - inward) * cos_a
                y3 = cy + (INNER_R - inward) * sin_a
                w = max(1, int(val * 5))
                canvas.create_line(x1, y1, x2, y2, fill=color,
                                   width=w, capstyle=tk.ROUND)
                canvas.create_line(x1, y1, x3, y3, fill=color,
                                   width=max(1, w - 1), capstyle=tk.ROUND)

        # ── Multiple waveform rings ────────────────────────────────────
        n = len(self.waveform)
        for ridx, (scale, hoff, amp, rot_speed) in enumerate([
            (0.55, 0.00, 35, 0.15),
            (0.75, 0.33, 50, -0.25),
            (0.95, 0.66, 65,  0.40),
        ]):
            ring_r = max(18, (INNER_R - 24) * scale + ridx * 8)
            wc = hsv_hex((hue + hoff) % 1.0, 1.0, 0.95)
            pts = []
            for i, s in enumerate(self.waveform):
                a = 2 * math.pi * i / n - math.pi / 2 + ao * rot_speed
                r = ring_r + s * amp
                pts.extend([cx + r * math.cos(a), cy + r * math.sin(a)])
            if len(pts) >= 6:
                canvas.create_polygon(pts, outline=wc, fill="",
                                      smooth=True, width=max(1, 2 - ridx // 2))

        # ── Orbiting particles ─────────────────────────────────────────
        mid = float(np.mean(self.fft_smooth[8:35]))
        for p in self._particles:
            r = p["radius"] + mid * 55
            x = cx + r * math.cos(p["angle"])
            y = cy + r * math.sin(p["angle"])
            pc = hsv_hex((hue + p["hoff"]) % 1.0, 1.0, 1.0)
            sz = p["size"] + int(mid * 7)
            canvas.create_oval(x - sz, y - sz, x + sz, y + sz,
                               fill=pc, outline="")
            # Comet tail
            tail_a = p["angle"] - math.copysign(0.12, p["speed"])
            tx = cx + (r - 8) * math.cos(tail_a)
            ty = cy + (r - 8) * math.sin(tail_a)
            tc = hsv_hex((hue + p["hoff"] + 0.1) % 1.0, 0.6, 0.6)
            canvas.create_line(x, y, tx, ty, fill=tc, width=max(1, sz - 1),
                               capstyle=tk.ROUND)

        # ── Center orb ────────────────────────────────────────────────
        bass = float(np.mean(self.fft_smooth[:6]))
        orb_r = int(INNER_R * 0.50 + bass * 48)
        # Layered glow with shifting hues
        for i in range(7, 0, -1):
            gr = orb_r + i * 10
            gh = (hue + i * 0.07) % 1.0
            gv = 0.04 + 0.14 * (i / 7) + self.beat_flash * 0.15
            canvas.create_oval(cx - gr, cy - gr, cx + gr, cy + gr,
                               fill=hsv_hex(gh, 1.0, gv), outline="")
        orb_color = hsv_hex((hue + 0.05) % 1.0, 0.85, 1.0)
        canvas.create_oval(cx - orb_r, cy - orb_r, cx + orb_r, cy + orb_r,
                           fill=orb_color, outline="")
        # Spinning 5-pointed star inside orb
        star_r = max(8, orb_r - 8)
        self._draw_star(canvas, cx, cy, star_r, points=5,
                        angle=ao * 3.5,
                        color=hsv_hex((hue + 0.55) % 1.0, 0.5, 1.0))
        # Bright core dot
        cr = max(4, orb_r - 22)
        canvas.create_oval(cx - cr, cy - cr, cx + cr, cy + cr,
                           fill="white", outline="")

        # ── HUD ───────────────────────────────────────────────────────
        fname = self.file_path.split("/")[-1]
        canvas.create_text(10, HEIGHT - 14, anchor="sw", text=fname,
                           fill="#aaaaaa", font=("Arial", 12))
        hint = "⏸ Space to pause  ·  ESC to quit"
        canvas.create_text(WIDTH - 10, HEIGHT - 14, anchor="se",
                           text=hint, fill="#555555", font=("Arial", 12))


def main():
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        # Use tkinter for file selection — no hidden dialog issues
        picker = tk.Tk()
        picker.withdraw()
        picker.attributes("-topmost", True)
        picker.update()
        file_path = filedialog.askopenfilename(
            parent=picker,
            title="Select an audio file",
            filetypes=[("Audio Files", "*.wav *.flac *.ogg *.aiff"), ("All Files", "*.*")],
        )
        picker.destroy()

    if not file_path:
        print("No file selected.")
        return

    print(f"Loading: {file_path}")
    try:
        # Quick probe to catch unsupported formats early
        sf.info(file_path)
    except Exception as e:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Unsupported file",
                             f"Could not open audio file:\n{e}\n\n"
                             "Supported formats: WAV, FLAC, OGG, AIFF\n"
                             "(MP3 is not supported — convert with ffmpeg first)")
        root.destroy()
        return

    root = tk.Tk()
    app = VisualizerApp(root, file_path)
    root.mainloop()


if __name__ == "__main__":
    main()
