#!/usr/bin/env python3
"""Record a full policy run in the Screeps client as a 2K video.

Method, and the measurements behind it:

- **CDP screencast, not X11 capture.** `Xvfb` plus `x11grab` was tried first and
  captures pure black: with no window manager on the virtual display chromium's
  window is never mapped, so the grab only ever sees the root window. That was
  verified down to a plain red HTML page under three GL backends
  (`swiftshader`, `--disable-gpu`, default) - all black, while a DevTools
  screenshot of the same page renders correctly. So frames are taken from the
  compositor, which is the path that demonstrably works, and the browser runs
  genuinely headless.
- **Constant output frame rate.** A screencast only emits frames when the page
  repaints, so a pacer writes the most recent frame into ffmpeg on a fixed
  schedule and repeats it when nothing new arrived. Video time therefore tracks
  wall time without post-hoc re-timestamping.
- **`h264_nvenc`**, so encoding runs on a separate ASIC and leaves the cores to
  the simulator and the shader cores to the policy.
- **HUD burned in afterwards from the run log.** The tick counter, population
  and economy figures are the trainer's own telemetry, so the numbers on screen
  cannot disagree with the run that produced them.
- **Framing is checked before recording**, on a real captured frame, and the run
  aborts rather than produce a blank or badly framed video.

Outputs: `<out>.mp4` (full run, HUD) and `<out>_<speed>x.mp4` (time lapse),
both 2560x1440.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

_RL_ROOT = Path(__file__).resolve().parents[1]
_REPO = _RL_ROOT.parents[1]

HUD_RE = re.compile(
    r"\[t=\s*(\d+)\]\s+r=([+\-\d.]+)\s+ep_ret=([\d.]+)\s+V=([\d.\-]+)\s+"
    r"harvest=(\d+)\s+control=(\d+)\s+creeps=(\d+)\s+storeE=(\d+)\s+rcl=(\d+)\s+ctrl=(\d+)"
)
URL_RE = re.compile(r"client URL:\s*(\S+)")
FINISHED_RE = re.compile(r"^\[watch\] finished ")


@dataclass
class HudSample:
    wall: float
    tick: int
    harvest: int
    control: int
    creeps: int
    rcl: int
    progress: int
    ep_ret: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="output prefix, no suffix")
    p.add_argument("--ticks", type=int, default=20000)
    p.add_argument(
        "--tick-ms", type=int, default=45,
        help="extra delay per tick; a tick costs compute plus this",
    )
    p.add_argument("--room", type=str, default="W7N3")
    p.add_argument("--curriculum", type=str, default="empty")
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--width", type=int, default=2560)
    p.add_argument("--height", type=int, default=1440)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--jpeg-quality", type=int, default=92)
    p.add_argument("--cdp-port", type=int, default=9333)
    p.add_argument("--client-port", type=int, default=21025)
    p.add_argument("--zoom-notches", type=int, default=4)
    p.add_argument(
        "--sample", action="store_true",
        help="sample actions instead of argmax; build sits below the argmax "
             "threshold late in training, so sampled decoding is what shows it",
    )
    p.add_argument(
        "--gl", type=str, default="vulkan",
        choices=tuple(("vulkan", "swiftshader", "default")),
        help="headless GL backend; software rasterization caps the client near 5 fps",
    )
    p.add_argument("--lapse", type=int, default=8, help="time-lapse speed factor")
    p.add_argument("--cq", type=int, default=20, help="nvenc constant quality")
    p.add_argument(
        "--deadline-minutes", type=float, default=40.0,
        help="abort if the projected run time exceeds this",
    )
    p.add_argument(
        "--allow-source-mismatch", action="store_true",
        help="record an artifact trained before the current source fingerprint",
    )
    p.add_argument(
        "--allow-unqualified-joint", action="store_true",
        help="record a complete but unqualified joint-pretrain artifact",
    )
    p.add_argument(
        "--drop-master", action="store_true",
        help="delete the live capture once the graded render exists",
    )
    return p.parse_args()


def port_is_free(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        return probe.connect_ex((host, port)) != 0


def kill_group(proc, timeout: float = 20.0) -> None:
    """Terminate a child and everything it spawned.

    `watch.py` starts the node backend as its own child and that backend owns the
    client port. Killing only the parent leaves the port bound, and the next run
    would attach its browser to a stale session - the one failure mode that
    produces a plausible-looking but wrong video.
    """
    if proc is None or proc.returncode is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        proc.terminate()


def _gray(image: bytes, cols: int = 320, rows: int = 180):
    import numpy as np

    done = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", "pipe:0", "-f", "rawvideo",
         "-pix_fmt", "gray", "-vf", f"scale={cols}:{rows}", "pipe:1"],
        input=image, capture_output=True, check=True,
    )
    gray = np.frombuffer(done.stdout, dtype=np.uint8)
    return gray.reshape(rows, cols) if gray.size == rows * cols else None


def frame_coverage(image: bytes, floor: int = 24) -> float:
    """Fraction of pixels meaningfully brighter than black, from a real frame."""
    gray = _gray(image)
    return 0.0 if gray is None else float((gray > floor).mean())


def room_box(image: bytes, floor: int = 40) -> tuple[float, float, float, float]:
    """Bounding box of the drawn room as frame fractions: x0, y0, x1, y1.

    Coverage cannot drive the zoom: the client paints its background `#1e1e1e`,
    which clears any low black threshold, so coverage barely moves when the room
    grows. The room's floor tiles are much brighter than that background, so a
    bounding box over pixels above `floor` measures the room itself.
    """
    import numpy as np

    gray = _gray(image)
    if gray is None:
        return (0.0, 0.0)
    mask = gray > floor
    rows = np.flatnonzero(mask.sum(axis=1) > 2)
    cols = np.flatnonzero(mask.sum(axis=0) > 2)
    if rows.size == 0 or cols.size == 0:
        return (0.0, 0.0)
    return (
        float(cols[0] / gray.shape[1]),
        float(rows[0] / gray.shape[0]),
        float((cols[-1] + 1) / gray.shape[1]),
        float((rows[-1] + 1) / gray.shape[0]),
    )


def box_height(box: tuple[float, float, float, float]) -> float:
    return box[3] - box[1]


def box_is_clipped(box: tuple[float, float, float, float], margin: float = 0.006) -> bool:
    """Whether the room runs off any edge.

    Height alone cannot tell "fills the frame" from "overflows it": both measure
    ~1.0, because the bounding box is over *visible* pixels. Requiring a margin
    on every side is what distinguishes them - the first recording lost the top
    and bottom of the room to exactly this blind spot.
    """
    return (
        box[0] <= margin or box[1] <= margin
        or box[2] >= 1.0 - margin or box[3] >= 1.0 - margin
    )


# Hide the client's own furniture: the recording is the room, nothing else.
STRIP_CHROME_JS = r"""
(() => {
  const canvases = [...document.querySelectorAll('canvas')];
  if (!canvases.length) return { ok: false, reason: 'no canvas' };
  const canvas = canvases.sort((a, b) => b.width * b.height - a.width * a.height)[0];
  let node = canvas;
  while (node && node.parentElement && node !== document.body) {
    for (const sibling of [...node.parentElement.children]) {
      if (sibling !== node) sibling.style.display = 'none';
    }
    const style = node.parentElement.style;
    style.margin = '0'; style.padding = '0';
    style.width = '100vw'; style.height = '100vh';
    node = node.parentElement;
  }
  document.documentElement.style.background = '#000';
  document.body.style.background = '#000';
  document.body.style.overflow = 'hidden';
  window.dispatchEvent(new Event('resize'));
  return { ok: true, canvas: `${canvas.width}x${canvas.height}` };
})()
"""

ROOM_READY_JS = (
    "[...document.querySelectorAll('canvas')]"
    ".some(c => c.width > 400 && c.height > 400)"
)


class Cdp:
    """Minimal DevTools client with deadlines and a screencast frame stream."""

    def __init__(self, port: int, timeout: float = 20.0) -> None:
        self.port = port
        self.timeout = timeout
        self._ws = None
        self._next_id = 0
        self._pending: dict[int, asyncio.Future] = {}
        self._pump: asyncio.Task | None = None
        self.latest_frame: bytes | None = None
        self.frames_seen = 0

    def _target(self) -> str:
        deadline = time.time() + 60
        last = "no answer"
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{self.port}/json/list", timeout=2,
                ) as response:
                    for tab in json.loads(response.read()):
                        if tab.get("type") == "page" and tab.get("webSocketDebuggerUrl"):
                            return tab["webSocketDebuggerUrl"]
                last = "no page target"
            except Exception as error:  # noqa: BLE001 - chromium is still booting
                last = str(error)
            time.sleep(0.5)
        raise RuntimeError(f"devtools never offered a page target: {last}")

    async def __aenter__(self) -> "Cdp":
        import websockets

        self._ws = await websockets.connect(
            self._target(), max_size=256 * 1024 * 1024, ping_interval=None,
        )
        self._pump = asyncio.create_task(self._read_loop())
        return self

    async def __aexit__(self, *_exc) -> None:
        if self._pump is not None:
            self._pump.cancel()
        if self._ws is not None:
            await self._ws.close()

    async def _read_loop(self) -> None:
        while True:
            try:
                message = json.loads(await self._ws.recv())
            except asyncio.CancelledError:
                return
            except Exception:  # noqa: BLE001 - socket closed at shutdown
                return
            if message.get("method") == "Page.screencastFrame":
                params = message["params"]
                self.latest_frame = base64.b64decode(params["data"])
                self.frames_seen += 1
                # Unacknowledged frames stop the stream, so ack before anything
                # else touches the connection.
                await self._send("Page.screencastFrameAck", sessionId=params["sessionId"])
                continue
            future = self._pending.pop(message.get("id", -1), None)
            if future is not None and not future.done():
                future.set_result(message)

    async def _send(self, method: str, **params) -> int:
        self._next_id += 1
        await self._ws.send(
            json.dumps({"id": self._next_id, "method": method, "params": params})
        )
        return self._next_id

    async def call(self, method: str, **params):
        message_id = await self._send(method, **params)
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[message_id] = future
        try:
            reply = await asyncio.wait_for(future, timeout=self.timeout)
        except asyncio.TimeoutError as error:
            self._pending.pop(message_id, None)
            raise TimeoutError(f"{method} did not answer in {self.timeout:.0f}s") from error
        if "error" in reply:
            raise RuntimeError(f"{method} failed: {reply['error']}")
        return reply.get("result", {})

    async def evaluate(self, expression: str):
        result = await self.call(
            "Runtime.evaluate", expression=expression, returnByValue=True, awaitPromise=True,
        )
        if result.get("exceptionDetails"):
            raise RuntimeError(f"page threw: {result['exceptionDetails']}")
        return result.get("result", {}).get("value")

    async def wheel(self, x: float, y: float, delta_y: float) -> None:
        await self.call(
            "Input.dispatchMouseEvent", type="mouseWheel", x=x, y=y,
            deltaX=0, deltaY=delta_y, pointerType="mouse",
        )

    async def screenshot(self, path: Path | None = None, retries: int = 3) -> bytes:
        for attempt in range(retries):
            try:
                result = await self.call(
                    "Page.captureScreenshot", format="png", fromSurface=True,
                )
                break
            except TimeoutError:
                if attempt == retries - 1:
                    raise
                await asyncio.sleep(1.0)
        png = base64.b64decode(result["data"])
        if path is not None:
            path.write_bytes(png)
        return png


# Measured on this machine with a WebGL probe page (renderer string, rAF rate,
# screencast rate at 2560x1440):
#   vulkan       ANGLE (NVIDIA, Vulkan 1.4.341)  rAF 3310/s  screencast 47.7/s
#   default      ANGLE (Google, SwiftShader)     rAF   36/s  screencast 40.0/s
#   swiftshader  ANGLE (Google, SwiftShader)     rAF  125/s  screencast  0.0/s
#   egl          no WebGL context at all
# Software rasterization draws the client's 2K tilemap at about 5 fps, so the
# hardware path is what makes a smooth recording possible at all.
GL_FLAGS = {
    "vulkan": ["--use-angle=vulkan", "--enable-features=Vulkan"],
    "swiftshader": ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
    "default": [],
}


def start_chromium(width: int, height: int, port: int, profile: Path, gl: str = "egl"):
    binary = shutil.which("chromium") or shutil.which("google-chrome-stable")
    if binary is None:
        raise RuntimeError("no chromium/chrome binary found")
    return subprocess.Popen(
        [
            binary, "--headless=new",
            "--no-first-run", "--no-default-browser-check",
            "--disable-features=Translate,MediaRouter",
            "--hide-scrollbars", "--force-device-scale-factor=1",
            "--disable-background-timer-throttling",
            "--disable-backgrounding-occluded-windows",
            "--disable-renderer-backgrounding",
            "--disable-frame-rate-limit", "--disable-gpu-vsync",
            *GL_FLAGS[gl],
            f"--user-data-dir={profile}",
            f"--remote-debugging-port={port}",
            f"--window-size={width},{height}",
            "about:blank",
        ],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def start_ffmpeg(args: argparse.Namespace, master: Path) -> subprocess.Popen:
    return subprocess.Popen(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "image2pipe", "-vcodec", "mjpeg",
            "-framerate", str(args.fps), "-i", "pipe:0",
            "-c:v", "h264_nvenc", "-preset", "p5", "-tune", "hq",
            "-rc", "vbr", "-cq", str(args.cq), "-b:v", "0",
            "-maxrate", "60M", "-bufsize", "120M", "-g", str(args.fps * 2),
            "-pix_fmt", "yuv420p",
            # Fragmented: a plain MP4 writes its index last, so the file cannot
            # be opened until the run ends. This one plays while it grows.
            "-movflags", "+frag_keyframe+empty_moov+default_base_moof",
            str(master),
        ],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )


def ass_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{int(hours)}:{int(minutes):02d}:{secs:05.2f}"


def write_hud(samples: list[HudSample], path: Path, total_seconds: float, title: str) -> None:
    header = (
        "[Script Info]\nScriptType: v4.00+\nPlayResX: 2560\nPlayResY: 1440\n"
        "WrapStyle: 2\nScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        "Style: hud,DejaVu Sans Mono,40,&H00EAEAEA,&H00EAEAEA,&HC8000000,&HB4000000,"
        "1,0,0,0,100,100,0,0,3,3,0,7,56,56,56,1\n"
        "Style: cap,DejaVu Sans,34,&H00CFCFCF,&H00CFCFCF,&HC8000000,&HB4000000,"
        "0,0,0,0,100,100,0,0,3,3,0,1,56,56,56,1\n\n"
        "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    lines = [header]
    for index, sample in enumerate(samples):
        start = sample.wall
        end = samples[index + 1].wall if index + 1 < len(samples) else total_seconds
        if end <= start:
            continue
        text = (
            f"tick {sample.tick:>6,}\\N"
            f"creeps {sample.creeps:>3}    RCL {sample.rcl}\\N"
            f"harvest {sample.harvest:>3}/t  upgrade {sample.control:>3}/t\\N"
            f"controller {sample.progress:>8,}\\N"
            f"return {sample.ep_ret:>10,.0f}"
        )
        lines.append(f"Dialogue: 0,{ass_time(start)},{ass_time(end)},hud,,0,0,0,,{text}\n")
    lines.append(
        f"Dialogue: 0,{ass_time(0.5)},{ass_time(min(14.0, total_seconds))},cap,,0,0,0,,{title}\n"
    )
    path.write_text("".join(lines), encoding="utf-8")


def run_ffmpeg(cmd: list[str], label: str) -> None:
    done = subprocess.run(cmd, capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError(f"{label} failed:\n{done.stderr[-2000:]}")


def probe(path: Path) -> dict:
    done = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "format=duration,size,bit_rate:stream=width,height,r_frame_rate,codec_name",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    return json.loads(done.stdout)


def verify_video(path: Path, min_coverage: float = 0.10) -> dict:
    """A video that decodes is not proof; sampled frames must contain the room."""
    import numpy as np

    info = probe(path)
    duration = float(info["format"]["duration"])
    stats = []
    for fraction in (0.1, 0.5, 0.9):
        done = subprocess.run(
            ["ffmpeg", "-v", "error", "-ss", f"{duration * fraction:.2f}", "-i", str(path),
             "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "gray", "-vf", "scale=320:180",
             "pipe:1"],
            capture_output=True, check=True,
        )
        gray = np.frombuffer(done.stdout, dtype=np.uint8)
        stats.append({
            "at": round(duration * fraction, 1),
            "mean": round(float(gray.mean()), 1) if gray.size else 0.0,
            "coverage": round(float((gray > 24).mean()), 4) if gray.size else 0.0,
        })
    worst = min(sample["coverage"] for sample in stats)
    if worst < min_coverage:
        raise RuntimeError(f"{path.name} frames are blank or near-blank: {stats}")
    return {"duration": duration, "samples": stats, "info": info}


async def record(args: argparse.Namespace, paths: dict) -> dict:
    """Drive the policy, the browser and the encoder together."""
    if not port_is_free(args.client_port):
        raise RuntimeError(
            f"port {args.client_port} is bound; a previous headful backend is "
            "still running and the browser would record it"
        )
    profile = Path(f"/tmp/showcase-chrome-{os.getpid()}")
    chromium = start_chromium(
        args.width, args.height, args.cdp_port, profile, gl=args.gl,
    )
    watch = None
    ffmpeg = None
    log = paths["log"].open("w", encoding="utf-8")
    samples: list[HudSample] = []
    result: dict = {}
    try:
        env = dict(os.environ)
        watch = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "samples.rl.agent.watch",
            "--checkpoint", str(args.checkpoint),
            "--room", args.room, "--curriculum", args.curriculum,
            "--ticks", str(args.ticks), "--seed", str(args.seed),
            "--headful", "--no-open",
            "--tick-ms", str(args.tick_ms), "--print-every", "10",
            *(["--sample"] if args.sample else ["--deterministic"]),
            *(["--allow-source-mismatch"] if args.allow_source_mismatch else []),
            *(["--allow-unqualified-joint"] if args.allow_unqualified_joint else []),
            cwd=str(_REPO), env=env, start_new_session=True,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )
        print(f"[showcase] policy running: {args.checkpoint.name}", flush=True)

        url = None
        while True:
            raw = await asyncio.wait_for(watch.stdout.readline(), timeout=300)
            if not raw:
                raise RuntimeError("watch exited before reporting a client URL")
            line = raw.decode("utf-8", "replace")
            log.write(line)
            match = URL_RE.search(line)
            if match:
                url = match.group(1)
                break
        print(f"[showcase] client {url}", flush=True)

        async with Cdp(args.cdp_port) as cdp:
            await cdp.call("Page.enable")
            await cdp.call(
                "Emulation.setDeviceMetricsOverride",
                width=args.width, height=args.height, deviceScaleFactor=1, mobile=False,
            )
            await cdp.call("Page.navigate", url=f"{url.rstrip('/')}/#!/room/shard0/{args.room}")
            for _ in range(60):
                await asyncio.sleep(1.0)
                if await cdp.evaluate(ROOM_READY_JS):
                    break
            else:
                raise RuntimeError("client never rendered a room canvas")
            await asyncio.sleep(3.0)
            strip = await cdp.evaluate(STRIP_CHROME_JS)
            await asyncio.sleep(1.5)

            # Zoom until the room fills the frame height. A square room in a
            # 16:9 frame tops out there; one notch further clips it, so the loop
            # steps back if it overshoots.
            box = room_box(await cdp.screenshot())
            track = [round(box_height(box), 3)]
            steps = 0
            for _ in range(max(0, args.zoom_notches)):
                await cdp.wheel(args.width / 2, args.height / 2, -240)
                await asyncio.sleep(0.8)
                candidate = room_box(await cdp.screenshot())
                grew = box_height(candidate) > box_height(box) * 1.01
                if box_is_clipped(candidate) or not grew:
                    await cdp.wheel(args.width / 2, args.height / 2, 240)
                    await asyncio.sleep(0.8)
                    box = room_box(await cdp.screenshot())
                    break
                box = candidate
                track.append(round(box_height(box), 3))
                steps += 1
                if box_height(box) >= 0.90:
                    break
            shot = await cdp.screenshot(paths["shot"])
            box = room_box(shot)
            coverage = frame_coverage(shot)
            print(
                f"[showcase] framing canvas={strip.get('canvas')} "
                f"room x=[{box[0]:.3f},{box[2]:.3f}] y=[{box[1]:.3f},{box[3]:.3f}] "
                f"height={box_height(box):.0%} clipped={box_is_clipped(box)} "
                f"coverage={coverage:.1%} zoom_steps={steps} track={track}",
                flush=True,
            )
            if box_is_clipped(box):
                raise RuntimeError(
                    f"the room runs off the frame: y=[{box[1]:.3f},{box[3]:.3f}] "
                    "x=[%.3f,%.3f]" % (box[0], box[2])
                )
            if box_height(box) < 0.55:
                raise RuntimeError(
                    f"the room fills only {box_height(box):.0%} of the frame height; "
                    "framing would waste most of the picture"
                )
            if coverage < 0.10:
                raise RuntimeError(
                    f"the room covers only {coverage:.1%} of the frame; refusing "
                    "to record a blank or badly framed video"
                )

            await cdp.call(
                "Page.startScreencast", format="jpeg", quality=args.jpeg_quality,
                maxWidth=args.width, maxHeight=args.height, everyNthFrame=1,
            )
            for _ in range(50):
                await asyncio.sleep(0.2)
                if cdp.latest_frame is not None:
                    break
            if cdp.latest_frame is None:
                raise RuntimeError("screencast produced no frames")

            ffmpeg = start_ffmpeg(args, paths["master"])
            t0 = time.monotonic()
            written = 0

            async def pace() -> None:
                """Feed exactly `fps` frames per wall second, repeating as needed."""
                nonlocal written
                period = 1.0 / args.fps
                while True:
                    frame = cdp.latest_frame
                    if frame is not None:
                        try:
                            ffmpeg.stdin.write(frame)
                            written += 1
                        except (BrokenPipeError, ValueError):
                            return
                    target = t0 + written * period
                    await asyncio.sleep(max(0.0, target - time.monotonic()))

            pacer = asyncio.create_task(pace())
            print(f"[showcase] recording -> {paths['master'].name}", flush=True)

            last_report = time.monotonic()
            while True:
                raw = await watch.stdout.readline()
                if not raw:
                    break
                line = raw.decode("utf-8", "replace")
                log.write(line)
                if FINISHED_RE.match(line):
                    log.write("[showcase] policy reported completion\n")
                    break
                match = HUD_RE.search(line)
                if not match:
                    continue
                tick = int(match.group(1))
                samples.append(HudSample(
                    wall=time.monotonic() - t0, tick=tick,
                    ep_ret=float(match.group(3)), harvest=int(match.group(5)),
                    control=int(match.group(6)), creeps=int(match.group(7)),
                    rcl=int(match.group(9)), progress=int(match.group(10)),
                ))
                now = time.monotonic()
                if now - last_report > 120:
                    last_report = now
                    elapsed = now - t0
                    rate = tick / max(1e-6, elapsed)
                    projected = args.ticks / max(1e-6, rate) / 60
                    print(
                        f"[showcase] tick {tick}/{args.ticks} {rate:.1f} ticks/s "
                        f"projected {projected:.1f} min frames={written} "
                        f"screencast={cdp.frames_seen}",
                        flush=True,
                    )
                    if projected > args.deadline_minutes:
                        raise RuntimeError(
                            f"projected {projected:.1f} min exceeds the "
                            f"{args.deadline_minutes:.0f} min deadline; lower --tick-ms"
                        )

            pacer.cancel()
            result = {
                "seconds": time.monotonic() - t0,
                "frames": written,
                "screencast_frames": cdp.frames_seen,
                "samples": samples,
            }
            try:
                await cdp.call("Page.stopScreencast")
            except Exception:  # noqa: BLE001 - shutting down anyway
                pass
    finally:
        if ffmpeg is not None and ffmpeg.stdin is not None:
            try:
                ffmpeg.stdin.close()
            except Exception:  # noqa: BLE001
                pass
            try:
                ffmpeg.wait(timeout=180)
            except subprocess.TimeoutExpired:
                ffmpeg.kill()
        kill_group(watch)
        chromium.terminate()
        try:
            chromium.wait(timeout=20)
        except subprocess.TimeoutExpired:
            chromium.kill()
        if watch is not None:
            try:
                await asyncio.wait_for(watch.wait(), timeout=20)
            except (asyncio.TimeoutError, ProcessLookupError):
                pass
        log.close()
        shutil.rmtree(profile, ignore_errors=True)
    return result


def _terminate(signum, _frame):
    raise KeyboardInterrupt(f"signal {signum}")


def main() -> int:
    args = parse_args()
    # mlq cancellation and `timeout` send SIGTERM; the default action would skip
    # cleanup and leak the browser and the node backend.
    signal.signal(signal.SIGTERM, _terminate)
    signal.signal(signal.SIGHUP, _terminate)

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    paths = {
        "master": out.with_name(out.name + "_live.mp4"),
        "final": out.with_suffix(".mp4"),
        "lapse": out.with_name(f"{out.name}_{args.lapse}x.mp4"),
        "log": out.with_name(out.name + "_run.log"),
        "hud": out.with_name(out.name + "_hud.ass"),
        "shot": out.with_name(out.name + "_framing.png"),
    }

    run = asyncio.run(record(args, paths))
    master = paths["master"]
    if not master.exists() or master.stat().st_size < 200_000:
        raise SystemExit(f"[showcase] capture produced no usable file: {master}")
    checked = verify_video(master)
    print(
        f"[showcase] captured {checked['duration'] / 60:.1f} min, "
        f"{run['frames']} frames written, {run['screencast_frames']} from the page, "
        f"{master.stat().st_size / 1e6:.0f} MB",
        flush=True,
    )
    print(f"[showcase] frame check {checked['samples']}", flush=True)

    title = (
        f"{args.checkpoint.stem}  ·  {'sampled' if args.sample else 'deterministic'}  ·  "
        f"one empty room, {args.ticks:,} ticks"
    )
    write_hud(run["samples"], paths["hud"], checked["duration"], title)
    run_ffmpeg(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(master),
         "-vf", f"eq=brightness=0.05:contrast=1.14:saturation=1.18,ass={paths['hud']}",
         "-c:v", "h264_nvenc", "-preset", "p5", "-tune", "hq", "-rc", "vbr",
         "-cq", str(args.cq), "-b:v", "0", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", "-an", str(paths["final"])],
        "HUD render",
    )
    run_ffmpeg(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(paths["final"]),
         "-vf", f"setpts=PTS/{args.lapse},fps={args.fps}",
         "-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", str(args.cq + 1),
         "-b:v", "0", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an",
         str(paths["lapse"])],
        "time lapse",
    )
    if args.drop_master:
        master.unlink(missing_ok=True)

    for key in ("final", "lapse"):
        checked = verify_video(paths[key])
        stream = checked["info"]["streams"][0]
        print(
            f"[showcase] {paths[key].name}: {stream['width']}x{stream['height']} "
            f"{stream['codec_name']} {stream['r_frame_rate']} "
            f"{checked['duration'] / 60:.2f} min "
            f"{int(checked['info']['format']['size']) / 1e6:.0f} MB "
            f"frames {checked['samples']}",
            flush=True,
        )
    if run["samples"]:
        last = run["samples"][-1]
        print(
            f"[showcase] final telemetry: tick {last.tick} creeps {last.creeps} "
            f"rcl {last.rcl} controller {last.progress} return {last.ep_ret:.0f}",
            flush=True,
        )
    print(f"[showcase] wrote {paths['final']} and {paths['lapse']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
