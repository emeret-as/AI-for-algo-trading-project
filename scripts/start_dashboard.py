#!/usr/bin/env python3
"""Start the FastAPI backend and Vite frontend for local development."""

import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"


def npm_command() -> str:
    return "npm.cmd" if os.name == "nt" else "npm"


def is_port_available(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False


def find_available_port(preferred_port: int, host: str = "127.0.0.1") -> int:
    candidate = preferred_port
    while True:
        if is_port_available(candidate, host):
            return candidate
        candidate += 1


def is_backend_ready(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/config", timeout=2) as response:
            return response.status == 200
    except Exception:
        return False


def run_command(cmd: list[str], cwd: Path, env: dict | None = None) -> subprocess.Popen:
    print(f"\n> {' '.join(cmd)}")
    return subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=sys.stdout,
        stderr=sys.stderr,
        text=True,
    )


def ensure_dependencies() -> None:
    print("Installing Python dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")], cwd=ROOT)

    if not (FRONTEND / "node_modules").exists():
        print("Installing frontend dependencies...")
        subprocess.check_call([npm_command(), "install"], cwd=FRONTEND)
    else:
        print("Frontend dependencies already installed.")


def main() -> None:
    ensure_dependencies()

    backend_port = 8000
    if is_backend_ready(backend_port):
        print(f"Backend already available at http://127.0.0.1:{backend_port}")
        backend = None
    else:
        backend_port = find_available_port(8000)
        print(f"Starting backend on http://127.0.0.1:{backend_port}")
        backend = run_command(
            [sys.executable, "-m", "uvicorn", "backend:app", "--host", "127.0.0.1", "--port", str(backend_port)],
            cwd=ROOT,
        )

    frontend_port = find_available_port(3000)
    print(f"Starting frontend on http://127.0.0.1:{frontend_port}")
    frontend_env = os.environ.copy()
    frontend_env["VITE_API_PROXY_TARGET"] = f"http://127.0.0.1:{backend_port}"
    frontend_env["PORT"] = str(frontend_port)
    frontend = run_command(
        [npm_command(), "run", "dev", "--", "--host", "127.0.0.1", "--port", str(frontend_port)],
        cwd=FRONTEND,
        env=frontend_env,
    )

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping services...")
        for proc in (backend, frontend):
            if proc is None:
                continue
            if proc.poll() is None:
                if os.name == "nt":
                    proc.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    proc.terminate()
        time.sleep(2)
        for proc in (backend, frontend):
            if proc is None:
                continue
            if proc.poll() is None:
                proc.kill()
        print("Services stopped.")


if __name__ == "__main__":
    main()
