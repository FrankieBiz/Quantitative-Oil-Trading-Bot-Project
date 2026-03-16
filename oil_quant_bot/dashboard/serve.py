"""
Production server for the Oil Quant Bot Dashboard.

Supports remote access from Mac, Windows, or mobile devices.

Usage:
    # Local only (default):
    python -m dashboard.serve

    # Expose on local network (access from phone/Mac on same WiFi):
    python -m dashboard.serve --host 0.0.0.0

    # With Cloudflare Tunnel (secure remote access from anywhere):
    python -m dashboard.serve --tunnel

    # Custom port:
    python -m dashboard.serve --port 8050
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import threading

from loguru import logger


def get_local_ip() -> str:
    """Get the machine's local network IP address."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def start_cloudflare_tunnel(port: int) -> None:
    """Start a Cloudflare Quick Tunnel in a background thread.

    This gives you a public HTTPS URL accessible from anywhere — no
    account required.  Install ``cloudflared`` first:

        # macOS
        brew install cloudflare/cloudflare/cloudflared

        # Windows (winget)
        winget install Cloudflare.cloudflared

        # Linux
        curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o cloudflared
        chmod +x cloudflared && sudo mv cloudflared /usr/local/bin/
    """

    def _run():
        try:
            proc = subprocess.Popen(
                ["cloudflared", "tunnel", "--url", f"http://localhost:{port}"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            for line in proc.stdout:
                line = line.strip()
                if ".trycloudflare.com" in line:
                    logger.info("Cloudflare Tunnel URL: {}", line)
                    print(f"\n  REMOTE URL: {line}\n")
        except FileNotFoundError:
            logger.error(
                "cloudflared not found. Install it first:\n"
                "  macOS:   brew install cloudflare/cloudflare/cloudflared\n"
                "  Windows: winget install Cloudflare.cloudflared\n"
                "  Linux:   see https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/install-and-setup/"
            )

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()


def main():
    parser = argparse.ArgumentParser(description="Oil Quant Bot Dashboard Server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8050, help="Port (default: 8050)")
    parser.add_argument("--tunnel", action="store_true", help="Start Cloudflare tunnel for remote access")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    from dashboard.app import DashboardApp

    local_ip = get_local_ip()

    print("\n" + "=" * 60)
    print("  Oil Quant Bot Dashboard")
    print("=" * 60)
    print(f"\n  Local:   http://localhost:{args.port}")
    if args.host == "0.0.0.0":
        print(f"  Network: http://{local_ip}:{args.port}")
        print(f"\n  Access from your phone or Mac at the Network URL above.")
        print(f"  (Devices must be on the same WiFi network)")
    if args.tunnel:
        print(f"\n  Starting Cloudflare tunnel for remote access...")
        start_cloudflare_tunnel(args.port)
    print("\n" + "=" * 60 + "\n")

    dashboard = DashboardApp()
    app = dashboard.create_app()
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
