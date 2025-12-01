#!/usr/bin/env python
"""
NPC Report Generation System - Main Entry Point
================================================

Run the system with different modes:
- gradio: Launch Gradio web interface
- api: Launch FastAPI backend
- both: Launch both services
"""

import argparse
import threading
import time
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_gradio(port: int = 7860, share: bool = False):
    """Run Gradio interface"""
    from npc_system.gradio_ui import launch_gradio
    launch_gradio(share=share, server_port=port)


def run_api(port: int = 8000):
    """Run FastAPI server"""
    import uvicorn
    from npc_system.api import app
    uvicorn.run(app, host="0.0.0.0", port=port)


def run_both(gradio_port: int = 7860, api_port: int = 8000, share: bool = False):
    """Run both Gradio and FastAPI"""
    
    # Start API in a thread
    api_thread = threading.Thread(
        target=run_api,
        kwargs={"port": api_port},
        daemon=True
    )
    api_thread.start()
    
    print(f"✅ FastAPI server started at http://localhost:{api_port}")
    print(f"📚 API docs available at http://localhost:{api_port}/docs")
    
    time.sleep(2)  # Wait for API to start
    
    # Run Gradio in main thread
    print(f"\n🚀 Starting Gradio interface...")
    run_gradio(port=gradio_port, share=share)


def main():
    parser = argparse.ArgumentParser(
        description="NPC Report Generation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run Gradio web interface
  python run.py gradio
  
  # Run FastAPI backend
  python run.py api
  
  # Run both services
  python run.py both
  
  # Run with custom ports
  python run.py both --gradio-port 7861 --api-port 8001
  
  # Share Gradio publicly
  python run.py gradio --share
        """
    )
    
    parser.add_argument(
        "mode",
        choices=["gradio", "api", "both"],
        help="Run mode: gradio (web UI), api (backend), or both"
    )
    
    parser.add_argument(
        "--gradio-port",
        type=int,
        default=7860,
        help="Port for Gradio interface (default: 7860)"
    )
    
    parser.add_argument(
        "--api-port",
        type=int,
        default=8000,
        help="Port for FastAPI server (default: 8000)"
    )
    
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio link"
    )
    
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Gemini API key (can also use GEMINI_API_KEY env var)"
    )
    
    args = parser.parse_args()
    
    # Set API key if provided
    if args.api_key:
        os.environ["GEMINI_API_KEY"] = args.api_key
    
    print("""
╔══════════════════════════════════════════════════════════════╗
║       NPC Tumor Report Generation System v1.0.0              ║
║       Hệ thống tạo báo cáo khối u vòm họng                   ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    if args.mode == "gradio":
        print(f"🌐 Starting Gradio interface on port {args.gradio_port}...")
        run_gradio(port=args.gradio_port, share=args.share)
        
    elif args.mode == "api":
        print(f"🔧 Starting FastAPI server on port {args.api_port}...")
        run_api(port=args.api_port)
        
    elif args.mode == "both":
        print(f"🚀 Starting both services...")
        print(f"   - Gradio: port {args.gradio_port}")
        print(f"   - FastAPI: port {args.api_port}")
        run_both(
            gradio_port=args.gradio_port,
            api_port=args.api_port,
            share=args.share
        )


if __name__ == "__main__":
    main()
