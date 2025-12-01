"""
NPC Report Generation System - FastAPI Backend
===============================================
REST API server for the NPC report generation system.
"""

import os
import json
from pathlib import Path
from typing import Optional, List
from datetime import datetime

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from pydantic import BaseModel

# Import our modules
from .config import get_config, update_gemini_api_key
from .pipeline import NPCReportPipeline


# ============================================================
# Pydantic Models for API
# ============================================================

class ConfigUpdate(BaseModel):
    gemini_api_key: Optional[str] = None

class ChatRequest(BaseModel):
    message: str
    stream: bool = False

class ProcessRequest(BaseModel):
    filename: str
    dataset: str = "test"  # "test" or "val"
    generate_report: bool = True
    stream: bool = False

class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool
    gemini_configured: bool
    timestamp: str


# ============================================================
# FastAPI Application
# ============================================================

def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    
    app = FastAPI(
        title="NPC Report Generation API",
        description="API for NPC (Nasopharyngeal Carcinoma) tumor segmentation and report generation",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # CORS middleware
    config = get_config()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.server.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize pipeline (lazy)
    pipeline: Optional[NPCReportPipeline] = None
    
    def get_pipeline() -> NPCReportPipeline:
        nonlocal pipeline
        if pipeline is None:
            pipeline = NPCReportPipeline()
            pipeline.initialize()
        return pipeline
    
    # ============================================================
    # API Endpoints
    # ============================================================
    
    @app.get("/", tags=["General"])
    async def root():
        """Root endpoint"""
        return {
            "message": "NPC Report Generation API",
            "docs": "/docs",
            "health": "/health"
        }
    
    @app.get("/health", response_model=HealthResponse, tags=["General"])
    async def health_check():
        """Health check endpoint"""
        config = get_config()
        p = get_pipeline()
        
        return HealthResponse(
            status="healthy",
            version=config.version,
            model_loaded=p.is_initialized,
            gemini_configured=bool(config.gemini.api_key),
            timestamp=datetime.now().isoformat()
        )
    
    @app.post("/config", tags=["Configuration"])
    async def update_config(config_update: ConfigUpdate):
        """Update system configuration"""
        if config_update.gemini_api_key:
            update_gemini_api_key(config_update.gemini_api_key)
            # Reinitialize pipeline with new key
            nonlocal pipeline
            pipeline = None
            get_pipeline()
            return {"message": "Configuration updated", "gemini_configured": True}
        
        return {"message": "No changes made"}
    
    @app.get("/cases", tags=["Cases"])
    async def list_cases():
        """List available cases"""
        p = get_pipeline()
        cases = p.list_available_cases()
        return {
            "test_cases": cases.get('test', []),
            "val_cases": cases.get('val', []),
            "total": len(cases.get('test', [])) + len(cases.get('val', []))
        }
    
    @app.post("/process", tags=["Processing"])
    async def process_case(request: ProcessRequest):
        """
        Process a case through the pipeline.
        
        Returns segmentation results, features, and AI report.
        """
        p = get_pipeline()
        
        # Get case path
        case_path = p.get_case_path(request.filename, request.dataset)
        if not case_path:
            raise HTTPException(status_code=404, detail=f"Case not found: {request.filename}")
        
        if request.stream:
            # Return streaming response
            async def generate():
                for update in p.process_case_stream(case_path):
                    yield f"data: {json.dumps(update, ensure_ascii=False)}\n\n"
            
            return StreamingResponse(
                generate(),
                media_type="text/event-stream"
            )
        else:
            # Regular processing
            results = p.process_case(case_path, generate_report=request.generate_report)
            
            if results['status'] == 'error':
                raise HTTPException(status_code=500, detail=results['error'])
            
            return results
    
    @app.post("/process/stream", tags=["Processing"])
    async def process_case_stream(request: ProcessRequest):
        """
        Process a case with streaming updates.
        
        Returns Server-Sent Events with progress updates.
        """
        p = get_pipeline()
        
        case_path = p.get_case_path(request.filename, request.dataset)
        if not case_path:
            raise HTTPException(status_code=404, detail=f"Case not found: {request.filename}")
        
        async def generate():
            for update in p.process_case_stream(case_path):
                yield f"data: {json.dumps(update, ensure_ascii=False)}\n\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream"
        )
    
    @app.post("/upload", tags=["Processing"])
    async def upload_and_process(
        file: UploadFile = File(...),
        generate_report: bool = Form(True)
    ):
        """
        Upload an HDF5 file and process it.
        """
        if not file.filename.endswith('.h5'):
            raise HTTPException(status_code=400, detail="Only HDF5 files (.h5) are supported")
        
        # Save uploaded file temporarily
        config = get_config()
        upload_dir = config.data.reports_dir / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = upload_dir / file.filename
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Process
        p = get_pipeline()
        results = p.process_case(file_path, generate_report=generate_report)
        
        # Clean up uploaded file
        # file_path.unlink()  # Uncomment to delete after processing
        
        if results['status'] == 'error':
            raise HTTPException(status_code=500, detail=results['error'])
        
        return results
    
    @app.post("/chat", tags=["Chat"])
    async def chat(request: ChatRequest):
        """
        Chat about the current case.
        
        Ask follow-up questions about the analysis results.
        """
        p = get_pipeline()
        
        if not p.current_case:
            raise HTTPException(
                status_code=400, 
                detail="No case loaded. Process a case first."
            )
        
        if request.stream:
            async def generate():
                for chunk in p.chat(request.message, stream=True):
                    yield f"data: {json.dumps({'chunk': chunk}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'done': True}, ensure_ascii=False)}\n\n"
            
            return StreamingResponse(
                generate(),
                media_type="text/event-stream"
            )
        else:
            response = p.chat(request.message)
            return {"response": response}
    
    @app.get("/chat/history", tags=["Chat"])
    async def get_chat_history():
        """Get chat history for current session"""
        p = get_pipeline()
        return {"history": p.get_chat_history()}
    
    @app.post("/chat/reset", tags=["Chat"])
    async def reset_chat():
        """Reset chat session"""
        p = get_pipeline()
        p.reset_chat()
        return {"message": "Chat session reset"}
    
    @app.get("/reports/{patient_id}", tags=["Reports"])
    async def get_report(patient_id: str):
        """Get saved report for a patient"""
        config = get_config()
        report_dir = config.data.reports_dir / patient_id
        
        if not report_dir.exists():
            raise HTTPException(status_code=404, detail=f"Report not found: {patient_id}")
        
        results_file = report_dir / "results.json"
        if results_file.exists():
            with open(results_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        raise HTTPException(status_code=404, detail="Results file not found")
    
    @app.get("/reports/{patient_id}/image/{image_type}", tags=["Reports"])
    async def get_report_image(patient_id: str, image_type: str):
        """
        Get visualization image for a patient.
        
        image_type: multi_slice, three_plane, or summary
        """
        config = get_config()
        
        valid_types = ['multi_slice', 'three_plane', 'summary']
        if image_type not in valid_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid image type. Must be one of: {valid_types}"
            )
        
        image_path = config.data.reports_dir / patient_id / f"{image_type}.png"
        
        if not image_path.exists():
            raise HTTPException(status_code=404, detail=f"Image not found: {image_type}")
        
        return FileResponse(image_path, media_type="image/png")
    
    @app.get("/reports", tags=["Reports"])
    async def list_reports():
        """List all generated reports"""
        config = get_config()
        reports_dir = config.data.reports_dir
        
        if not reports_dir.exists():
            return {"reports": []}
        
        reports = []
        for patient_dir in reports_dir.iterdir():
            if patient_dir.is_dir() and (patient_dir / "results.json").exists():
                with open(patient_dir / "results.json", 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    reports.append({
                        "patient_id": data.get('patient_id'),
                        "timestamp": data.get('timestamp'),
                        "status": data.get('status')
                    })
        
        return {"reports": sorted(reports, key=lambda x: x.get('timestamp', ''), reverse=True)}
    
    return app


# Create app instance
app = create_app()


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the FastAPI server"""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
