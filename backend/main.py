"""
Backend API for BRAZEN LinkedIn AI Web Application

This API provides REST endpoints that connect to the MCP server tools.
"""

import sys
import os
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import time
from pathlib import Path

# Add MCP server to path
sys.path.insert(0, '/workspace/brazen-linkedin-ai/src')

# Import MCP server functions - use the server.py functions directly
from brazen_linkedin_ai.models import ContentInput, QUALITY_GATES
from brazen_linkedin_ai.agents import (
    ContextAnalyzer, DraftGenerator, BrandVoiceCritic, 
    AlgorithmOptimizer, MetaOrchestrator
)

app = FastAPI(title="BRAZEN LinkedIn AI API", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class FormInput(BaseModel):
    """Form input model matching frontend structure"""
    # Section 1: Audience & Strategic Intent
    audience: str
    goal: str
    emotion: str
    
    # Section 2: Content Foundation
    content_source: str
    pillar: str
    story: str
    quote: Optional[str] = None
    
    # Section 3: Project & Client Details
    client: str
    specific_project: Optional[str] = None
    additional_context: Optional[str] = None
    
    # Section 4: Voice & Format
    brand_voice: str
    tone: str
    visual: str
    length: str
    
    # Section 5: Focus & Message
    key_message: str
    engagement_question_type: str


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "BRAZEN LinkedIn AI API",
        "version": "1.0.0",
        "description": "REST API for LinkedIn content generation",
        "mcp_server_id": "326356910518361",
        "endpoints": [
            "/api/validate",
            "/api/generate",
            "/api/system-info"
        ]
    }


@app.get("/api/system-info")
async def get_system_info():
    """Get system information and capabilities"""
    return {
        "system_name": "BRAZEN LinkedIn AI",
        "version": "1.0.0",
        "mcp_server_id": "326356910518361",
        "agent_architecture": {
            "total_agents": 5,
            "agents": [
                "Context Analyzer",
                "Draft Generator", 
                "Brand Voice Critic",
                "Algorithm Optimizer",
                "Meta Orchestrator"
            ]
        },
        "brand_voice_standards": {
            "perspective": "Film/TV-first, studio leader voice",
            "authenticity_guarantee": "8+/10 score",
            "quality_gates": 5
        },
        "performance_targets": {
            "processing_time": "< 5 seconds",
            "authenticity_score": "8+/10"
        }
    }


@app.post("/api/validate")
async def validate_input(form_data: FormInput):
    """Validate form input without full processing"""
    try:
        # Convert to ContentInput for validation
        input_dict = form_data.model_dump()
        content_input = ContentInput(**input_dict)
        
        return {
            "is_valid": True,
            "errors": [],
            "warnings": []
        }
    except Exception as e:
        return {
            "is_valid": False,
            "errors": [str(e)],
            "warnings": []
        }


@app.post("/api/generate")
async def generate_linkedin_post(form_data: FormInput):
    """
    Main endpoint for generating LinkedIn posts.
    Processes form input through the 5-agent architecture.
    """
    start_time = time.time()
    
    try:
        # Convert FormInput to ContentInput
        input_dict = form_data.model_dump()
        
        try:
            content_input = ContentInput(**input_dict)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Input validation failed",
                    "details": str(e),
                    "success": False
                }
            )
        
        # Initialize agents
        context_analyzer = ContextAnalyzer()
        draft_generator = DraftGenerator()
        brand_voice_critic = BrandVoiceCritic()
        algorithm_optimizer = AlgorithmOptimizer()
        meta_orchestrator = MetaOrchestrator()
        
        # Process through 5-agent workflow
        agent_results = {}
        
        # Agent 1: Context Analyzer
        context_result = context_analyzer.process(content_input)
        agent_results["context_analyzer"] = context_result
        
        if not context_result.success:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Context analysis failed",
                    "details": context_result.errors,
                    "quality_gates_passed": [],
                    "success": False
                }
            )
        
        # Agent 2: Draft Generator
        from brazen_linkedin_ai.models import ValidationResult
        validation_result_dict = context_result.output["validation_result"]
        validation_result = ValidationResult(**validation_result_dict)
        enhanced_context = context_result.output["enhanced_context"]
        
        draft_result = draft_generator.process(content_input, validation_result, enhanced_context)
        agent_results["draft_generator"] = draft_result
        
        if not draft_result.success:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Draft generation failed",
                    "details": draft_result.errors,
                    "quality_gates_passed": [1],
                    "success": False
                }
            )
        
        # Agent 3: Brand Voice Critic
        draft = draft_result.output["draft"]
        brand_voice_result = brand_voice_critic.process(draft, content_input)
        agent_results["brand_voice_critic"] = brand_voice_result
        
        if not brand_voice_result.success:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Brand voice validation failed",
                    "details": brand_voice_result.errors,
                    "authenticity_analysis": brand_voice_result.output.get("authenticity_analysis", {}),
                    "violations": brand_voice_result.output.get("violations", []),
                    "quality_gates_passed": [1, 2],
                    "success": False
                }
            )
        
        # Agent 4: Algorithm Optimizer
        from brazen_linkedin_ai.models import AuthenticityResult
        authenticity_result_dict = brand_voice_result.output["authenticity_analysis"]
        authenticity_result = AuthenticityResult(**authenticity_result_dict)
        algorithm_result = algorithm_optimizer.process(draft, authenticity_result, content_input)
        agent_results["algorithm_optimizer"] = algorithm_result
        
        if not algorithm_result.success:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Algorithm optimization failed",
                    "details": algorithm_result.errors,
                    "authenticity_score": authenticity_result.authenticity_score,
                    "quality_gates_passed": [1, 2, 3],
                    "success": False
                }
            )
        
        # Agent 5: Meta Orchestrator (Final Approval)
        final_result = meta_orchestrator.process(agent_results, content_input)
        
        if not final_result.success:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Final approval failed",
                    "details": final_result.errors,
                    "quality_gates_passed": [1, 2, 3, 4],
                    "success": False
                }
            )
        
        # Return successful result
        final_output = final_result.output
        processing_time = time.time() - start_time
        
        return {
            "success": True,
            "final_post": final_output["final_post"],
            "authenticity_score": final_output["authenticity_score"],
            "algorithm_optimizations": final_output.get("algorithm_optimizations", []),
            "brand_voice_compliance": final_output.get("brand_voice_compliance", True),
            "quality_gates_passed": final_output["quality_gates_passed"],
            "generation_metadata": {
                **final_output.get("generation_metadata", {}),
                "processing_time": f"{processing_time:.2f}s"
            },
            "suggested_hashtags": final_output.get("suggested_hashtags", []),
            "engagement_optimizations": final_output.get("engagement_optimizations", [])
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error",
                "details": str(e),
                "success": False
            }
        )


# Serve static files in production
static_dir = Path(__file__).parent.parent / "dist"
if static_dir.exists():
    app.mount("/assets", StaticFiles(directory=str(static_dir / "assets")), name="assets")
    
    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        """Serve frontend for all non-API routes"""
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="API endpoint not found")
        
        file_path = static_dir / full_path
        if file_path.is_file():
            return FileResponse(file_path)
        
        # Default to index.html for SPA routing
        return FileResponse(static_dir / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
