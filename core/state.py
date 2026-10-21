from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class PipelineState(BaseModel):
    """Shared state passed through the LangGraph pipeline."""

    # Run metadata
    run_id: str = Field(default="", description="Unique run identifier")
    status: str = Field(default="initialized", description="Current pipeline status")

    # Phase 1: Ingestion
    uploaded_files: list[str] = Field(default_factory=list, description="Paths to uploaded CSV files")
    business_intent: str = Field(default="", description="User's natural language analytical goal")

    # Phase 2: Profiling
    profile_path: str = Field(default="", description="Path to generated metadata profile JSON")

    # Phase 3: STTM
    sttm_bronze_path: str = Field(default="", description="Path to Bronze STTM rules")
    sttm_silver_path: str = Field(default="", description="Path to Silver STTM rules")
    sttm_gold_path: str = Field(default="", description="Path to Gold STTM rules")
    hitl_approved: bool = Field(default=False, description="Whether HITL gate has been approved")

    # Phase 4: Execution
    bronze_output_paths: list[str] = Field(default_factory=list, description="Bronze layer output file paths")
    silver_output_paths: list[str] = Field(default_factory=list, description="Silver layer output file paths")
    gold_output_paths: list[str] = Field(default_factory=list, description="Gold layer output file paths")

    # Phase 5: Reporting
    report_path: str = Field(default="", description="Path to final executive report")

    # Error handling
    error: Optional[str] = Field(default=None, description="Error message if pipeline fails")
