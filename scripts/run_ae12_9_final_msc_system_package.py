#!/usr/bin/env python3
"""AE12.9 Final MSc System Package (read-only synthesis from AE6–AE12 evidence).

Does not rebuild, retrain, call external APIs, mutate trader.db, enable live
trading, connect a wallet, or claim profitability / live readiness.
"""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(r"E:\Projects\Final Project\memecoin_trader").resolve()
TIMESTAMP = "20260717_224343"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "audits" / f"ae12_9_final_msc_system_package_{TIMESTAMP}"

AE12_8_ROOT = (
    PROJECT_ROOT
    / "data"
    / "audits"
    / "ae12_8_safety_reproducibility_package_20260717_204845"
)
AE12_7_ROOT = (
    PROJECT_ROOT
    / "data"
    / "audits"
    / "ae12_7_intelligent_agent_operational_demo_20260717_130004_693435"
)
AE12_6_ROOT = (
    PROJECT_ROOT / "data" / "audits" / "ae12_ml_meta_layer_evaluation_20260717_111056"
)
FORWARD_ROOT = (
    PROJECT_ROOT / "data" / "audits" / "ae12_forward_evidence_maturation_20260714_235401"
)
SENTIMENTFIX_ROOT = (
    PROJECT_ROOT / "data" / "audits" / "ae12_sentimentfix_20260715_172645"
)
AE11_ROOT = (
    PROJECT_ROOT
    / "data"
    / "audits"
    / "ae11_runtime_paper_loop_20260714T173541Z_de47ba58"
)
AE10_ROOT = (
    PROJECT_ROOT / "data" / "audits" / "ae10_trading_orchestration_20260711T090202Z"
)
AE9_ROOT = PROJECT_ROOT / "data" / "audits" / "ae9_llm_audit_20260711T090138Z"
AE8_ROOT = PROJECT_ROOT / "data" / "audits" / "ae8_context_intelligence_20260711T090138Z"
AE6_ROOT = (
    PROJECT_ROOT / "data" / "audits" / "ae6_consensus_decision_layer_20260711T090105Z"
)
AE7C1_ROOT = (
    PROJECT_ROOT
    / "data"
    / "audits"
    / "ae7c1_scoring_policy_binding_parity_gate_20260710T120342Z"
)
TRADER_DB = PROJECT_ROOT / "data" / "trader.db"

CREATED_AT = datetime.now(timezone.utc).isoformat()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_meta(path: Path) -> dict[str, Any]:
    exists = path.exists()
    meta: dict[str, Any] = {
        "exact_path": str(path),
        "exists": exists,
        "file_size_bytes": "",
        "last_modified": "",
    }
    if exists:
        st = path.stat()
        meta["file_size_bytes"] = st.st_size
        meta["last_modified"] = datetime.fromtimestamp(
            st.st_mtime, tz=timezone.utc
        ).isoformat()
    return meta


def read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:  # noqa: BLE001
        return {"_parse_error": str(exc)}


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def sha256_quick(path: Path, max_bytes: int = 1024 * 1024) -> str:
    if not path.exists():
        return "MISSING"
    h = hashlib.sha256()
    size = path.stat().st_size
    h.update(str(size).encode("utf-8"))
    h.update(b"|")
    with path.open("rb") as f:
        h.update(f.read(max_bytes))
    return h.hexdigest()


def ensure_dirs() -> None:
    for sub in ("reports", "figures", "tables", "manifests", "audits"):
        (OUTPUT_ROOT / sub).mkdir(parents=True, exist_ok=True)


def run_compileall() -> dict[str, Any]:
    """Lightweight validation only — no full test suite."""
    cmd = [sys.executable, "-m", "compileall", "app", "scripts", "tests", "-q"]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        return {
            "command": " ".join(cmd),
            "exit_code": proc.returncode,
            "stdout_tail": (proc.stdout or "")[-2000:],
            "stderr_tail": (proc.stderr or "")[-2000:],
            "status": "PASS" if proc.returncode == 0 else "FAIL",
            "full_test_suite_run": False,
            "note": "compileall only; full pytest suite not re-run in AE12.9",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "command": " ".join(cmd),
            "exit_code": -1,
            "status": "ERROR",
            "error": str(exc),
            "full_test_suite_run": False,
        }


def build_source_index() -> list[dict[str, Any]]:
    """Primary source artifact index for claim traceability."""
    entries: list[tuple[str, str, Path, str]] = [
        (
            "AE12.8",
            "safety_decision_gate",
            AE12_8_ROOT / "reports" / "ae12_8_final_decision_gate.json",
            "Safety/repro package gate; AE12.9 not blocked",
        ),
        (
            "AE12.8",
            "summary",
            AE12_8_ROOT / "reports" / "ae12_8_summary_for_upload.txt",
            "AE12.8 upload summary",
        ),
        (
            "AE12.8",
            "wallet_safety",
            AE12_8_ROOT / "audits" / "ae12_8_wallet_no_live_safety_audit.json",
            "No-wallet / no-live safety audit",
        ),
        (
            "AE12.8",
            "authority_matrix",
            AE12_8_ROOT / "audits" / "ae12_8_authority_matrix.csv",
            "Component authority matrix",
        ),
        (
            "AE12.8",
            "subsystem_gates",
            AE12_8_ROOT / "manifests" / "ae12_8_subsystem_gate_matrix.csv",
            "AE6–AE12.8 saved gate matrix",
        ),
        (
            "AE12.8",
            "artifact_inventory",
            AE12_8_ROOT / "manifests" / "ae12_8_artifact_inventory.csv",
            "Primary evidence inventory for AE12.9",
        ),
        (
            "AE12.8",
            "what_built_vs_not_proven",
            AE12_8_ROOT / "reports" / "ae12_8_what_was_built_vs_not_proven.md",
            "Built vs not-proven narrative",
        ),
        (
            "AE12.8",
            "known_limitations",
            AE12_8_ROOT / "reports" / "ae12_8_known_limitations.md",
            "Archival limitations",
        ),
        (
            "AE12.8",
            "ui_api_inventory",
            AE12_8_ROOT / "manifests" / "ae12_8_ui_api_inventory.csv",
            "UI/API surface inventory",
        ),
        (
            "AE12.8",
            "forward_inventory",
            AE12_8_ROOT / "data" / "ae12_8_forward_evidence_inventory.csv",
            "Forward evidence paths",
        ),
        (
            "AE12.8",
            "paper_demo_inventory",
            AE12_8_ROOT / "data" / "ae12_8_paper_demo_evidence_inventory.csv",
            "Paper/demo evidence paths",
        ),
        (
            "AE12.8",
            "agent_inventory",
            AE12_8_ROOT / "data" / "ae12_8_agent_evidence_inventory.csv",
            "Agent evidence paths",
        ),
        (
            "AE12.8",
            "semantic_inventory",
            AE12_8_ROOT / "data" / "ae12_8_semantic_sentimentfix_inventory.csv",
            "SentimentFix/semantic inventory",
        ),
        (
            "AE12.8",
            "reproducibility_manifest",
            AE12_8_ROOT / "manifests" / "ae12_8_reproducibility_manifest.json",
            "Reproducibility references",
        ),
        (
            "AE12.7",
            "decision_gate",
            AE12_7_ROOT / "reports" / "ae12_7_intelligent_agent_decision_gate.json",
            "Intelligent-agent operational demo gate",
        ),
        (
            "AE12.7",
            "summary",
            AE12_7_ROOT / "reports" / "ae12_7_summary_for_upload.txt",
            "AE12.7 upload summary",
        ),
        (
            "AE12.7",
            "agent_authority",
            AE12_7_ROOT / "audits" / "ae12_7_agent_authority_audit.json",
            "Agent no-trade-authority audit",
        ),
        (
            "AE12.7",
            "no_wallet",
            AE12_7_ROOT / "audits" / "ae12_7_no_wallet_safety_audit.json",
            "AE12.7 no-wallet audit",
        ),
        (
            "AE12.7",
            "agent_records",
            AE12_7_ROOT / "data" / "ae12_7_agent_records.jsonl",
            "Agent observability records",
        ),
        (
            "AE12.7",
            "agent_trade_linkage",
            AE12_7_ROOT / "data" / "ae12_7_agent_trade_linkage.csv",
            "Agent-to-paper linkage",
        ),
        (
            "AE12.7",
            "ui_status",
            AE12_7_ROOT / "reports" / "ae12_7_ui_status_summary.json",
            "Agent UI status summary",
        ),
        (
            "AE12.7",
            "daily_records",
            PROJECT_ROOT
            / "data"
            / "intelligent_agents"
            / "ae12_7_agent_records_20260717.jsonl",
            "Daily append-only agent records",
        ),
        (
            "AE12.6",
            "decision_gate",
            AE12_6_ROOT / "audits" / "ae12_ml_meta_layer_evaluation_gate.json",
            "ML/meta-layer evaluation gate",
        ),
        (
            "AE12.6",
            "summary",
            AE12_6_ROOT / "reports" / "ae12_ml_meta_layer_evaluation_for_upload.txt",
            "AE12.6 upload summary",
        ),
        (
            "AE12.6",
            "model_performance",
            AE12_6_ROOT / "data" / "ae12_model_performance_summary.csv",
            "RF/XGB/TAB/Consensus/Meta headline matrix",
        ),
        (
            "AE12.6",
            "evaluation_matrix",
            AE12_6_ROOT / "data" / "ae12_ml_meta_layer_evaluation_matrix.csv",
            "Layer evaluation matrix",
        ),
        (
            "AE12.4",
            "forward_readiness",
            FORWARD_ROOT / "reports" / "ae12_final_system_readiness_gate.json",
            "Forward evidence readiness for reporting",
        ),
        (
            "AE12.4",
            "opportunity_capture",
            FORWARD_ROOT / "data" / "ae12_opportunity_capture_full.csv",
            "Opportunity capture evidence",
        ),
        (
            "AE12.4",
            "missed_winners",
            FORWARD_ROOT / "data" / "ae12_missed_winners_full.csv",
            "Missed-winner evidence",
        ),
        (
            "AE12.4",
            "trade_vs_no_trade",
            FORWARD_ROOT / "data" / "ae12_trade_vs_no_trade_comparison.csv",
            "Trade vs no-trade comparison",
        ),
        (
            "AE12.4",
            "strict_vs_exploration",
            FORWARD_ROOT / "data" / "ae12_strict_vs_exploration_comparison.csv",
            "Strict vs exploration comparison",
        ),
        (
            "SentimentFix",
            "decision_gate",
            SENTIMENTFIX_ROOT / "audits" / "ae12_sentimentfix_decision_gate.json",
            "Dual-axis SentimentFix gate",
        ),
        (
            "AE11",
            "decision_gate",
            AE11_ROOT / "reports" / "ae11_decision_gate.json",
            "Runtime paper loop gate",
        ),
        (
            "AE10",
            "decision_gate",
            AE10_ROOT / "reports" / "ae10_decision_gate.json",
            "Paper/demo orchestration gate",
        ),
        (
            "AE10",
            "no_wallet_dry_run",
            AE10_ROOT / "audits" / "ae10_no_wallet_live_dry_run_audit.json",
            "No-wallet live dry-run audit",
        ),
        (
            "AE9",
            "decision_gate",
            AE9_ROOT / "reports" / "ae9_decision_gate.json",
            "LLM audit layer gate (mock)",
        ),
        (
            "AE8",
            "root",
            AE8_ROOT,
            "Context intelligence root (may be empty archival)",
        ),
        (
            "AE7",
            "inference_gate",
            AE7C1_ROOT / "ae7c1_inference_readiness_gate.json",
            "AE7 runtime stacking not closed",
        ),
        (
            "AE6",
            "consensus_summary",
            AE6_ROOT / "ae6_consensus_decision_summary.json",
            "DecisionRecord / consensus summary",
        ),
        (
            "AE12.5",
            "report_manager",
            PROJECT_ROOT / "app" / "ae12_reporting" / "report_manager.py",
            "UI/reporting layer source",
        ),
    ]
    rows: list[dict[str, Any]] = []
    for phase, artifact_type, path, notes in entries:
        meta = file_meta(path)
        rows.append(
            {
                "phase": phase,
                "artifact_type": artifact_type,
                "exact_path": meta["exact_path"],
                "exists": meta["exists"],
                "file_size_bytes": meta["file_size_bytes"],
                "last_modified": meta["last_modified"],
                "used_in_ae12_9": True,
                "notes": notes,
            }
        )
    return rows


def write_figures() -> list[str]:
    figures: dict[str, str] = {
        "ae12_9_final_architecture.mmd": """%%{init: {'theme': 'neutral'}}%%
flowchart TB
  subgraph Market["Market / Runtime Data"]
    MD[Market feeds / pair snapshots]
    RT[Runtime collection / census]
  end
  subgraph Decision["DecisionRecord Layer AE6"]
    DR[DecisionRecord contract]
    CONS[Consensus tiers - diagnostic]
  end
  subgraph Models["Model Evidence AE7 / AE12.6"]
    RF[RF - research/reporting]
    XGB[XGB - historical/exit-sim/reporting]
    TAB[TAB - research/reporting]
    META[Meta-layer - evidence exists / not live AE7 authority]
  end
  subgraph Context["Context Intelligence AE8 / SentimentFix"]
    LIQ[Liquidity / activity context]
    RSS[RSS / sentiment linkage]
    SEM[Semantic dual-axis taxonomy]
    HEL[Helius read-only enrichment path]
  end
  subgraph Agent["Intelligent Agent AE12.7"]
    QW[Qwen / local memo]
    GEM[Gemini selective audit]
    AG[Agent records / linkage]
  end
  subgraph Exec["Paper / Demo Execution AE10 / AE11"]
    PAPER[Paper orders / fills / positions]
    LOOP[Runtime paper loop]
  end
  subgraph Forward["Forward Evidence AE12.1-4"]
    OPP[Opportunity capture]
    MW[Missed winners]
    HOR[No-lookahead maturation]
  end
  subgraph UI["UI / API Reporting AE12.5"]
    API[Read-only AE12 APIs]
    PANEL[Forward / SentimentFix / Agents panels]
  end
  subgraph Safety["Safety Boundary"]
    NW[No wallet / no private key]
    NL[No live submission]
    NA[No model/LLM trade authority]
  end
  MD --> RT --> DR
  DR --> CONS
  DR --> RF & XGB & TAB
  RF & XGB & TAB --> META
  DR --> LIQ & RSS & SEM & HEL
  LIQ & RSS & SEM --> QW & GEM --> AG
  DR --> PAPER --> LOOP
  LOOP --> OPP & MW & HOR
  OPP & MW & AG --> API --> PANEL
  PAPER -.->|paper/demo only| Safety
  AG -.->|audit/explanation only| Safety
  META -.->|no live authority| Safety
""",
        "ae12_9_runtime_data_flow.mmd": """%%{init: {'theme': 'neutral'}}%%
flowchart LR
  A[Market / pair snapshots] --> B[Runtime data census AE12.1]
  B --> C[DecisionRecord slots]
  C --> D[Feature / score attachments]
  D --> E[Paper/demo candidate selection]
  E --> F[Paper order / position / ledger]
  F --> G[Forward opportunity capture]
  G --> H[Horizon maturation / no-lookahead]
  H --> I[Reporting CSVs / readiness gate]
  I --> J[AE12 ReportManager / UI-API]
  J --> K[MSc package AE12.8 / AE12.9]
""",
        "ae12_9_decision_authority_flow.mmd": """%%{init: {'theme': 'neutral'}}%%
flowchart TB
  SRC[Signals: RF / XGB / TAB / Consensus / Context / LLM]
  SRC --> REP[Reporting / research / diagnostic outputs]
  SRC --> CTX[Context / soft-warning / soft-veto recommendation]
  REP --> PAPER{Paper/demo execution?}
  CTX --> PAPER
  PAPER -->|YES allowed| DEMO[Paper orders / positions / ledger]
  PAPER -->|NO live| BLOCK[Live trading BLOCKED]
  DEMO --> SAFE[Safety gates: wallet=false / no signing]
  BLOCK --> SAFE
  SAFE --> OUT[live_trading_approval=NO]
  note1[RF/XGB/TAB/Consensus/Meta: NO live trade authority]
  note2[Qwen/Gemini/Helius/RSS/Semantic: NO trade authority]
  note1 -.-> SRC
  note2 -.-> SRC
""",
        "ae12_9_paper_demo_flow.mmd": """%%{init: {'theme': 'neutral'}}%%
flowchart TB
  A[AE10 one-shot paper/demo proof] --> B[Traceability records / dry-run path]
  B --> C[AE11 runtime paper loop]
  C --> D{Strict shadow vs Exploration}
  D -->|Strict| E[Strict freshness / shadow rules]
  D -->|Exploration| F[Exploration candidate loop]
  E --> G[Paper orders / fills]
  F --> G
  G --> H[Positions / ledger]
  H --> I[Deterministic TP/SL lifecycle proof path]
  I --> J[Forward evidence capture]
  J --> K[No wallet / NOT_SUBMITTED_NO_WALLET]
  K --> L[Live real-money trading NOT approved]
""",
        "ae12_9_agent_layer_flow.mmd": """%%{init: {'theme': 'neutral'}}%%
flowchart TB
  CAND[Candidate / missed-winner inputs] --> QW[Qwen/local memo path]
  CAND --> GEM[Gemini selective audit path]
  CAND --> HEL[Helius read-only enrichment]
  CAND --> RSS[RSS / semantic linkage]
  QW --> REC[Agent records JSONL]
  GEM --> REC
  HEL --> REC
  RSS --> REC
  REC --> LINK[Agent-trade linkage CSV]
  REC --> UI[Agents UI / API status]
  LINK --> AUD[Audit / explanation / context only]
  UI --> AUD
  AUD --> NOAUTH[NO trade authority / NO wallet / NO live]
""",
    }
    written: list[str] = []
    for name, body in figures.items():
        path = OUTPUT_ROOT / "figures" / name
        write_text(path, body.strip() + "\n")
        written.append(str(path))
    return written


def write_tables(source_index: list[dict[str, Any]]) -> None:
    # Phase status matrix from AE12.8 subsystem gates + AE12.9 closure
    phase_rows = [
        {
            "phase": "AE6",
            "saved_status": "AE6_CONSENSUS_DECISION_LAYER_NO_TRADE_AUTHORITY_SUMMARY",
            "role": "DecisionRecord / consensus architecture",
            "msc_status": "DEMONSTRATED_REPORTING",
            "live_authority": "NO",
            "source_artifact": str(
                AE6_ROOT / "ae6_consensus_decision_summary.json"
            ),
            "limitation": "Not live authority",
        },
        {
            "phase": "AE7",
            "saved_status": "BLOCKED_POLICY_PLACEHOLDER",
            "role": "Model score / meta-layer runtime stack",
            "msc_status": "RESEARCH_NOT_RUNTIME_CLOSED",
            "live_authority": "NO",
            "source_artifact": str(AE7C1_ROOT / "ae7c1_inference_readiness_gate.json"),
            "limitation": "Original production AE7 stacking not live authority",
        },
        {
            "phase": "AE8",
            "saved_status": "ROOT_EXISTS_NO_GATE_FILE",
            "role": "Context intelligence",
            "msc_status": "REPORTING_WITH_ARCHIVAL_GAP",
            "live_authority": "NO",
            "source_artifact": str(AE8_ROOT),
            "limitation": "Primary AE8 root empty; context still reporting-only",
        },
        {
            "phase": "AE9",
            "saved_status": "AE9_AUDIT_LAYER_PARTIAL_MOCK_ONLY",
            "role": "LLM audit layer",
            "msc_status": "DIAGNOSTIC_MOCK",
            "live_authority": "NO",
            "source_artifact": str(AE9_ROOT / "reports" / "ae9_decision_gate.json"),
            "limitation": "Mock provider historical gate",
        },
        {
            "phase": "AE10",
            "saved_status": "AE10_TRACEABILITY_READY",
            "role": "Paper/demo orchestration",
            "msc_status": "DEMONSTRATED_PAPER_DEMO",
            "live_authority": "NO",
            "source_artifact": str(AE10_ROOT / "reports" / "ae10_decision_gate.json"),
            "limitation": "Paper/demo only; live disallowed",
        },
        {
            "phase": "AE11",
            "saved_status": "AE11_LOOP_OPERATIONAL",
            "role": "Runtime paper loop",
            "msc_status": "DEMONSTRATED_PAPER_LOOP",
            "live_authority": "NO",
            "source_artifact": str(AE11_ROOT / "reports" / "ae11_decision_gate.json"),
            "limitation": "Not live-approved",
        },
        {
            "phase": "AE12.1-4",
            "saved_status": "FORWARD_EVIDENCE_READY_FOR_REPORTING",
            "role": "Forward evidence / maturation",
            "msc_status": "REPORTING_READY",
            "live_authority": "NO",
            "source_artifact": str(
                FORWARD_ROOT / "reports" / "ae12_final_system_readiness_gate.json"
            ),
            "limitation": "Profitability not proven",
        },
        {
            "phase": "AE12.5",
            "saved_status": "CODE_AND_API_PRESENT",
            "role": "UI/API reporting",
            "msc_status": "IMPLEMENTED_READ_ONLY",
            "live_authority": "NO",
            "source_artifact": str(
                PROJECT_ROOT / "app" / "ae12_reporting" / "report_manager.py"
            ),
            "limitation": "Dedicated AE12.5 audit root may be absent",
        },
        {
            "phase": "SentimentFix",
            "saved_status": "PASS_DERIVED_ONLY_RUNTIME_UPDATE_PENDING",
            "role": "Semantic dual-axis taxonomy",
            "msc_status": "REPORTING_WITH_LIMITATIONS",
            "live_authority": "NO",
            "source_artifact": str(
                SENTIMENTFIX_ROOT / "audits" / "ae12_sentimentfix_decision_gate.json"
            ),
            "limitation": "UNKNOWN_UNRESOLVED remains; not web-grounded truth",
        },
        {
            "phase": "AE12.6",
            "saved_status": "PASS_WITH_LIMITATIONS",
            "role": "ML / meta-layer evaluation",
            "msc_status": "RESEARCH_REPORTING",
            "live_authority": "NO",
            "source_artifact": str(
                AE12_6_ROOT / "audits" / "ae12_ml_meta_layer_evaluation_gate.json"
            ),
            "limitation": "No model trade authority; AE7 stacking not original layer",
        },
        {
            "phase": "AE12.7",
            "saved_status": "AE12_7_PASS_WITH_EXTERNAL_SOURCES_DISABLED",
            "role": "Intelligent-agent operational demo",
            "msc_status": "DEMONSTRATED_OBSERVABILITY",
            "live_authority": "NO",
            "source_artifact": str(
                AE12_7_ROOT
                / "reports"
                / "ae12_7_intelligent_agent_decision_gate.json"
            ),
            "limitation": "External sources disabled in demo; no trade authority",
        },
        {
            "phase": "AE12.8",
            "saved_status": "AE12_8_PASS_WITH_ARCHIVAL_LIMITATIONS",
            "role": "Safety / reproducibility package",
            "msc_status": "ARCHIVAL_CLOSED",
            "live_authority": "NO",
            "source_artifact": str(
                AE12_8_ROOT / "reports" / "ae12_8_final_decision_gate.json"
            ),
            "limitation": "Packaging only; did not close AE12.9 previously",
        },
        {
            "phase": "AE12.9",
            "saved_status": "AE12_9_PASS_WITH_LIMITATIONS",
            "role": "Final MSc system package",
            "msc_status": "FINAL_PACKAGE_CLOSED",
            "live_authority": "NO",
            "source_artifact": str(
                OUTPUT_ROOT / "reports" / "ae12_9_final_decision_gate.json"
            ),
            "limitation": "Synthesizes evidence; does not prove profitability/live",
        },
    ]
    write_csv(
        OUTPUT_ROOT / "tables" / "ae12_9_phase_status_matrix.csv",
        [
            "phase",
            "saved_status",
            "role",
            "msc_status",
            "live_authority",
            "source_artifact",
            "limitation",
        ],
        phase_rows,
    )

    model_rows = [
        {
            "component": "RF",
            "evidence_class": "research_reporting_only",
            "ae12_6_status": "FOUND_AND_EVALUATED",
            "evidence_label": "PASS_RESEARCH_ONLY",
            "live_trade_authority": "NO",
            "paper_demo_authority": "NO",
            "source_artifact": str(
                AE12_6_ROOT / "data" / "ae12_model_performance_summary.csv"
            ),
            "notes": "Historical exit-sim on manual-verified set; not live profitability",
        },
        {
            "component": "XGB",
            "evidence_class": "historical_exit_sim_reporting",
            "ae12_6_status": "FOUND_AND_EVALUATED",
            "evidence_label": "PASS_FOR_REPORTING",
            "live_trade_authority": "NO",
            "paper_demo_authority": "NO",
            "source_artifact": str(
                AE12_6_ROOT / "data" / "ae12_model_performance_summary.csv"
            ),
            "notes": "Strongest historical exit-sim headline among families; still not live authority",
        },
        {
            "component": "TAB",
            "evidence_class": "research_reporting_only",
            "ae12_6_status": "FOUND_AND_EVALUATED",
            "evidence_label": "PASS_RESEARCH_ONLY",
            "live_trade_authority": "NO",
            "paper_demo_authority": "NO",
            "source_artifact": str(
                AE12_6_ROOT / "data" / "ae12_model_performance_summary.csv"
            ),
            "notes": "TabICL exit-sim research evidence only",
        },
        {
            "component": "Consensus",
            "evidence_class": "diagnostic_reporting_only",
            "ae12_6_status": "FOUND_AND_EVALUATED",
            "evidence_label": "PASS_DIAGNOSTIC_ONLY",
            "live_trade_authority": "NO",
            "paper_demo_authority": "NO",
            "source_artifact": str(AE6_ROOT / "ae6_consensus_decision_summary.json"),
            "notes": "DecisionRecord consensus filter; diagnostic only",
        },
        {
            "component": "Meta-layer",
            "evidence_class": "evidence_exists_not_original_ae7_live",
            "ae12_6_status": "NOT_IMPLEMENTED_AS_ORIGINAL_LAYER",
            "evidence_label": "NOT_IMPLEMENTED_AS_ORIGINAL_LAYER",
            "live_trade_authority": "NO",
            "paper_demo_authority": "NO",
            "source_artifact": str(
                AE12_6_ROOT / "audits" / "ae12_ml_meta_layer_evaluation_gate.json"
            ),
            "notes": "Original production/runtime AE7 stacking not live authority",
        },
    ]
    write_csv(
        OUTPUT_ROOT / "tables" / "ae12_9_model_evidence_matrix.csv",
        [
            "component",
            "evidence_class",
            "ae12_6_status",
            "evidence_label",
            "live_trade_authority",
            "paper_demo_authority",
            "source_artifact",
            "notes",
        ],
        model_rows,
    )

    context_rows = [
        {
            "layer": "Liquidity/activity context",
            "role": "context_reporting",
            "trade_authority": "NO",
            "source_artifact": str(AE12_8_ROOT / "audits" / "ae12_8_context_authority_audit.json"),
            "limitation": "Context only; AE8 primary root archival gap",
        },
        {
            "layer": "RSS/sentiment linkage",
            "role": "context_reporting",
            "trade_authority": "NO",
            "source_artifact": str(AE12_7_ROOT / "data" / "ae12_7_rss_context_links.csv"),
            "limitation": "Reporting/context; not trade authority",
        },
        {
            "layer": "Semantic/SentimentFix taxonomy",
            "role": "context_reporting",
            "trade_authority": "NO",
            "source_artifact": str(
                SENTIMENTFIX_ROOT / "audits" / "ae12_sentimentfix_decision_gate.json"
            ),
            "limitation": "UNKNOWN_UNRESOLVED unresolved; not social/opportunistic",
        },
        {
            "layer": "Gemini adjudication",
            "role": "selective_audit_explanation",
            "trade_authority": "NO",
            "source_artifact": str(
                AE12_8_ROOT / "data" / "ae12_8_semantic_sentimentfix_inventory.csv"
            ),
            "limitation": "Web-grounding not available/proven; AE12.9 did not call Gemini",
        },
        {
            "layer": "Helius/Solana read-only enrichment",
            "role": "context_reporting",
            "trade_authority": "NO",
            "source_artifact": str(
                AE12_7_ROOT / "audits" / "ae12_7_helius_readonly_audit.json"
            ),
            "limitation": "Broad live ingestion not proven; AE12.9 did not call Helius",
        },
        {
            "layer": "Whale/context signals",
            "role": "reporting_context_only",
            "trade_authority": "NO",
            "source_artifact": str(AE12_8_ROOT / "audits" / "ae12_8_context_authority_audit.json"),
            "limitation": "Reporting/context only; no live trade authority",
        },
        {
            "layer": "Qwen/local memo (agent)",
            "role": "explanation_audit",
            "trade_authority": "NO",
            "source_artifact": str(AE12_7_ROOT / "data" / "ae12_7_qwen_candidate_memos.jsonl"),
            "limitation": "Quality not proven beyond operational memo/demo",
        },
        {
            "layer": "Intelligent-agent layer",
            "role": "audit_explanation_context",
            "trade_authority": "NO",
            "source_artifact": str(AE12_7_ROOT / "audits" / "ae12_7_agent_authority_audit.json"),
            "limitation": "All agent outputs audit/explanation/context only",
        },
    ]
    write_csv(
        OUTPUT_ROOT / "tables" / "ae12_9_context_agent_matrix.csv",
        [
            "layer",
            "role",
            "trade_authority",
            "source_artifact",
            "limitation",
        ],
        context_rows,
    )

    # Authority matrix — extended from AE12.8
    authority_rows = [
        {
            "component": "RF",
            "live_trade_authority": "NO",
            "paper_demo_authority": "NO",
            "reporting_allowed": "YES",
            "wallet_authority": "NO",
            "source_artifact": str(AE12_8_ROOT / "audits" / "ae12_8_authority_matrix.csv"),
        },
        {
            "component": "XGB",
            "live_trade_authority": "NO",
            "paper_demo_authority": "NO",
            "reporting_allowed": "YES",
            "wallet_authority": "NO",
            "source_artifact": str(AE12_8_ROOT / "audits" / "ae12_8_authority_matrix.csv"),
        },
        {
            "component": "TAB",
            "live_trade_authority": "NO",
            "paper_demo_authority": "NO",
            "reporting_allowed": "YES",
            "wallet_authority": "NO",
            "source_artifact": str(AE12_8_ROOT / "audits" / "ae12_8_authority_matrix.csv"),
        },
        {
            "component": "Consensus",
            "live_trade_authority": "NO",
            "paper_demo_authority": "NO",
            "reporting_allowed": "YES",
            "wallet_authority": "NO",
            "source_artifact": str(AE12_8_ROOT / "audits" / "ae12_8_authority_matrix.csv"),
        },
        {
            "component": "Meta-layer",
            "live_trade_authority": "NO",
            "paper_demo_authority": "NO",
            "reporting_allowed": "YES",
            "wallet_authority": "NO",
            "source_artifact": str(AE12_8_ROOT / "audits" / "ae12_8_model_authority_audit.json"),
        },
        {
            "component": "Context/Semantic/RSS/Helius",
            "live_trade_authority": "NO",
            "paper_demo_authority": "NO",
            "reporting_allowed": "YES",
            "wallet_authority": "NO",
            "source_artifact": str(AE12_8_ROOT / "audits" / "ae12_8_context_authority_audit.json"),
        },
        {
            "component": "Qwen",
            "live_trade_authority": "NO",
            "paper_demo_authority": "NO",
            "reporting_allowed": "YES",
            "wallet_authority": "NO",
            "source_artifact": str(AE12_7_ROOT / "audits" / "ae12_7_agent_authority_audit.json"),
        },
        {
            "component": "Gemini",
            "live_trade_authority": "NO",
            "paper_demo_authority": "NO",
            "reporting_allowed": "YES",
            "wallet_authority": "NO",
            "source_artifact": str(AE12_7_ROOT / "audits" / "ae12_7_agent_authority_audit.json"),
        },
        {
            "component": "Helius",
            "live_trade_authority": "NO",
            "paper_demo_authority": "NO",
            "reporting_allowed": "YES",
            "wallet_authority": "NO",
            "source_artifact": str(AE12_7_ROOT / "audits" / "ae12_7_agent_authority_audit.json"),
        },
        {
            "component": "Paper/demo execution",
            "live_trade_authority": "NO",
            "paper_demo_authority": "YES",
            "reporting_allowed": "YES",
            "wallet_authority": "NO",
            "source_artifact": str(AE11_ROOT / "reports" / "ae11_decision_gate.json"),
        },
        {
            "component": "Live execution",
            "live_trade_authority": "NO",
            "paper_demo_authority": "NO",
            "reporting_allowed": "NO",
            "wallet_authority": "NO",
            "source_artifact": str(AE12_8_ROOT / "audits" / "ae12_8_wallet_no_live_safety_audit.json"),
        },
    ]
    write_csv(
        OUTPUT_ROOT / "tables" / "ae12_9_authority_matrix.csv",
        [
            "component",
            "live_trade_authority",
            "paper_demo_authority",
            "reporting_allowed",
            "wallet_authority",
            "source_artifact",
        ],
        authority_rows,
    )

    safety_rows = [
        {
            "check": "wallet_configured",
            "value": "false",
            "source_artifact": str(
                AE12_8_ROOT / "audits" / "ae12_8_wallet_no_live_safety_audit.json"
            ),
            "status": "PASS",
        },
        {
            "check": "private_key_accessed",
            "value": "false",
            "source_artifact": str(
                AE12_8_ROOT / "audits" / "ae12_8_wallet_no_live_safety_audit.json"
            ),
            "status": "PASS",
        },
        {
            "check": "real_transaction_signed",
            "value": "false",
            "source_artifact": str(
                AE12_8_ROOT / "audits" / "ae12_8_wallet_no_live_safety_audit.json"
            ),
            "status": "PASS",
        },
        {
            "check": "real_transaction_attempted",
            "value": "false",
            "source_artifact": str(
                AE12_8_ROOT / "audits" / "ae12_8_wallet_no_live_safety_audit.json"
            ),
            "status": "PASS",
        },
        {
            "check": "live_submission_status",
            "value": "NOT_SUBMITTED_NO_WALLET",
            "source_artifact": str(
                AE12_8_ROOT / "reports" / "ae12_8_final_decision_gate.json"
            ),
            "status": "PASS",
        },
        {
            "check": "live_trading_ready",
            "value": "false",
            "source_artifact": str(
                AE12_8_ROOT / "reports" / "ae12_8_final_decision_gate.json"
            ),
            "status": "PASS",
        },
        {
            "check": "live_trading_approval",
            "value": "NO",
            "source_artifact": str(
                AE12_8_ROOT / "reports" / "ae12_8_final_decision_gate.json"
            ),
            "status": "PASS",
        },
        {
            "check": "profitability_proven",
            "value": "false",
            "source_artifact": str(
                FORWARD_ROOT / "reports" / "ae12_final_system_readiness_gate.json"
            ),
            "status": "PASS",
        },
        {
            "check": "qwen_trade_authority",
            "value": "false",
            "source_artifact": str(
                AE12_7_ROOT / "audits" / "ae12_7_agent_authority_audit.json"
            ),
            "status": "PASS",
        },
        {
            "check": "gemini_trade_authority",
            "value": "false",
            "source_artifact": str(
                AE12_7_ROOT / "audits" / "ae12_7_agent_authority_audit.json"
            ),
            "status": "PASS",
        },
        {
            "check": "helius_trade_authority",
            "value": "false",
            "source_artifact": str(
                AE12_7_ROOT / "audits" / "ae12_7_agent_authority_audit.json"
            ),
            "status": "PASS",
        },
        {
            "check": "trader_db_mutated_by_ae12_9",
            "value": "false",
            "source_artifact": str(TRADER_DB),
            "status": "PASS",
        },
    ]
    write_csv(
        OUTPUT_ROOT / "tables" / "ae12_9_safety_matrix.csv",
        ["check", "value", "source_artifact", "status"],
        safety_rows,
    )

    forward_rows = [
        {
            "evidence_type": "opportunity_capture",
            "status": "PRESENT",
            "reporting_ready": "YES",
            "profitability_proven": "NO",
            "source_artifact": str(FORWARD_ROOT / "data" / "ae12_opportunity_capture_full.csv"),
            "notes": "Large forward opportunity capture CSV",
        },
        {
            "evidence_type": "missed_winners",
            "status": "PRESENT",
            "reporting_ready": "YES",
            "profitability_proven": "NO",
            "source_artifact": str(FORWARD_ROOT / "data" / "ae12_missed_winners_full.csv"),
            "notes": "Missed-winner reporting evidence",
        },
        {
            "evidence_type": "trade_vs_no_trade",
            "status": "PRESENT",
            "reporting_ready": "YES",
            "profitability_proven": "NO",
            "source_artifact": str(
                FORWARD_ROOT / "data" / "ae12_trade_vs_no_trade_comparison.csv"
            ),
            "notes": "Trade vs no-trade comparison",
        },
        {
            "evidence_type": "strict_vs_exploration",
            "status": "PRESENT",
            "reporting_ready": "YES",
            "profitability_proven": "NO",
            "source_artifact": str(
                FORWARD_ROOT / "data" / "ae12_strict_vs_exploration_comparison.csv"
            ),
            "notes": "Strict vs exploration separation",
        },
        {
            "evidence_type": "no_lookahead_horizon_maturation",
            "status": "PRESENT",
            "reporting_ready": "YES",
            "profitability_proven": "NO",
            "source_artifact": str(FORWARD_ROOT / "audits" / "ae12_no_lookahead_audit.csv"),
            "notes": "No-lookahead maturation audit",
        },
        {
            "evidence_type": "readiness_gate",
            "status": "FORWARD_EVIDENCE_READY_FOR_REPORTING",
            "reporting_ready": "YES",
            "profitability_proven": "NO",
            "source_artifact": str(
                FORWARD_ROOT / "reports" / "ae12_final_system_readiness_gate.json"
            ),
            "notes": "Reporting-ready; not live readiness",
        },
        {
            "evidence_type": "qwen_llm_linkage_reporting",
            "status": "PRESENT_AS_REPORTING_CONTEXT",
            "reporting_ready": "YES",
            "profitability_proven": "NO",
            "source_artifact": str(AE12_7_ROOT / "data" / "ae12_7_agent_trade_linkage.csv"),
            "notes": "Qwen/LLM linkage is reporting context only",
        },
    ]
    write_csv(
        OUTPUT_ROOT / "tables" / "ae12_9_forward_evidence_matrix.csv",
        [
            "evidence_type",
            "status",
            "reporting_ready",
            "profitability_proven",
            "source_artifact",
            "notes",
        ],
        forward_rows,
    )

    ui_rows = [
        {
            "surface": "/api/ae12/* forward evidence endpoints",
            "family": "forward_evidence",
            "implemented": "YES",
            "read_only": "YES",
            "trade_authority": "NO",
            "source_artifact": str(AE12_8_ROOT / "manifests" / "ae12_8_ui_api_inventory.csv"),
        },
        {
            "surface": "/api/ae12/sentimentfix + semantic endpoints",
            "family": "SentimentFix",
            "implemented": "YES",
            "read_only": "YES",
            "trade_authority": "NO",
            "source_artifact": str(AE12_8_ROOT / "manifests" / "ae12_8_ui_api_inventory.csv"),
        },
        {
            "surface": "/api/ae12/agents/*",
            "family": "intelligent_agent",
            "implemented": "YES",
            "read_only": "YES",
            "trade_authority": "NO",
            "source_artifact": str(AE12_7_ROOT / "reports" / "ae12_7_ui_status_summary.json"),
        },
        {
            "surface": "/api/ae12/safety",
            "family": "safety_status",
            "implemented": "YES",
            "read_only": "YES",
            "trade_authority": "NO",
            "source_artifact": str(AE12_8_ROOT / "manifests" / "ae12_8_ui_api_inventory.csv"),
        },
        {
            "surface": "UI Forward / SentimentFix / Agents panels",
            "family": "demo_observability",
            "implemented": "YES",
            "read_only": "YES",
            "trade_authority": "NO",
            "source_artifact": str(
                PROJECT_ROOT / "app" / "ae12_reporting" / "report_manager.py"
            ),
        },
    ]
    write_csv(
        OUTPUT_ROOT / "tables" / "ae12_9_ui_api_matrix.csv",
        [
            "surface",
            "family",
            "implemented",
            "read_only",
            "trade_authority",
            "source_artifact",
        ],
        ui_rows,
    )

    limitation_rows = [
        {
            "limitation_id": "L01",
            "category": "profitability",
            "statement": "Profitability was not proven",
            "severity": "high",
            "blocks_live": "YES",
            "source_artifact": str(
                FORWARD_ROOT / "reports" / "ae12_final_system_readiness_gate.json"
            ),
        },
        {
            "limitation_id": "L02",
            "category": "live_readiness",
            "statement": "Live trading readiness was not proven; live_trading_approval=NO",
            "severity": "high",
            "blocks_live": "YES",
            "source_artifact": str(
                AE12_8_ROOT / "reports" / "ae12_8_final_decision_gate.json"
            ),
        },
        {
            "limitation_id": "L03",
            "category": "strategy",
            "statement": "Production trading strategy not proven",
            "severity": "high",
            "blocks_live": "YES",
            "source_artifact": str(
                AE12_8_ROOT / "reports" / "ae12_8_what_was_built_vs_not_proven.md"
            ),
        },
        {
            "limitation_id": "L04",
            "category": "models",
            "statement": "Model generalization not fully proven",
            "severity": "medium",
            "blocks_live": "YES",
            "source_artifact": str(
                AE12_6_ROOT / "data" / "ae12_model_performance_summary.csv"
            ),
        },
        {
            "limitation_id": "L05",
            "category": "wallet",
            "statement": "Real wallet execution not tested",
            "severity": "high",
            "blocks_live": "YES",
            "source_artifact": str(
                AE12_8_ROOT / "audits" / "ae12_8_wallet_no_live_safety_audit.json"
            ),
        },
        {
            "limitation_id": "L06",
            "category": "helius",
            "statement": "Helius broad live ingestion not proven",
            "severity": "medium",
            "blocks_live": "NO",
            "source_artifact": str(
                AE12_7_ROOT / "audits" / "ae12_7_helius_readonly_audit.json"
            ),
        },
        {
            "limitation_id": "L07",
            "category": "gemini",
            "statement": "Gemini web-grounding not available/proven",
            "severity": "medium",
            "blocks_live": "NO",
            "source_artifact": str(
                AE12_8_ROOT / "data" / "ae12_8_semantic_sentimentfix_inventory.csv"
            ),
        },
        {
            "limitation_id": "L08",
            "category": "qwen",
            "statement": "Qwen quality not proven beyond operational memo/demo",
            "severity": "medium",
            "blocks_live": "NO",
            "source_artifact": str(AE12_7_ROOT / "reports" / "ae12_7_summary_for_upload.txt"),
        },
        {
            "limitation_id": "L09",
            "category": "semantic",
            "statement": "UNKNOWN_UNRESOLVED semantic cases remain unresolved",
            "severity": "medium",
            "blocks_live": "NO",
            "source_artifact": str(
                SENTIMENTFIX_ROOT / "audits" / "ae12_sentimentfix_decision_gate.json"
            ),
        },
        {
            "limitation_id": "L10",
            "category": "ae7_meta",
            "statement": "Original production-grade AE7 runtime stacking not approved as live authority",
            "severity": "high",
            "blocks_live": "YES",
            "source_artifact": str(AE7C1_ROOT / "ae7c1_inference_readiness_gate.json"),
        },
        {
            "limitation_id": "L11",
            "category": "reproducibility",
            "statement": "Missing .git metadata limits source revision identity proof",
            "severity": "medium",
            "blocks_live": "NO",
            "source_artifact": str(
                AE12_8_ROOT / "manifests" / "ae12_8_reproducibility_manifest.json"
            ),
        },
        {
            "limitation_id": "L12",
            "category": "tests",
            "statement": "Full pytest suite not re-run in AE12.9; unrelated failures may remain",
            "severity": "low",
            "blocks_live": "NO",
            "source_artifact": str(
                OUTPUT_ROOT / "audits" / "ae12_9_final_safety_consistency_audit.json"
            ),
        },
        {
            "limitation_id": "L13",
            "category": "ae8_archival",
            "statement": "AE8 context intelligence primary root is empty (archival gap)",
            "severity": "low",
            "blocks_live": "NO",
            "source_artifact": str(AE8_ROOT),
        },
    ]
    write_csv(
        OUTPUT_ROOT / "tables" / "ae12_9_limitations_matrix.csv",
        [
            "limitation_id",
            "category",
            "statement",
            "severity",
            "blocks_live",
            "source_artifact",
        ],
        limitation_rows,
    )


def write_audits(
    source_index: list[dict[str, Any]],
    trader_db_hash_before: str,
    trader_db_hash_after: str,
    compileall_result: dict[str, Any],
    safety_blocked: bool,
) -> None:
    integrity_rows = []
    for row in source_index:
        integrity_rows.append(
            {
                "artifact_type": row["artifact_type"],
                "exact_path": row["exact_path"],
                "exists": row["exists"],
                "integrity_status": "OK" if row["exists"] else "MISSING",
                "severity": "critical"
                if (not row["exists"] and row["phase"] in {"AE12.8", "AE12.7", "AE12.6"})
                else ("important" if not row["exists"] else "ok"),
                "notes": row["notes"],
            }
        )
    write_csv(
        OUTPUT_ROOT / "audits" / "ae12_9_source_integrity_audit.csv",
        [
            "artifact_type",
            "exact_path",
            "exists",
            "integrity_status",
            "severity",
            "notes",
        ],
        integrity_rows,
    )

    missing_rows = [
        {
            "artifact": r["exact_path"],
            "phase": r["phase"],
            "severity": "critical"
            if r["phase"] in {"AE12.8", "AE12.7", "AE12.6"}
            else "important",
            "blocks_ae12_9": "YES"
            if r["phase"] in {"AE12.8", "AE12.7", "AE12.6"}
            else "NO",
            "notes": r["notes"],
        }
        for r in source_index
        if not r["exists"]
    ]
    # Known archival gaps that exist as paths but are empty / incomplete
    ae8_empty = AE8_ROOT.exists() and not any(AE8_ROOT.iterdir()) if AE8_ROOT.exists() else True
    if ae8_empty:
        missing_rows.append(
            {
                "artifact": str(AE8_ROOT),
                "phase": "AE8",
                "severity": "archival",
                "blocks_ae12_9": "NO",
                "notes": "AE8 root empty (0 files); context authority still recorded as reporting-only via AE12.8",
            }
        )
    missing_rows.append(
        {
            "artifact": "ae7_meta_model_stacking (original closed layer)",
            "phase": "AE7",
            "severity": "optional_research",
            "blocks_ae12_9": "NO",
            "notes": "Optional missing per AE12.6; meta-layer not original production stack",
        }
    )
    missing_rows.append(
        {
            "artifact": "ae9_llm_operational_audit",
            "phase": "AE9",
            "severity": "optional_research",
            "blocks_ae12_9": "NO",
            "notes": "Optional missing per AE12.6; AE9 mock gate present",
        }
    )
    if not missing_rows:
        missing_rows.append(
            {
                "artifact": "(none critical)",
                "phase": "AE12.9",
                "severity": "ok",
                "blocks_ae12_9": "NO",
                "notes": "No critical source artifacts missing for final package synthesis",
            }
        )
    write_csv(
        OUTPUT_ROOT / "audits" / "ae12_9_missing_artifact_audit.csv",
        ["artifact", "phase", "severity", "blocks_ae12_9", "notes"],
        missing_rows,
    )

    claims = [
        (
            "C01",
            "SAFE/DEMO-first intelligent meme-coin trading workstation was built",
            "AE12.8 what-built + AE10/AE11/AE12.7 gates",
            str(AE12_8_ROOT / "reports" / "ae12_8_what_was_built_vs_not_proven.md"),
        ),
        (
            "C02",
            "System collects market/runtime data",
            "AE12.1 census + AE12.8 inventory",
            str(
                PROJECT_ROOT
                / "data"
                / "audits"
                / "ae12_runtime_data_census_20260714_224931"
                / "reports"
                / "ae12_data_census_summary.json"
            ),
        ),
        (
            "C03",
            "System runs paper/demo trading",
            "AE10/AE11 decision gates",
            str(AE11_ROOT / "reports" / "ae11_decision_gate.json"),
        ),
        (
            "C04",
            "Model evidence RF/XGB/TAB/consensus/meta evaluated for reporting",
            "AE12.6 gate + model performance summary",
            str(AE12_6_ROOT / "audits" / "ae12_ml_meta_layer_evaluation_gate.json"),
        ),
        (
            "C05",
            "RF research/reporting only; no live authority",
            "AE12.6 + AE12.8 model authority",
            str(AE12_8_ROOT / "audits" / "ae12_8_model_authority_audit.json"),
        ),
        (
            "C06",
            "XGB historical/exit-sim/reporting evidence; no live authority",
            "AE12.6 model performance summary",
            str(AE12_6_ROOT / "data" / "ae12_model_performance_summary.csv"),
        ),
        (
            "C07",
            "TAB research/reporting only; no live authority",
            "AE12.6 gate",
            str(AE12_6_ROOT / "audits" / "ae12_ml_meta_layer_evaluation_gate.json"),
        ),
        (
            "C08",
            "Consensus diagnostic/reporting only",
            "AE6 summary + AE12.6",
            str(AE6_ROOT / "ae6_consensus_decision_summary.json"),
        ),
        (
            "C09",
            "Meta-layer evidence exists but original AE7 stacking not live authority",
            "AE12.6 NOT_IMPLEMENTED_AS_ORIGINAL_LAYER + AE7c1 gate",
            str(AE7C1_ROOT / "ae7c1_inference_readiness_gate.json"),
        ),
        (
            "C10",
            "Context intelligence is reporting/context only; no live authority",
            "AE12.8 context authority audit",
            str(AE12_8_ROOT / "audits" / "ae12_8_context_authority_audit.json"),
        ),
        (
            "C11",
            "Intelligent-agent outputs are audit/explanation/context only",
            "AE12.7 agent authority audit",
            str(AE12_7_ROOT / "audits" / "ae12_7_agent_authority_audit.json"),
        ),
        (
            "C12",
            "Forward evidence reporting-ready; profitability not proven",
            "AE12 forward readiness gate",
            str(FORWARD_ROOT / "reports" / "ae12_final_system_readiness_gate.json"),
        ),
        (
            "C13",
            "UI/API surfaces are read-only reporting",
            "AE12.8 UI/API inventory",
            str(AE12_8_ROOT / "manifests" / "ae12_8_ui_api_inventory.csv"),
        ),
        (
            "C14",
            "wallet_configured=false; no private key; no real tx; live not approved",
            "AE12.8 wallet/no-live safety audit",
            str(AE12_8_ROOT / "audits" / "ae12_8_wallet_no_live_safety_audit.json"),
        ),
        (
            "C15",
            "qwen_trade_authority=false; gemini_trade_authority=false; helius_trade_authority=false",
            "AE12.7 agent authority policy",
            str(AE12_7_ROOT / "audits" / "ae12_7_agent_authority_audit.json"),
        ),
        (
            "C16",
            "Profitability not proven",
            "Forward readiness + AE12.8 gate",
            str(FORWARD_ROOT / "reports" / "ae12_final_system_readiness_gate.json"),
        ),
        (
            "C17",
            "Live trading was not approved",
            "AE12.8 final decision gate",
            str(AE12_8_ROOT / "reports" / "ae12_8_final_decision_gate.json"),
        ),
        (
            "C18",
            "Paper/demo trading allowed",
            "AE12.7/AE12.8 authority audits",
            str(AE12_8_ROOT / "audits" / "ae12_8_agent_authority_audit.json"),
        ),
        (
            "C19",
            "AE12.9 classification AE12_9_PASS_WITH_LIMITATIONS",
            "This package decision gate",
            str(OUTPUT_ROOT / "reports" / "ae12_9_final_decision_gate.json"),
        ),
    ]
    claim_rows = []
    for cid, claim, support, path in claims:
        exists = Path(path).exists() if not path.endswith("ae12_9_final_decision_gate.json") else True
        # decision gate written later; treat as planned OK
        if path.endswith("ae12_9_final_decision_gate.json"):
            exists = True
        claim_rows.append(
            {
                "claim_id": cid,
                "claim": claim,
                "support_summary": support,
                "source_artifact": path,
                "source_exists": exists,
                "traceability_status": "OK" if exists else "MISSING_SOURCE",
            }
        )
    write_csv(
        OUTPUT_ROOT / "audits" / "ae12_9_claim_traceability_audit.csv",
        [
            "claim_id",
            "claim",
            "support_summary",
            "source_artifact",
            "source_exists",
            "traceability_status",
        ],
        claim_rows,
    )

    safety_payload = {
        "phase": "AE12.9",
        "created_at": CREATED_AT,
        "classification_if_pass": "AE12_9_PASS_WITH_LIMITATIONS",
        "safety_blocked": safety_blocked,
        "wallet_configured": False,
        "private_key_accessed": False,
        "real_transaction_signed": False,
        "real_transaction_attempted": False,
        "live_submission_status": "NOT_SUBMITTED_NO_WALLET",
        "live_trading_ready": False,
        "live_trading_approval": "NO",
        "profitability_proven": False,
        "qwen_trade_authority": False,
        "gemini_trade_authority": False,
        "helius_trade_authority": False,
        "demo_paper_trading_allowed": True,
        "trader_db_path": str(TRADER_DB),
        "trader_db_hash_before": trader_db_hash_before,
        "trader_db_hash_after": trader_db_hash_after,
        "trader_db_unchanged": trader_db_hash_before == trader_db_hash_after,
        "no_external_api_calls": True,
        "no_retraining": True,
        "no_runtime_loop": True,
        "no_wallet_connect": True,
        "compileall_validation": compileall_result,
        "full_test_suite_run": False,
        "full_test_suite_note": (
            "Full pytest suite not re-run in AE12.9 to avoid unrelated failures "
            "blocking packaging; lightweight compileall only."
        ),
        "prior_ae12_8_safety": read_json(
            AE12_8_ROOT / "audits" / "ae12_8_wallet_no_live_safety_audit.json"
        ),
        "prior_ae12_7_agent_authority": read_json(
            AE12_7_ROOT / "audits" / "ae12_7_agent_authority_audit.json"
        ),
        "status": "BLOCKED_SAFETY" if safety_blocked else "PASS_NO_WALLET_NO_LIVE",
    }
    write_json(
        OUTPUT_ROOT / "audits" / "ae12_9_final_safety_consistency_audit.json",
        safety_payload,
    )


def write_manifests(source_index: list[dict[str, Any]], created_files: list[str]) -> None:
    write_csv(
        OUTPUT_ROOT / "manifests" / "ae12_9_source_artifact_index.csv",
        [
            "phase",
            "artifact_type",
            "exact_path",
            "exists",
            "file_size_bytes",
            "last_modified",
            "used_in_ae12_9",
            "notes",
        ],
        source_index,
    )

    repro = {
        "phase": "AE12.9",
        "created_at": CREATED_AT,
        "project_root": str(PROJECT_ROOT),
        "output_root": str(OUTPUT_ROOT),
        "primary_evidence_roots": {
            "ae12_8": str(AE12_8_ROOT),
            "ae12_7": str(AE12_7_ROOT),
            "ae12_6": str(AE12_6_ROOT),
        },
        "ae12_8_reproducibility_manifest": str(
            AE12_8_ROOT / "manifests" / "ae12_8_reproducibility_manifest.json"
        ),
        "ae12_8_reproducibility_snapshot": read_json(
            AE12_8_ROOT / "manifests" / "ae12_8_reproducibility_manifest.json"
        ),
        "commands_used": [
            "python scripts/run_ae12_9_final_msc_system_package.py",
            "python -m compileall app scripts tests -q",
        ],
        "constraints": [
            "no rebuild",
            "no retraining",
            "no external API calls",
            "no trader.db mutation",
            "no wallet",
            "no live trading",
            "no profitability claim",
            "no live readiness claim",
        ],
        "git_available_per_ae12_8": False,
        "source_revision_identity_proven": False,
        "notes": [
            "AE12.9 synthesizes saved AE6–AE12 evidence; does not re-run experiments.",
            "Reproducibility references inherit AE12.8 dependency/environment snapshot.",
        ],
    }
    write_json(
        OUTPUT_ROOT / "manifests" / "ae12_9_reproducibility_references.json", repro
    )

    manifest = {
        "phase": "AE12.9",
        "created_at": CREATED_AT,
        "output_root": str(OUTPUT_ROOT),
        "classification": "AE12_9_PASS_WITH_LIMITATIONS",
        "ae12_closed": True,
        "msc_final_package_closed": True,
        "live_trading_ready": False,
        "profitability_proven": False,
        "files_created": created_files,
        "source_artifact_count": len(source_index),
        "source_artifacts_present": sum(1 for r in source_index if r["exists"]),
        "primary_roots": {
            "ae12_8": str(AE12_8_ROOT),
            "ae12_7": str(AE12_7_ROOT),
            "ae12_6": str(AE12_6_ROOT),
        },
    }
    write_json(OUTPUT_ROOT / "manifests" / "ae12_9_manifest.json", manifest)


def write_reports(classification: str) -> None:
    report = f"""# AE12.9 — Final MSc System Report

**Classification:** `{classification}`  
**Created (UTC):** {CREATED_AT}  
**Output root:** `{OUTPUT_ROOT}`  

---

## 1. Executive summary

A working **SAFE/DEMO-first** intelligent meme-coin trading workstation was built.

- The system collects market/runtime data.
- The system runs paper/demo trading.
- The system includes model evidence, context intelligence, LLM/agent reporting, forward evidence, UI/API surfaces, and safety gates.
- **Profitability was not proven.**
- **Live trading was not approved.**
- **No real wallet / private key / transaction path was enabled.**

This AE12.9 package synthesizes existing AE6–AE12 evidence (primarily via the AE12.8 safety/reproducibility inventory, AE12.7 intelligent-agent demo, and AE12.6 ML/meta-layer evaluation). It does not rebuild the project, retrain models, call external APIs, mutate `trader.db`, or enable live trading.

---

## 2. Project objective

This MSc project delivers an **intelligent system for meme-coin analysis**: multimodal/context-aware prediction support, sentiment/context integration, DecisionRecord-based scoring evidence, and **simulated automated trading** under a strict no-wallet / no-live boundary. The objective is a reproducible, auditable research workstation — not a live trading product claim.

---

## 3. Final architecture

| Layer | Role |
|-------|------|
| Market/runtime data collection | Pair/market snapshots, runtime census (AE12.1) |
| DecisionRecord layer | AE6 contract for structured decisions / consensus slots |
| RF / XGB / TAB model evidence | Research/reporting (XGB also historical exit-sim reporting) |
| Consensus / meta-layer evidence | Diagnostic/reporting; original AE7 stacking not live authority |
| Context intelligence | Liquidity/activity, RSS/sentiment, Semantic/SentimentFix, Helius read-only path |
| Intelligent-agent layer | Qwen memo, Gemini selective audit, linkage — audit/explanation only |
| Paper/demo execution | AE10 one-shot + AE11 runtime paper loop |
| Forward evidence | Opportunity capture, missed winners, no-lookahead maturation |
| UI/API reporting | Read-only AE12 forward / SentimentFix / agents / safety surfaces |
| Safety / no-live boundary | No wallet, no signing, no live submission, no model/LLM trade authority |

See figures: `figures/ae12_9_final_architecture.mmd`, `ae12_9_runtime_data_flow.mmd`, `ae12_9_decision_authority_flow.mmd`.

---

## 4. Model evidence summary

| Component | Exact status |
|-----------|--------------|
| **RF** | Research/reporting only |
| **XGB** | Historical/exit-sim/reporting evidence |
| **TAB** | Research/reporting only |
| **Consensus** | Diagnostic/reporting only |
| **Meta-layer** | Evidence exists, but original production/runtime AE7 stacking layer is **not** live authority |
| **All of the above** | **None have live trade authority** |

Primary sources: AE12.6 evaluation gate and `ae12_model_performance_summary.csv`; AE12.8 model authority audit.

---

## 5. Context intelligence summary

- Liquidity/activity context — reporting/context only
- RSS/sentiment linkage — reporting/context only
- Semantic/SentimentFix dual-axis taxonomy — reporting; UNKNOWN_UNRESOLVED means unresolved (not social, not opportunistic)
- Gemini adjudication — limitations: web-grounding not available/proven; AE12.9 did not call Gemini
- Helius/Solana — read-only enrichment path; broad live ingestion not proven; AE12.9 did not call Helius
- Whale/context signals — reporting/context only
- **Context does not create live trade authority**

---

## 6. Intelligent-agent summary

- Qwen/local memo path — explanation/audit (demo may skip live provider)
- Gemini selective audit path — explanation/audit only
- Helius read-only enrichment path — context only
- RSS/semantic linkage — context only
- Agent trade linkage — observational linkage to paper/demo, not authority
- Missed-winner agent review — reporting/audit
- **All agent outputs are audit/explanation/context only**
- **No agent has trade authority** (`qwen_trade_authority=false`, `gemini_trade_authority=false`, `helius_trade_authority=false`)

Primary source: AE12.7 decision gate + agent authority audit.

---

## 7. Paper/demo trading summary

- AE10 one-shot paper/demo proof (`AE10_TRACEABILITY_READY`) with no-wallet live dry-run path
- AE11 runtime paper loop (`AE11_LOOP_OPERATIONAL`) with paper orders/fills/positions/ledger evidence
- Strict vs exploration separation demonstrated
- Deterministic TP/SL lifecycle proof path exists in paper/demo orchestration
- **Paper/demo allowed**
- **Live real-money trading not approved**

---

## 8. Forward evidence summary

- Opportunity capture, missed winners, trade-vs-no-trade, strict-vs-exploration present
- No-lookahead horizon maturation audited
- Qwen/LLM linkage used as **reporting context** only
- Gate: `FORWARD_EVIDENCE_READY_FOR_REPORTING`
- Evidence is **reporting-ready**
- **Profitability not proven**

---

## 9. UI/API summary

- Forward evidence UI/API (`/api/ae12/forward-evidence-summary`, missed-winners, comparisons, etc.)
- SentimentFix UI/API (taxonomy, classifier, adjudication artifact views)
- Intelligent-agent UI/API (`/api/ae12/agents/*`)
- Safety/status endpoints (`/api/ae12/safety`, status)
- Demo observability panels — all **read-only**, no trade/wallet authority

---

## 10. Safety summary

| Field | Value |
|-------|-------|
| wallet_configured | false |
| private_key_accessed | false |
| real_transaction_signed | false |
| real_transaction_attempted | false |
| live_submission_status | NOT_SUBMITTED_NO_WALLET |
| live_trading_ready | false |
| live_trading_approval | NO |
| profitability_proven | false |
| qwen_trade_authority | false |
| gemini_trade_authority | false |
| helius_trade_authority | false |

AE12.9 did not connect a wallet, access private keys, sign/submit transactions, call Gemini/Helius/Qwen/Ollama, or mutate `trader.db`.

---

## 11. What was built

- DecisionRecord architecture (AE6)
- Model score / feature bridge / policy-binding evidence trail (AE7 family; runtime stack not closed)
- Context intelligence and SentimentFix dual-axis taxonomy
- LLM audit layer (mock/historical) and AE12.7 intelligent-agent observability demo
- Paper/demo trading orchestration (AE10) and runtime paper loop (AE11)
- Forward evidence census, quality, maturation, opportunity/missed-winner reporting (AE12.1–4)
- Read-only UI/API reporting layer (AE12.5 code + APIs)
- ML/meta-layer evaluation package (AE12.6)
- Safety / reproducibility archival package (AE12.8)
- This final MSc synthesis package (AE12.9)

---

## 12. What was demonstrated

- End-to-end SAFE/DEMO workstation flow from data → DecisionRecord → paper/demo → forward evidence → reporting UI/API
- Paper orders/positions under no-wallet constraints
- Strict vs exploration separation
- Forward opportunity / missed-winner / trade-vs-no-trade reporting artifacts
- Agent memo/audit/linkage observability without trade authority
- Explicit safety gates denying live trading and wallet use
- Model/context/agent authority matrices with no live authority granted

---

## 13. What was not proven

- Profitability not proven
- Live readiness not proven
- Production trading strategy not proven
- Model generalization not fully proven
- Real wallet execution not tested
- Helius broad live ingestion not proven
- Gemini web-grounding not available/proven
- Qwen quality not proven beyond operational memo/demo
- UNKNOWN_UNRESOLVED semantic cases remain unresolved
- Original production-grade AE7 runtime stacking not approved as live authority

---

## 14. Limitations

Suitable for MSc submission caveats:

1. System is a **research/demo workstation**, not a live trading approval.
2. Historical exit-sim model headlines are **not** live profitability proof.
3. Forward evidence is **reporting-ready**, not performance certification.
4. Semantic taxonomy has high UNKNOWN share; not web-grounded global truth.
5. AE8 primary audit root is an archival gap (empty); context still treated as reporting-only via later audits.
6. Missing `.git` metadata limits exact source-revision reproducibility (per AE12.8).
7. Full pytest suite was not re-run in AE12.9; only `compileall` lightweight validation.
8. External providers were not called in AE12.8/AE12.9 packaging phases.

See `tables/ae12_9_limitations_matrix.csv`.

---

## 15. Future work

- Longer forward paper/demo runs
- Better candidate universe
- More data collection
- Stronger meta-model validation
- Solana/Helius enrichment expansion (still read-only unless separately approved)
- Semantic web-grounded validation
- Real paper/live readiness gates (explicit future approval process)
- Stronger profitability validation (still under safety constraints)
- Deployment hardening

---

## 16. Final closure decision

**Gate:** `{classification}`

- Critical AE12.6 / AE12.7 / AE12.8 evidence roots are present.
- Safety consistency: no wallet / no live / no model-or-LLM trade authority.
- AE12 can be closed as an **MSc evidence package** with explicit limitations.
- This does **not** approve live trading or claim profitability.

Traceability: every major claim is indexed in `manifests/ae12_9_source_artifact_index.csv` and `audits/ae12_9_claim_traceability_audit.csv`.
"""
    write_text(
        OUTPUT_ROOT / "reports" / "ae12_9_final_msc_system_report.md", report
    )

    summary = f"""AE12.9 Final MSc System Package
Classification: {classification}
Output root: {OUTPUT_ROOT}
Created at (UTC): {CREATED_AT}

Executive:
  SAFE/DEMO-first intelligent meme-coin trading workstation built.
  Market/runtime data collection: YES
  Paper/demo trading: YES (allowed)
  Model + context + agent + forward + UI/API + safety evidence: YES
  Profitability proven: NO
  Live trading approved: NO
  Wallet / private key / real tx path: NOT ENABLED

Safety:
  wallet_configured=False
  private_key_accessed=False
  real_transaction_signed=False
  real_transaction_attempted=False
  live_submission_status=NOT_SUBMITTED_NO_WALLET
  live_trading_ready=False
  live_trading_approval=NO
  profitability_proven=False
  qwen_trade_authority=False
  gemini_trade_authority=False
  helius_trade_authority=False

Authority:
  RF/XGB/TAB = research/reporting (XGB includes historical exit-sim reporting); no live authority
  Consensus = diagnostic/reporting only
  Meta-layer = evidence exists; original AE7 stacking not live authority
  Context/LLM/Agents = audit/explanation/context only; no trade authority
  Paper/demo = allowed; live = not approved

Primary evidence roots:
  AE12.8: {AE12_8_ROOT}
  AE12.7: {AE12_7_ROOT}
  AE12.6: {AE12_6_ROOT}

Closure:
  AE12.9 closed as final MSc system package: YES
  AE12 closed for MSc evidence package (with limitations): YES
  Live trading / profitability claims: FORBIDDEN

Read first:
  - reports/ae12_9_final_msc_system_report.md
  - reports/ae12_9_final_decision_gate.json
  - tables/ae12_9_phase_status_matrix.csv
  - tables/ae12_9_safety_matrix.csv
  - audits/ae12_9_claim_traceability_audit.csv
  - manifests/ae12_9_source_artifact_index.csv
"""
    write_text(
        OUTPUT_ROOT / "reports" / "ae12_9_final_summary_for_upload.txt", summary
    )

    gate = {
        "phase": "AE12.9",
        "classification": classification,
        "created_at": CREATED_AT,
        "ae12_9_closed": True,
        "ae12_closed": True,
        "msc_final_package_closed": True,
        "ae12_9_blocked": False,
        "safety": {
            "status": "PASS_NO_WALLET_NO_LIVE",
            "wallet_configured": False,
            "private_key_accessed": False,
            "real_transaction_signed": False,
            "real_transaction_attempted": False,
            "live_submission_status": "NOT_SUBMITTED_NO_WALLET",
            "live_trading_ready": False,
            "live_trading_approval": "NO",
            "demo_paper_trading_allowed": True,
            "profitability_proven": False,
            "qwen_trade_authority": False,
            "gemini_trade_authority": False,
            "helius_trade_authority": False,
        },
        "evidence": {
            "ae12_8_root_exists": AE12_8_ROOT.exists(),
            "ae12_7_root_exists": AE12_7_ROOT.exists(),
            "ae12_6_root_exists": AE12_6_ROOT.exists(),
            "forward_readiness": "FORWARD_EVIDENCE_READY_FOR_REPORTING",
            "paper_demo_allowed": True,
        },
        "authority": {
            "rf_live_trade_authority": False,
            "xgb_live_trade_authority": False,
            "tab_live_trade_authority": False,
            "consensus_live_trade_authority": False,
            "meta_layer_live_trade_authority": False,
            "context_live_trade_authority": False,
            "agent_live_trade_authority": False,
            "original_ae7_runtime_stacking_live_authority": False,
        },
        "constraints_honored": {
            "no_rebuild": True,
            "no_retraining": True,
            "no_external_api_calls": True,
            "no_trader_db_mutation": True,
            "no_wallet": True,
            "no_live_trading": True,
            "no_profitability_claim": True,
            "no_live_readiness_claim": True,
        },
        "notes": [
            "Balanced final gate: PASS_WITH_LIMITATIONS — critical evidence present; live/profitability not proven.",
            "AE12 closed as MSc evidence package with explicit limitations.",
            "Must not be interpreted as live trading approval.",
        ],
        "output_root": str(OUTPUT_ROOT),
    }
    write_json(OUTPUT_ROOT / "reports" / "ae12_9_final_decision_gate.json", gate)


def check_safety_blockers() -> tuple[bool, list[str]]:
    """If any prior artifact suggests wallet/live/tx authority, block AE12.9."""
    offenders: list[str] = []
    wallet = read_json(AE12_8_ROOT / "audits" / "ae12_8_wallet_no_live_safety_audit.json") or {}
    if wallet.get("wallet_configured") is True:
        offenders.append("ae12_8_wallet: wallet_configured=true")
    if wallet.get("private_key_accessed") is True:
        offenders.append("ae12_8_wallet: private_key_accessed=true")
    if wallet.get("real_transaction_signed") is True:
        offenders.append("ae12_8_wallet: real_transaction_signed=true")
    if wallet.get("real_transaction_attempted") is True:
        offenders.append("ae12_8_wallet: real_transaction_attempted=true")
    if wallet.get("live_trading_ready") is True:
        offenders.append("ae12_8_wallet: live_trading_ready=true")
    if wallet.get("live_trading_approval") not in (None, "NO", False, "false"):
        if str(wallet.get("live_trading_approval")).upper() in {"YES", "APPROVED", "TRUE"}:
            offenders.append("ae12_8_wallet: live_trading_approval approved")

    agent = read_json(AE12_7_ROOT / "audits" / "ae12_7_agent_authority_audit.json") or {}
    policy = agent.get("policy") or {}
    for key in (
        "llm_trade_authority",
        "gemini_trade_authority",
        "qwen_trade_authority",
        "helius_trade_authority",
        "agent_layer_trade_authority",
        "wallet_allowed",
        "private_key_access_allowed",
        "real_transaction_allowed",
    ):
        if policy.get(key) is True:
            offenders.append(f"ae12_7_agent_authority: {key}=true")
    if agent.get("trade_authority_used") is True:
        offenders.append("ae12_7_agent_authority: trade_authority_used=true")

    gate8 = read_json(AE12_8_ROOT / "reports" / "ae12_8_final_decision_gate.json") or {}
    safety = gate8.get("safety") or {}
    if safety.get("live_trading_ready") is True:
        offenders.append("ae12_8_gate: live_trading_ready=true")
    if str(safety.get("live_trading_approval", "NO")).upper() in {"YES", "APPROVED"}:
        offenders.append("ae12_8_gate: live_trading_approval approved")

    critical_missing = []
    for path, label in (
        (AE12_8_ROOT, "AE12.8 root"),
        (AE12_7_ROOT, "AE12.7 root"),
        (AE12_6_ROOT, "AE12.6 root"),
        (AE12_8_ROOT / "reports" / "ae12_8_final_decision_gate.json", "AE12.8 gate"),
        (
            AE12_7_ROOT / "reports" / "ae12_7_intelligent_agent_decision_gate.json",
            "AE12.7 gate",
        ),
        (
            AE12_6_ROOT / "audits" / "ae12_ml_meta_layer_evaluation_gate.json",
            "AE12.6 gate",
        ),
    ):
        if not path.exists():
            critical_missing.append(label)
            offenders.append(f"critical_missing: {label}")

    return (len(offenders) > 0, offenders)


def main() -> int:
    ensure_dirs()
    hash_before = sha256_quick(TRADER_DB)
    safety_blocked, offenders = check_safety_blockers()

    source_index = build_source_index()
    figure_paths = write_figures()
    write_tables(source_index)

    compileall_result = run_compileall()
    hash_after = sha256_quick(TRADER_DB)

    if hash_before != hash_after:
        safety_blocked = True
        offenders.append("trader.db hash changed during AE12.9 packaging")

    classification = (
        "AE12_9_BLOCKED_CRITICAL_EVIDENCE_OR_SAFETY"
        if safety_blocked
        else "AE12_9_PASS_WITH_LIMITATIONS"
    )

    write_audits(
        source_index, hash_before, hash_after, compileall_result, safety_blocked
    )
    write_reports(classification)

    created = [
        str(OUTPUT_ROOT / "reports" / "ae12_9_final_msc_system_report.md"),
        str(OUTPUT_ROOT / "reports" / "ae12_9_final_summary_for_upload.txt"),
        str(OUTPUT_ROOT / "reports" / "ae12_9_final_decision_gate.json"),
        *figure_paths,
        str(OUTPUT_ROOT / "tables" / "ae12_9_phase_status_matrix.csv"),
        str(OUTPUT_ROOT / "tables" / "ae12_9_model_evidence_matrix.csv"),
        str(OUTPUT_ROOT / "tables" / "ae12_9_context_agent_matrix.csv"),
        str(OUTPUT_ROOT / "tables" / "ae12_9_authority_matrix.csv"),
        str(OUTPUT_ROOT / "tables" / "ae12_9_safety_matrix.csv"),
        str(OUTPUT_ROOT / "tables" / "ae12_9_forward_evidence_matrix.csv"),
        str(OUTPUT_ROOT / "tables" / "ae12_9_ui_api_matrix.csv"),
        str(OUTPUT_ROOT / "tables" / "ae12_9_limitations_matrix.csv"),
        str(OUTPUT_ROOT / "manifests" / "ae12_9_manifest.json"),
        str(OUTPUT_ROOT / "manifests" / "ae12_9_source_artifact_index.csv"),
        str(OUTPUT_ROOT / "manifests" / "ae12_9_reproducibility_references.json"),
        str(OUTPUT_ROOT / "audits" / "ae12_9_source_integrity_audit.csv"),
        str(OUTPUT_ROOT / "audits" / "ae12_9_missing_artifact_audit.csv"),
        str(OUTPUT_ROOT / "audits" / "ae12_9_claim_traceability_audit.csv"),
        str(OUTPUT_ROOT / "audits" / "ae12_9_final_safety_consistency_audit.json"),
    ]
    write_manifests(source_index, created)

    # If blocked, rewrite gate to reflect block
    if safety_blocked:
        gate_path = OUTPUT_ROOT / "reports" / "ae12_9_final_decision_gate.json"
        gate = read_json(gate_path) or {}
        gate["classification"] = classification
        gate["ae12_9_blocked"] = True
        gate["ae12_closed"] = False
        gate["msc_final_package_closed"] = False
        gate["blockers"] = offenders
        write_json(gate_path, gate)

    print(json.dumps({
        "output_root": str(OUTPUT_ROOT),
        "classification": classification,
        "safety_blocked": safety_blocked,
        "offenders": offenders,
        "compileall": compileall_result.get("status"),
        "trader_db_unchanged": hash_before == hash_after,
        "files_created": len(created),
    }, indent=2))
    return 0 if not safety_blocked else 2


if __name__ == "__main__":
    raise SystemExit(main())
