# Code and Run Instructions

## 1. Repository

The source-code submission package is available at:

https://github.com/ezrabest/MemeCoin-AI-Thesis-Code

This repository contains the implementation code used for the MSc thesis:

A Multi-Layer AI Framework for Meme-Coin Market Monitoring, Risk Explanation, and Paper-Trading Validation

Author: Dr. Ezra Ella
Supervisor: Dr. Nir Andelman
Advisor: Dr. Sharon Yalov-Handzel

## 2. Scope

The repository is provided for thesis review and reproducibility inspection.

All experiments reported in the thesis were conducted in demo / paper-trading mode only.

The system was not deployed for live trading, did not connect to a real wallet, and did not execute real transactions.

## 3. Excluded materials

The following materials are intentionally excluded from the public GitHub repository:

- local databases, including trader.db
- raw provider payloads
- cached API/RPC responses
- generated audit packages
- large CSV, JSONL, parquet, ZIP, and cache files
- API keys
- .env files
- wallet keys, seed phrases, or private credentials
- local logs and temporary files
- full academic submission package files such as proposal, interim report, final thesis draft, and final defense presentation

These exclusions are intentional for safety, privacy, size, and reproducibility-control reasons.

## 4. Recommended local setup

Clone the repository:

    git clone https://github.com/ezrabest/MemeCoin-AI-Thesis-Code.git
    cd MemeCoin-AI-Thesis-Code

Create a Python virtual environment:

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip

Install dependencies if a requirements file exists:

    pip install -r requirements.txt

Or, if the repository contains a pyproject file:

    pip install -e .

## 5. Environment configuration

Do not commit real API keys.

If needed, copy the example environment file:

    Copy-Item .env.example .env

Recommended safe defaults:

    APP_MODE=DEMO
    LIVE_TRADING_ENABLED=false
    WALLET_CONNECTION_ENABLED=false

## 6. Running the application or scripts

Because the thesis evidence was generated from local data and timestamped audit packages, exact end-to-end reruns may require the original local data environment.

The public GitHub repository is intended primarily for:

- source-code review
- architecture inspection
- implementation reproducibility
- demonstration of system structure
- review of safety boundaries

External API-dependent behavior may differ over time because providers may change schemas, rate limits, availability, or historical coverage.

## 7. Safety boundary

Do not enable live trading from this code without a separate production safety review.

Do not connect a real wallet.

Do not place real orders.

Do not add API keys or secrets to Git.

Do not interpret paper/demo outputs as live profitability proof.

## 8. Thesis evidence

Validated thesis evidence packages are documented at a high level in:

- DATA_AND_EVIDENCE_LINKS_PUBLIC.md
- EVIDENCE_PACKAGE_INDEX_PUBLIC.csv

The full local evidence packages are preserved separately for supervisor/examiner review and are not pushed to the public GitHub repository.
