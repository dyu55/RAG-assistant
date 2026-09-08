#!/usr/bin/env bash
# ==============================================================================
# Local CI Pre-flight Gate Runner for RAG-Assistant
# Runs the full CI/CD verification suite locally before committing & pushing.
# ==============================================================================

set -eo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}🚀 Starting RAG-Assistant Local CI Verification Pipeline${NC}"
echo -e "${BLUE}======================================================================${NC}"

FAILED=0

# ── 1. Secret Hygiene & Placeholder Scan ──────────────────────────────────────
echo -e "\n${YELLOW}[Step 1/4] 🔒 Scanning for sensitive secrets and credential hygiene...${NC}"
python3 -c "
import sys, re, os

patterns = [
    re.compile(r'sk-[a-zA-Z0-9]{20,}'),
    re.compile(r'gsk_[a-zA-Z0-9]{20,}'),
    re.compile(r'(?i)password\s*=\s*[\'\"][^\'\"]{4,}[\'\"]')
]

scan_dirs = ['src', 'tests', 'docs']
found = False

for sdir in scan_dirs:
    if not os.path.exists(sdir):
        continue
    for root, _, files in os.walk(sdir):
        for f in files:
            if not f.endswith(('.py', '.md', '.txt', '.json', '.yaml', '.yml', '.env.example', '.html', '.js')):
                continue
            path = os.path.join(root, f)
            with open(path, 'r', encoding='utf-8', errors='ignore') as fp:
                for idx, line in enumerate(fp, 1):
                    for pat in patterns:
                        if pat.search(line):
                            print(f'❌ Potential secret at {path}:{idx}: {line.strip()[:60]}')
                            found = True

if found:
    sys.exit(1)
else:
    print('✅ No sensitive credentials detected.')
"
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ PASSED: Secret hygiene verification passed.${NC}"
else
    echo -e "${RED}❌ FAILED: Potential live API key pattern detected!${NC}"
    FAILED=1
fi

# ── 2. Ruff Lint & Format Check ──────────────────────────────────────────────
echo -e "\n${YELLOW}[Step 2/4] 🧹 Running Ruff code quality and format checks...${NC}"
if uv run ruff check . && uv run ruff format --check .; then
    echo -e "${GREEN}✅ PASSED: Code style and linting conform to PEP 8 standards.${NC}"
else
    echo -e "${RED}❌ FAILED: Ruff lint or format check failed. Run 'uv run ruff format .' to fix.${NC}"
    FAILED=1
fi

# ── 3. Bandit Security Audit ──────────────────────────────────────────────────
echo -e "\n${YELLOW}[Step 3/4] 🛡️ Running Bandit static security analysis...${NC}"
if command -v bandit &> /dev/null; then
    if bandit -r src/ -ll -q; then
        echo -e "${GREEN}✅ PASSED: No security vulnerabilities found by Bandit.${NC}"
    else
        echo -e "${RED}❌ FAILED: Bandit security scan reported issues.${NC}"
        FAILED=1
    fi
else
    echo -e "${YELLOW}⚠️ Notice: bandit not found in local PATH (handled in GitHub Actions CI container).${NC}"
fi

# ── 4. Full Pytest Regression & Coverage ──────────────────────────────────────
echo -e "\n${YELLOW}[Step 4/4] 🧪 Running Pytest unit & integration test suites...${NC}"
if uv run pytest --cov=rag_assistant --cov-report=term-missing --cov-fail-under=85 tests/; then
    echo -e "${GREEN}✅ PASSED: All unit and integration test suites passed!${NC}"
else
    echo -e "${RED}❌ FAILED: One or more pytest test cases failed or coverage fell below 85%!${NC}"
    FAILED=1
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo -e "\n${BLUE}======================================================================${NC}"
if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}🎉 ALL CI GATES PASSED! Code is safe and ready for commit & push.${NC}"
    echo -e "${BLUE}======================================================================${NC}"
    exit 0
else
    echo -e "${RED}💥 CI GATE FAILED! Please fix the errors above before pushing.${NC}"
    echo -e "${BLUE}======================================================================${NC}"
    exit 1
fi
