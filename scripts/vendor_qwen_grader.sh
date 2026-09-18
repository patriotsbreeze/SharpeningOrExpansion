#!/usr/bin/env bash
# Vendor Qwen's grader. Run on a machine with network access.
#
# Worth the effort: SimpleRL, Oat-Zero and PRIME all report numbers with this grader, so
# running it is what makes our numbers comparable to theirs. At k=256 a 1% difference in
# extraction recall moves pass@256 visibly, and the whole sharpening-vs-expansion claim lives
# in that tail -- so grader choice is a robustness axis, not an implementation detail.
set -Eeuo pipefail

DEST="$(dirname "$0")/../src/soe/grade/qwen_vendor"
TMP=$(mktemp -d)
REPO="https://github.com/QwenLM/Qwen2.5-Math"

git clone --depth 1 "${REPO}" "${TMP}/qwen"
SHA=$(git -C "${TMP}/qwen" rev-parse HEAD)

mkdir -p "${DEST}"
cp "${TMP}/qwen/evaluation/grader.py"         "${DEST}/grader.py"
cp "${TMP}/qwen/evaluation/math_normalize.py" "${DEST}/math_normalize.py" 2>/dev/null || true
cp "${TMP}/qwen/LICENSE"                      "${DEST}/LICENSE" 2>/dev/null || true
touch "${DEST}/__init__.py"

cat > "${DEST}/PROVENANCE.md" <<PROV
# Vendored from QwenLM/Qwen2.5-Math

- Upstream: ${REPO}
- Commit:   ${SHA}
- Vendored: $(date -u +%Y-%m-%d)
- Files:    evaluation/grader.py, evaluation/math_normalize.py

## Why vendored rather than pinned as a dependency

It is not published to PyPI, and the numbers reported by SimpleRL, Oat-Zero and PRIME were
produced with it. Copying it here is what makes our results comparable to theirs.

## Local modifications

None yet. **Record every edit here as a diff.** Silent divergence from upstream would
invalidate the comparability argument that is the only reason this file exists.
PROV

rm -rf "${TMP}"
echo "vendored @ ${SHA} -> ${DEST}"
