# shellcheck shell=bash
# Azure CLI wrapper. Sourced, not executed.
#
# Two entry points, and the distinction is the whole point:
#
#   azq  <args...>   read-only query; always executes
#   azrun <args...>  MUTATING; under DRY_RUN=1 prints the fully-expanded command and succeeds
#
# None of the `az` commands in this repository have ever run against a real subscription, and
# they are the most expensive code here to get wrong: a malformed flag surfaces only once GPUs
# are billing. A dry run turns that into a laptop-time failure.
#
# DRY_RUN is fail-safe in the same shape reap.sh already used: any value other than the
# literal "0" means dry. Callers that mutate by default (launch.sh, teardown.sh) leave it
# unset; reap.sh keeps its safer default of on.

az_dry() { [[ -n "${DRY_RUN:-}" && "${DRY_RUN}" != "0" ]]; }

# Print a command the way a human would paste it back, quoting only what needs it.
_az_fmt() {
  local out="" a
  for a in "$@"; do
    if [[ "${a}" =~ [[:space:]\"\'\$\`\\] || -z "${a}" ]]; then
      out+=" '${a//\'/\'\\\'\'}'"
    else
      out+=" ${a}"
    fi
  done
  printf '%s' "${out# }"
}

azrun() {
  if az_dry; then
    printf '[dry-run] %s\n' "$(_az_fmt az "$@")"
    return 0
  fi
  az "$@"
}

azq() { az "$@"; }

# Announce the mode once, so a dry run is never mistaken for a real one in a log.
az_banner() {
  if az_dry; then
    echo "[dry-run] DRY_RUN=${DRY_RUN} -- no Azure resource will be created, changed or deleted."
  fi
}
