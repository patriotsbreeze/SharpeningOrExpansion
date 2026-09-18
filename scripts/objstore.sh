# shellcheck shell=bash
# Object-store abstraction. Sourced, not executed.
#
# The artifact tree has the same layout locally and remotely, so both operations are plain
# recursive copies with no path rewriting to get wrong. Two operations are all the harness
# needs:
#
#   objstore_push          -- upload the artifact tree (shards + markers)
#   objstore_pull_markers  -- download ONLY "*.done.json" (kilobytes), never the shards
#
# The asymmetry is the point. Resume only needs to know WHICH chunks are finished, and the
# markers encode exactly that. Pulling the shards too would mean re-downloading hundreds of
# gigabytes every time a spot VM comes back.
#
# Backend is chosen by SOE_OBJSTORE: azure (default) | s3 | none.

set -o pipefail

objstore_backend() { echo "${SOE_OBJSTORE:-azure}"; }

objstore_configured() {
  case "$(objstore_backend)" in
    azure) [[ -n "${AZ_CONTAINER_URL:-}" ]] ;;
    s3)    [[ -n "${S3_URI:-}" ]] ;;
    *)     return 1 ;;
  esac
}

# Fail fast on a misconfiguration rather than silently running with no durability. A run that
# looks fine and syncs nowhere is discovered only after the VM is reclaimed.
objstore_require() {
  if ! objstore_configured; then
    case "$(objstore_backend)" in
      azure) echo "[objstore] AZ_CONTAINER_URL is unset (expected a container SAS or https URL)" >&2 ;;
      s3)    echo "[objstore] S3_URI is unset" >&2 ;;
      none)  return 0 ;;
      *)     echo "[objstore] unknown SOE_OBJSTORE='$(objstore_backend)'" >&2 ;;
    esac
    return 1
  fi
}

objstore_push() {
  local local_dir="$1" remote_sub="$2"
  objstore_configured || { echo "[objstore] not configured; skipping push" >&2; return 0; }
  case "$(objstore_backend)" in
    azure)
      # --delete-destination=false is explicit and load-bearing: workers push overlapping
      # subtrees concurrently, and a sync that pruned "extra" destination files would delete
      # other workers' shards.
      azcopy sync "${local_dir}" "$(_az_url "${remote_sub}")" \
        --recursive --delete-destination=false --output-level=essential
      ;;
    s3)
      aws s3 sync "${local_dir}" "${S3_URI}/${remote_sub}" --only-show-errors
      ;;
    none) return 0 ;;
  esac
}

objstore_pull_markers() {
  local remote_sub="$1" local_dir="$2"
  objstore_configured || { echo "[objstore] not configured; nothing to resume from" >&2; return 0; }
  mkdir -p "${local_dir}"
  case "$(objstore_backend)" in
    azure)
      # copy, not sync: sync would want the whole tree present locally to compare against.
      azcopy copy "$(_az_url "${remote_sub}")/*" "${local_dir}" \
        --recursive --include-pattern "*.done.json" --overwrite=true --output-level=essential
      ;;
    s3)
      aws s3 sync "${S3_URI}/${remote_sub}" "${local_dir}" \
        --exclude "*" --include "*.done.json" --only-show-errors
      ;;
    none) return 0 ;;
  esac
}

# Push a single file (used to publish the shared problem manifest from the lead worker).
objstore_push_file() {
  local local_file="$1" remote_sub="$2"
  objstore_configured || return 0
  case "$(objstore_backend)" in
    azure) azcopy copy "${local_file}" "$(_az_url "${remote_sub}")" --overwrite=true --output-level=essential ;;
    s3)    aws s3 cp "${local_file}" "${S3_URI}/${remote_sub}" --only-show-errors ;;
    none)  return 0 ;;
  esac
}

# Pull a subtree wholesale. Used only for the manifest directory, which is kilobytes.
objstore_pull_dir() {
  local remote_sub="$1" local_dir="$2"
  objstore_configured || return 0
  mkdir -p "${local_dir}"
  case "$(objstore_backend)" in
    azure) azcopy copy "$(_az_url "${remote_sub}")/*" "${local_dir}" \
             --recursive --overwrite=true --output-level=essential 2>/dev/null || return 1 ;;
    s3)    aws s3 sync "${S3_URI}/${remote_sub}" "${local_dir}" --only-show-errors || return 1 ;;
    none)  return 1 ;;
  esac
}

# AZ_CONTAINER_URL may or may not carry a SAS query string; splice the path in before it.
_az_url() {
  local sub="$1" base="${AZ_CONTAINER_URL}"
  if [[ "${base}" == *"?"* ]]; then
    echo "${base%%\?*}/${sub}?${base#*\?}"
  else
    echo "${base}/${sub}"
  fi
}
