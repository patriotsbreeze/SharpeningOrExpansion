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
# Backend is chosen by SOE_OBJSTORE: azure (default) | s3 | file | none.
#
# The `file` backend treats SOE_FILE_STORE as the remote. It exists so the resume path --
# the riskiest part of the whole system, and the part a unit test cannot otherwise reach --
# can be exercised end to end without a cloud account. It mirrors the azure backend's
# semantics exactly, including the two-pass ordering.
#
# On Azure we use `azcopy copy`, never `azcopy sync` and never `az storage blob sync`.
# `az storage blob sync` hardcodes --delete-destination=true and does not mention it in
# --help, so a worker pushing its own partial subtree would DELETE every other worker's
# shards. `azcopy sync` has the same default. `copy` cannot prune, which is exactly the
# property we want when N machines write into one container concurrently.
#
# Note the flag asymmetry that makes this easy to get wrong: `azcopy copy` defaults
# --recursive to FALSE (it would silently upload only top-level files and exit 0), while
# `azcopy sync` defaults it to TRUE. Always pass --recursive explicitly.

set -o pipefail

objstore_backend() { echo "${SOE_OBJSTORE:-azure}"; }

objstore_configured() {
  case "$(objstore_backend)" in
    azure) [[ -n "${AZ_CONTAINER_URL:-}" ]] ;;
    s3)    [[ -n "${S3_URI:-}" ]] ;;
    file)  [[ -n "${SOE_FILE_STORE:-}" ]] ;;
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
      file)  echo "[objstore] SOE_FILE_STORE is unset" >&2 ;;
      none)  return 0 ;;
      *)     echo "[objstore] unknown SOE_OBJSTORE='$(objstore_backend)'" >&2 ;;
    esac
    return 1
  fi
}

# Push the artifact tree in two passes: everything except markers, then a SNAPSHOT of the
# markers that already existed when the first pass began.
#
# The marker-last invariant is the entire basis of resume correctness, and it has to survive
# the network hop, not just the local disk. A single recursive upload transfers files in
# whatever order the tool picks, so a ".done.json" can land in blob storage before the shard
# it vouches for. If the VM is then evicted -- and with --eviction-policy Delete its disk is
# gone -- the next worker sees that marker, concludes the chunk is finished, and skips it
# forever: recorded complete, data nowhere.
#
# Two passes are NOT sufficient on their own, which is subtle enough to be worth stating.
# These pushes run every SYNC_INTERVAL while `soe generate` is still writing, so the tree is
# not quiescent. If pass 2 re-enumerated the markers, a chunk completed after pass 1 walked
# its directory would have its marker uploaded with its shard left behind -- exactly the
# failure the two passes are meant to prevent. So the marker list is snapshotted BEFORE pass
# 1 runs, and pass 2 uploads only that list. Every marker sent is then provably one whose
# shard was on disk before pass 1 enumerated.
objstore_push() {
  local local_dir="$1" remote_sub="$2"
  objstore_configured || { echo "[objstore] not configured; skipping push" >&2; return 0; }

  local parent base snapshot
  parent="$(cd "$(dirname "${local_dir}")" && pwd)"
  base="$(basename "${local_dir}")"
  snapshot="$(mktemp)"
  # Paths relative to ${parent}, which is what --list-of-files expects.
  ( cd "${parent}" && find "${base}" -type f -name '*.done.json' ) > "${snapshot}" || {
    rm -f "${snapshot}"; return 1; }

  local rc=0
  case "$(objstore_backend)" in
    azure)
      # copy, not sync: copy has no prune semantics at all, so it cannot remove another
      # worker's shards. overwrite=ifSourceNewer makes repeated pushes cheap.
      #
      # azcopy places the SOURCE DIRECTORY as a child of the destination, so the destination
      # is the container root and the "${remote_sub}" component comes from the local dir's
      # own basename. Passing the full remote path here would nest it twice.
      azcopy copy "${local_dir}" "$(_az_url "")" \
        --recursive --exclude-pattern "*.done.json" \
        --overwrite=ifSourceNewer --output-level=essential || rc=1
      if [[ ${rc} -eq 0 && -s "${snapshot}" ]]; then
        ( cd "${parent}" && azcopy copy "." "$(_az_url "")" \
            --recursive --list-of-files "${snapshot}" \
            --overwrite=ifSourceNewer --output-level=essential ) || rc=1
      fi
      ;;
    s3)
      aws s3 sync "${local_dir}" "${S3_URI}/${remote_sub}" \
        --exclude "*.done.json" --only-show-errors || rc=1
      if [[ ${rc} -eq 0 ]]; then
        while IFS= read -r rel; do
          [[ -n "${rel}" ]] || continue
          aws s3 cp "${parent}/${rel}" "${S3_URI}/${rel#"${base}"/}" --only-show-errors \
            || { rc=1; break; }
        done < "${snapshot}"
      fi
      ;;
    file)
      mkdir -p "${SOE_FILE_STORE}"
      _fs_copy_all "${parent}" "${base}" "${SOE_FILE_STORE}" '!' -name '*.done.json' || rc=1
      if [[ ${rc} -eq 0 ]]; then
        while IFS= read -r rel; do
          [[ -n "${rel}" ]] || continue
          mkdir -p "${SOE_FILE_STORE}/$(dirname "${rel}")"
          cp -f "${parent}/${rel}" "${SOE_FILE_STORE}/${rel}" || { rc=1; break; }
        done < "${snapshot}"
      fi
      ;;
    none) ;;
  esac
  rm -f "${snapshot}"
  return "${rc}"
}

# find -exec swallows the exit status of every command it runs, so a copy loop built on it
# reports success even when every single file failed. This propagates per-file failure.
_fs_copy_all() {
  local parent="$1" base="$2" dest="$3"; shift 3
  local f rc=0
  while IFS= read -r -d '' f; do
    mkdir -p "${dest}/$(dirname "${f}")" || { rc=1; break; }
    cp -f "${parent}/${f}" "${dest}/${f}" || { rc=1; break; }
  done < <(cd "${parent}" && find "${base}" -type f "$@" -print0)
  return "${rc}"
}

objstore_pull_markers() {
  local remote_sub="$1" local_dir="$2"
  objstore_configured || { echo "[objstore] not configured; nothing to resume from" >&2; return 0; }
  mkdir -p "${local_dir}"
  case "$(objstore_backend)" in
    azure)
      azcopy copy "$(_az_url "${remote_sub}" "/*")" "${local_dir}" \
        --recursive --include-pattern "*.done.json" --overwrite=true --output-level=essential
      ;;
    s3)
      aws s3 sync "${S3_URI}/${remote_sub}" "${local_dir}" \
        --exclude "*" --include "*.done.json" --only-show-errors
      ;;
    file)
      local base parent
      base="$(basename "${local_dir}")"
      parent="$(cd "$(dirname "${local_dir}")" && pwd)"
      [[ -d "${SOE_FILE_STORE}/${base}" ]] || return 0   # nothing published yet is not an error
      _fs_copy_all "${SOE_FILE_STORE}" "${base}" "${parent}" -name '*.done.json' || return 1
      ;;
    none) return 0 ;;
  esac
}

# Publish a single file WRITE-ONCE, first writer wins.
#
# Used for the problem manifest, where overwriting is actively dangerous: problem_idx is a
# position in that file and feeds every seed, so a second publisher replacing it mid-run
# repartitions the experiment underneath the workers already generating against the first
# version. Write-once also makes leadership takeover safe -- if two workers both decide to
# prepare, exactly one wins and everyone then consumes the winner's bytes.
#
# Returns 0 if we wrote it, 0 if it already existed (the caller re-pulls either way), 1 on a
# real failure.
objstore_publish_once() {
  local local_file="$1" remote_sub="$2"
  objstore_configured || return 0
  case "$(objstore_backend)" in
    azure) azcopy copy "${local_file}" "$(_az_url "${remote_sub}")" \
             --overwrite=false --output-level=essential ;;
    s3)    if aws s3api head-object --bucket "${S3_URI#s3://}" --key "${remote_sub}" \
                >/dev/null 2>&1; then return 0; fi
           aws s3 cp "${local_file}" "${S3_URI}/${remote_sub}" --only-show-errors ;;
    file)  mkdir -p "${SOE_FILE_STORE}/$(dirname "${remote_sub}")" || return 1
           if [[ -e "${SOE_FILE_STORE}/${remote_sub}" ]]; then return 0; fi
           cp -n "${local_file}" "${SOE_FILE_STORE}/${remote_sub}" ;;
    none)  return 0 ;;
  esac
}

# Pull a subtree wholesale. Used only for the manifest directory, which is kilobytes.
objstore_pull_dir() {
  local remote_sub="$1" local_dir="$2"
  objstore_configured || return 0
  mkdir -p "${local_dir}"
  case "$(objstore_backend)" in
    azure) azcopy copy "$(_az_url "${remote_sub}" "/*")" "${local_dir}" \
             --recursive --overwrite=true --output-level=essential 2>/dev/null || return 1 ;;
    s3)    aws s3 sync "${S3_URI}/${remote_sub}" "${local_dir}" --only-show-errors || return 1 ;;
    file)  [[ -d "${SOE_FILE_STORE}/${remote_sub}" ]] || return 1
           cp -rf "${SOE_FILE_STORE}/${remote_sub}/." "${local_dir}/" || return 1 ;;
    none)  return 1 ;;
  esac
}

# Build a blob URL.
#
# AZ_CONTAINER_URL may carry a SAS query string, so every path component -- including a
# trailing "/*" wildcard -- has to be spliced in BEFORE the "?". Appending after it silently
# folds the path into the signature's query and the request fails to authenticate, or worse,
# targets the wrong prefix.
#
#   _az_url "exp=s1"        -> https://acct.blob.../soe/exp=s1?sv=...
#   _az_url "exp=s1" "/*"   -> https://acct.blob.../soe/exp=s1/*?sv=...
#   _az_url ""              -> https://acct.blob.../soe?sv=...
_az_url() {
  local sub="${1:-}" suffix="${2:-}" base="${AZ_CONTAINER_URL}" path query
  if [[ "${base}" == *"?"* ]]; then
    path="${base%%\?*}"
    query="?${base#*\?}"
  else
    path="${base}"
    query=""
  fi
  path="${path%/}"
  [[ -n "${sub}" ]] && path="${path}/${sub}"
  echo "${path}${suffix}${query}"
}
