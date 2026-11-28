# shellcheck shell=bash
# Shared helpers for scripts/download_*.sh. Sourced, not run.
# Portable to macOS bash 3.2: no associative arrays, mapfile or ${x,,}; no GNU-only flags.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_DIR=""

die() {
  echo "error: $*" >&2
  exit 1
}

info() {
  echo "$*"
}

cleanup() {
  # Only ever removes the temp directory this script created.
  # Archives may store read-only folders (MVTec AD does), so make them writable first.
  if [ -n "$TMP_DIR" ] && [ -d "$TMP_DIR" ]; then
    chmod -R u+w "$TMP_DIR" 2>/dev/null || true
    rm -rf "$TMP_DIR"
  fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM

# Absolute path of $1, which must be an existing directory.
abs_dir() {
  (cd "$1" && pwd)
}

# Picks Python like the Makefile: $PYTHON, else .venv/bin/python, else python3.
# Fails early if argus is not importable.
setup_python() {
  if [ -z "${PYTHON:-}" ]; then
    if [ -x "$ROOT/.venv/bin/python" ]; then
      PYTHON="$ROOT/.venv/bin/python"
    else
      PYTHON=python3
    fi
  fi
  command -v "$PYTHON" >/dev/null 2>&1 || die "Python not found ($PYTHON). Set PYTHON=/path/to/python."
  "$PYTHON" -c 'import argus.data.verify, argus.data.archive' 2>/dev/null ||
    die "cannot import argus with $PYTHON. Run 'make install' in $ROOT first (or set PYTHON=...)."
}

# Value of ARGUS_DATA_DIR from the environment, else from $ROOT/.env, else "data".
# A relative value is taken relative to the repo root, like the other ARGUS_* paths.
env_data_dir() {
  local val="${ARGUS_DATA_DIR:-}" line
  if [ -z "$val" ] && [ -f "$ROOT/.env" ]; then
    line=$(grep -E '^[[:space:]]*ARGUS_DATA_DIR[[:space:]]*=' "$ROOT/.env" | tail -n 1 || true)
    val=${line#*=}
    val=${val%%[[:space:]]#*}
    val=$(printf '%s' "$val" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')
    val=${val#\"}
    val=${val%\"}
    val=${val#\'}
    val=${val%\'}
  fi
  [ -n "$val" ] || val=data
  case "$val" in
    /*) printf '%s\n' "$val" ;;
    *) printf '%s\n' "$ROOT/$val" ;;
  esac
}

# Resolves DATA_DIR from --data-dir (relative to the current directory) or env_data_dir.
resolve_data_dir() {
  local arg="$1" dir
  if [ -n "$arg" ]; then dir="$arg"; else dir=$(env_data_dir); fi
  mkdir -p "$dir" || die "cannot create data directory $dir"
  DATA_DIR=$(abs_dir "$dir")
}

need_tool() {
  command -v "$1" >/dev/null 2>&1 || die "'$1' is required but not installed. $2"
}

# tar must read .tar.xz: bsdtar (macOS) has xz built in; GNU tar calls the xz program.
need_tar_xz() {
  need_tool tar "Install tar (e.g. 'apt-get install tar')."
  if ! tar --version 2>/dev/null | grep -qi bsdtar; then
    need_tool xz "GNU tar needs xz for .tar.xz archives: 'apt-get install xz-utils' or 'brew install xz'."
  fi
}

# Creates the script's temp directory inside the data dir (same filesystem, so the final
# move into place is a rename).
make_tmp_dir() {
  TMP_DIR=$(mktemp -d "$DATA_DIR/.argus-tmp-$1.XXXXXX") || die "cannot create a temp directory in $DATA_DIR"
}

# Downloads URL $1 to file $2 with resume support (partial data kept in $2.part).
download() {
  local url="$1" dest="$2"
  need_tool curl "Install curl, or download the file yourself and pass its path."
  mkdir -p "$(dirname "$dest")"
  info "downloading $url"
  info "  -> $dest (resumes if interrupted)"
  curl -L --fail --retry 3 -C - -o "$dest.part" "$url" ||
    die "download failed. Rerun the same command to resume; partial data is in $dest.part"
  mv -f "$dest.part" "$dest"
}

# check_existing DATASET ROOT_DIR NAME...: for each NAME already under ROOT_DIR, prints
# "skip" if it verifies and stops (deleting nothing) if it does not. Sets MISSING to the
# space-separated names that are absent.
check_existing() {
  local dataset="$1" root_dir="$2" name out
  shift 2
  MISSING=""
  for name in "$@"; do
    if [ -e "$root_dir/$name" ]; then
      if out=$("$PYTHON" -m argus.data.verify --dataset "$dataset" --data-dir "$DATA_DIR" \
        --config "$CONFIG" --category "$name" --quiet 2>&1); then
        info "skip $name: already present and verified"
      else
        printf '%s\n' "$out" >&2
        die "$root_dir/$name exists but does not verify (see above). Nothing was changed.
  Remove it yourself (rm -rf \"$root_dir/$name\") and rerun to extract it again."
      fi
    else
      MISSING="$MISSING $name"
    fi
  done
}
