#!/usr/bin/env bash
# Extracts the MVTec AD categories listed in configs/datasets/mvtec_ad.yaml into
#   $ARGUS_DATA_DIR/mvtec_ad/{category}/{train/good,test/<type>,ground_truth/<type>}
# MVTec AD needs its licence accepted on mvtec.com, so this script does not fetch it by
# itself: pass the archive you downloaded (a path is used in place, never copied) or a URL.
#
# Usage: scripts/download_mvtec_ad.sh <archive-path-or-url> [more archives...]
#                                     [--data-dir DIR] [--config PATH]
#   archives        mvtec_anomaly_detection.tar.xz (all 15 categories, ~4.9 GB) or
#                   per-category archives such as hazelnut.tar.xz and screw.tar.xz
#   --data-dir DIR  default: $ARGUS_DATA_DIR (env or .env), else data/ in the repo
#   --config PATH   default: configs/datasets/mvtec_ad.yaml
# Rerunning skips categories that are already present and verified.
set -euo pipefail

# shellcheck source=scripts/_lib.sh
. "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"

DATASET=mvtec_ad
MVTEC_PAGE="https://www.mvtec.com/company/research/datasets/mvtec-ad"

usage() {
  sed -n '2,13p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

where_to_get_it() {
  cat >&2 <<EOF
usage: scripts/download_mvtec_ad.sh <archive-path-or-url> [more archives...] [--data-dir DIR] [--config PATH]

MVTec AD is not downloaded automatically: its licence has to be accepted first.
  1. Open $MVTEC_PAGE and use the download form there.
     Licence: CC BY-NC-SA 4.0 (non-commercial use; fine for this university project).
  2. Download mvtec_anomaly_detection.tar.xz (~4.9 GB, all categories), or only the
     per-category archives this project uses (see configs/datasets/mvtec_ad.yaml).
  3. Run:  make data-mvtec ARCHIVE=~/Downloads/mvtec_anomaly_detection.tar.xz
     or:   scripts/download_mvtec_ad.sh ~/Downloads/hazelnut.tar.xz ~/Downloads/screw.tar.xz
EOF
}

data_dir_arg="" CONFIG="$ROOT/configs/datasets/mvtec_ad.yaml"
ARCHIVES=()
while [ $# -gt 0 ]; do
  case "$1" in
    --data-dir) [ $# -ge 2 ] || die "--data-dir needs a value"; data_dir_arg="$2"; shift 2 ;;
    --config) [ $# -ge 2 ] || die "--config needs a value"; CONFIG="$2"; shift 2 ;;
    -h | --help) usage; exit 0 ;;
    -*) usage >&2; die "unknown option: $1" ;;
    *) ARCHIVES+=("$1"); shift ;;
  esac
done
if [ "${#ARCHIVES[@]}" -eq 0 ]; then
  where_to_get_it
  exit 2
fi

need_tar_xz
setup_python
resolve_data_dir "$data_dir_arg"
[ -f "$CONFIG" ] || die "config not found: $CONFIG"
CONFIG=$(cd "$(dirname "$CONFIG")" && pwd)/$(basename "$CONFIG")

DEST="$DATA_DIR/$DATASET"
CATEGORIES=$("$PYTHON" -m argus.data.verify --dataset "$DATASET" --config "$CONFIG" --list-categories)
# shellcheck disable=SC2086  # names are checked against [A-Za-z0-9_.-]+ by argus.data.verify
check_existing "$DATASET" "$DEST" $CATEGORIES
if [ -z "$MISSING" ]; then
  exit 0
fi

make_tmp_dir "$DATASET"
STAGE="$TMP_DIR/stage"
mkdir -p "$STAGE/$DATASET"
TODO="$MISSING"         # categories not yet found in any archive
ARCHIVE_ARGS=()         # --archive PATH sha256 DIGEST CATS, for the source record
license_file=""
tab=$(printf '\t')

i=0
for src in "${ARCHIVES[@]}"; do
  i=$((i + 1))
  if [ -z "$(echo "$TODO" | tr -d ' ')" ]; then
    info "all configured categories found; not reading $src"
    continue
  fi

  # 2. URL -> download into the cache (with resume); path -> use in place.
  case "$src" in
    http://* | https://*)
      name=${src%%\?*}
      name=${name##*/}
      [ -n "$name" ] || die "cannot tell the file name from $src"
      archive="$DATA_DIR/_downloads/$DATASET/$name"
      if [ -f "$archive" ]; then
        info "using cached $archive"
      else
        download "$src" "$archive"
      fi
      ;;
    *)
      [ -f "$src" ] || die "archive not found: $src"
      archive=$(cd "$(dirname "$src")" && pwd)/$(basename "$src")
      ;;
  esac

  info "SHA-256 of $archive (cached next to it after the first run)"
  sha=$("$PYTHON" -m argus.data.archive hash sha256 "$archive")
  info "  $sha"

  # Works whether or not the archive has a wrapper directory.
  info "listing $archive"
  tar -tJf "$archive" >"$TMP_DIR/list$i.txt" || die "cannot read $archive as .tar.xz"
  # shellcheck disable=SC2086
  "$PYTHON" -m argus.data.archive tar-plan "$TMP_DIR/list$i.txt" $TODO >"$TMP_DIR/plan$i.txt"

  members=()
  cats=""
  while IFS= read -r line; do
    key=${line%%"$tab"*}
    member=${line#*"$tab"}
    if [ "$key" = "@license" ]; then
      if [ -z "$license_file" ]; then members+=("$member"); fi
    else
      members+=("$member")
      cats="$cats $key"
    fi
  done <"$TMP_DIR/plan$i.txt"
  if [ -z "$cats" ]; then
    info "no configured category left to extract in $src"
    continue
  fi

  RAW="$TMP_DIR/raw$i"
  mkdir -p "$RAW"
  info "extracting$cats from $(basename "$archive")"
  tar -xJf "$archive" -C "$RAW" "${members[@]}" || die "tar failed on $archive"
  # MVTec AD stores folders as r-x and files as r--: without owner write permission the
  # folders cannot be moved into place (macOS) or cleaned up later.
  chmod -R u+rwX "$RAW"

  done_cats=""
  while IFS= read -r line; do
    key=${line%%"$tab"*}
    member=${line#*"$tab"}
    if [ "$key" = "@license" ]; then
      if [ -z "$license_file" ] && [ -f "$RAW/$member" ]; then license_file="$RAW/$member"; fi
      continue
    fi
    mv "$RAW/$member" "$STAGE/$DATASET/$key"
    "$PYTHON" -m argus.data.verify --dataset "$DATASET" --data-dir "$STAGE" --config "$CONFIG" \
      --category "$key" --quiet >&2 ||
      die "$key from $archive does not match $CONFIG (see above). Nothing was moved into $DEST."
    done_cats="${done_cats:+$done_cats,}$key"
    new_todo=""
    for c in $TODO; do [ "$c" = "$key" ] || new_todo="$new_todo $c"; done
    TODO="$new_todo"
  done <"$TMP_DIR/plan$i.txt"
  ARCHIVE_ARGS+=(--archive "$archive" sha256 "$sha" "$done_cats")
done

# 3. Move the verified categories into place.
mkdir -p "$DEST"
present=()
# shellcheck disable=SC2086
for c in $MISSING; do
  if [ -d "$STAGE/$DATASET/$c" ]; then
    mv "$STAGE/$DATASET/$c" "$DEST/$c"
  fi
done
if [ -n "$license_file" ]; then mv -f "$license_file" "$DEST/license.txt"; fi
for c in $CATEGORIES; do
  if [ -d "$DEST/$c" ]; then present+=(--category "$c"); fi
done

# 4. Verify, print counts and per-type breakdown, record the source.
if [ "${#present[@]}" -gt 0 ] && [ "${#ARCHIVE_ARGS[@]}" -gt 0 ]; then
  "$PYTHON" -m argus.data.verify --dataset "$DATASET" --data-dir "$DATA_DIR" --config "$CONFIG" \
    "${present[@]}" --json "$DEST/.argus_source.json" "${ARCHIVE_ARGS[@]}" \
    --meta "license=CC BY-NC-SA 4.0" --meta "source=$MVTEC_PAGE" ||
    die "verification failed after extraction (see above)"
  info "wrote $DEST/.argus_source.json"
fi
if [ -n "$(echo "$TODO" | tr -d ' ')" ]; then
  die "not found in the given archives:$TODO. Pass the archive(s) that contain them and rerun."
fi
