#!/usr/bin/env bash
# Downloads the public unsupervised split of ISP-AD (Zenodo record 14911043) and lays it out as
#   $ARGUS_DATA_DIR/isp_ad/{LSM_1,LSM_2,ASM}/{train/good,test/{good,area,points},ground_truth/{area,points}}
# Only unsupervised.zip (~426 MB) is fetched; the record's supervised/additional archives
# (0.55-5.5 GB each) are not used. LSM_1/train/good_reduced (byte-identical copies of 500
# train/good images) and the archive's configs/ folder are not extracted.
#
# Usage: scripts/download_isp_ad.sh [--data-dir DIR] [--archive PATH] [--config PATH]
#   --data-dir DIR   default: $ARGUS_DATA_DIR (env or .env), else data/ in the repo
#   --archive PATH   use an unsupervised.zip already on disk (e.g. copied from Google Drive)
#   --config PATH    default: configs/datasets/isp_ad.yaml
# Rerunning skips modalities that are already present and verified.
set -euo pipefail

# shellcheck source=scripts/_lib.sh
. "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"

RECORD_ID=14911043
RECORD_VERSION=1.0
ZIP_NAME=unsupervised.zip
ZIP_URL="https://zenodo.org/api/records/$RECORD_ID/files/$ZIP_NAME/content"
API_URL="https://zenodo.org/api/records/$RECORD_ID"
PINNED_MD5=edfb54cb26629b53963fa08bec9e4a6b
DATASET=isp_ad

usage() {
  sed -n '2,13p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

data_dir_arg="" archive="" CONFIG="$ROOT/configs/datasets/isp_ad.yaml"
while [ $# -gt 0 ]; do
  case "$1" in
    --data-dir) [ $# -ge 2 ] || die "--data-dir needs a value"; data_dir_arg="$2"; shift 2 ;;
    --archive) [ $# -ge 2 ] || die "--archive needs a value"; archive="$2"; shift 2 ;;
    --config) [ $# -ge 2 ] || die "--config needs a value"; CONFIG="$2"; shift 2 ;;
    -h | --help) usage; exit 0 ;;
    *) usage >&2; die "unknown argument: $1" ;;
  esac
done

need_tool unzip "Install it with 'apt-get install unzip' (Linux) or 'brew install unzip'."
setup_python
resolve_data_dir "$data_dir_arg"
[ -f "$CONFIG" ] || die "config not found: $CONFIG"
CONFIG=$(cd "$(dirname "$CONFIG")" && pwd)/$(basename "$CONFIG")
if [ -n "$archive" ]; then
  [ -f "$archive" ] || die "archive not found: $archive"
  archive=$(cd "$(dirname "$archive")" && pwd)/$(basename "$archive")
fi

EXPECTED_MD5="$PINNED_MD5"
if [ -n "${ISP_AD_MD5:-}" ]; then
  EXPECTED_MD5="$ISP_AD_MD5"
  info "note: ISP_AD_MD5 is set, so the pinned MD5 is overridden. This exists for tests only."
fi

DEST="$DATA_DIR/$DATASET"
MODALITIES=$("$PYTHON" -m argus.data.verify --dataset "$DATASET" --config "$CONFIG" --list-categories)
# shellcheck disable=SC2086  # names are checked against [A-Za-z0-9_.-]+ by argus.data.verify
check_existing "$DATASET" "$DEST" $MODALITIES
if [ -z "$MISSING" ]; then
  exit 0
fi

make_tmp_dir "$DATASET"

# 1-2. Find the archive: --archive, else the cached download, else Zenodo.
if [ -z "$archive" ]; then
  need_tool curl "Install curl, or download $ZIP_URL yourself and pass --archive PATH."
  curl -sSL --fail --retry 3 -o "$TMP_DIR/record.json" "$API_URL" ||
    die "cannot read the Zenodo record $API_URL"
  info_line=$("$PYTHON" -m argus.data.archive zenodo-check "$TMP_DIR/record.json" "$ZIP_NAME" "$PINNED_MD5")
  tab=$(printf '\t')
  expected_size=${info_line%%"$tab"*}
  RECORD_VERSION=${info_line#*"$tab"}
  archive="$DATA_DIR/_downloads/$DATASET/$ZIP_NAME"
  size_now=""
  if [ -f "$archive" ]; then
    size_now=$("$PYTHON" -c 'import os, sys; print(os.path.getsize(sys.argv[1]))' "$archive")
  fi
  if [ "$size_now" = "$expected_size" ]; then
    info "using cached $archive"
  else
    download "$ZIP_URL" "$archive"
  fi
fi

# 3. Integrity: MD5 against the pinned value.
info "checking MD5 of $archive"
md5=$("$PYTHON" -m argus.data.archive hash md5 "$archive")
if [ "$md5" != "$EXPECTED_MD5" ]; then
  die "MD5 mismatch for $archive: got $md5, expected $EXPECTED_MD5. Nothing was extracted.
  If this is a partial or corrupt download, delete the file and rerun."
fi
info "MD5 ok ($md5)"

# 4. Extract only the missing modalities into the temp dir, verify each, then move it into place.
unzip -Z1 "$archive" >"$TMP_DIR/members.txt" || die "cannot list $archive (not a zip file?)"
RAW="$TMP_DIR/raw"
STAGE="$TMP_DIR/stage"
mkdir -p "$RAW" "$STAGE/$DATASET"
for m in $MISSING; do
  grep -q "^unsupervised/$m/" "$TMP_DIR/members.txt" || die "modality $m is not in $archive"
done
for m in $MISSING; do
  info "extracting $m"
  # Never extract train/good_reduced (duplicates of train/good). configs/ is outside the
  # pattern anyway. Exclusions are passed only when they match, so unzip stays quiet.
  if grep -q "^unsupervised/$m/train/good_reduced/" "$TMP_DIR/members.txt"; then
    info "  skipping $m/train/good_reduced (byte-identical copies of train/good images)"
    unzip -qq "$archive" "unsupervised/$m/*" -x "unsupervised/$m/train/good_reduced/*" -d "$RAW" ||
      die "unzip failed for $m"
  else
    unzip -qq "$archive" "unsupervised/$m/*" -d "$RAW" || die "unzip failed for $m"
  fi
  chmod -R u+rwX "$RAW/unsupervised/$m"  # in case the zip stores read-only entries
  mv "$RAW/unsupervised/$m" "$STAGE/$DATASET/$m"
  "$PYTHON" -m argus.data.verify --dataset "$DATASET" --data-dir "$STAGE" --config "$CONFIG" \
    --category "$m" --quiet >&2 || die "$m from $archive does not match $CONFIG (see above). Nothing was moved into $DEST."
done
for f in license.txt changelog.txt; do
  if grep -qx "unsupervised/$f" "$TMP_DIR/members.txt"; then
    unzip -qq -o "$archive" "unsupervised/$f" -d "$RAW"
  fi
done

mkdir -p "$DEST"
extracted=""
for m in $MISSING; do
  mv "$STAGE/$DATASET/$m" "$DEST/$m"
  extracted="${extracted:+$extracted,}$m"
done
for f in license.txt changelog.txt; do
  if [ -f "$RAW/unsupervised/$f" ]; then mv -f "$RAW/unsupervised/$f" "$DEST/$f"; fi
done

# 5. Verify everything and record the source.
"$PYTHON" -m argus.data.verify --dataset "$DATASET" --data-dir "$DATA_DIR" --config "$CONFIG" \
  --json "$DEST/.argus_source.json" \
  --archive "$archive" md5 "$md5" "$extracted" \
  --meta "zenodo_record=$RECORD_ID" --meta "version=$RECORD_VERSION" --meta "url=$ZIP_URL" \
  --meta "license=CC BY-NC-SA 4.0" ||
  die "verification failed after extraction (see above)"
info "wrote $DEST/.argus_source.json"
