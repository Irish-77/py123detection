#!/usr/bin/env bash
# Map a git branch name to the directory it is published under on the gh-pages branch.
#
#   main                   -> ""             (the site root)
#   feat/mmcv_integration  -> feat-mmcv-integration
#
# The default branch is decided by the caller; this script only sanitises the name, because a
# branch name may contain characters that are legal in git but not in a URL path segment.
set -euo pipefail

branch="${1:?usage: branch-slug.sh <branch-name>}"

slug="$(printf '%s' "$branch" \
  | tr '[:upper:]' '[:lower:]' \
  | sed -e 's#[^a-z0-9._-]#-#g' -e 's#-\{2,\}#-#g' -e 's#^[-.]*##' -e 's#[-.]*$##')"

# Names made up entirely of separators (or ".."), would otherwise escape the site directory.
case "$slug" in
  "" | "." | "..") slug="preview" ;;
esac

printf '%s\n' "$slug"
