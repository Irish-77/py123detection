#!/usr/bin/env bash
# Publish (or remove) one documentation build inside the gh-pages branch.
#
#   publish-docs.sh ""     docs/_build   # default branch -> site root
#   publish-docs.sh my-br  docs/_build   # other branch   -> /my-br/
#   publish-docs.sh my-br                # no build dir   -> delete /my-br/
#
# gh-pages holds every branch at once, so a deploy may only touch its own subtree:
#   /            the default branch's build
#   /<slug>/     one directory per other branch, named by branch-slug.sh
#
# A root deploy therefore keeps the preview directories of branches that still exist on the
# remote, and drops the rest -- which is also how previews of deleted branches get collected if
# the delete event was ever missed.
#
# Environment: GH_TOKEN, GITHUB_REPOSITORY, DEFAULT_BRANCH, and GITHUB_REF_NAME/GITHUB_SHA for
# the commit message.
set -euo pipefail

slug="${1?usage: publish-docs.sh <slug> [build-dir]}"
build_dir="${2:-}"

: "${DEFAULT_BRANCH:?}"

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# GH_PAGES_REMOTE overrides the remote, so the deploy can be rehearsed against a local clone.
remote="${GH_PAGES_REMOTE:-https://x-access-token:${GH_TOKEN:?}@github.com/${GITHUB_REPOSITORY:?}.git}"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

if [ -n "$build_dir" ]; then
  build_dir="$(cd "$build_dir" && pwd)"   # absolute: the script cds into the checkout below
fi

sync_tree() {
  if [ -z "$build_dir" ]; then
    # Cleanup mode: drop one preview directory, leave everything else alone.
    [ -n "$slug" ] || { echo "refusing to delete the site root" >&2; exit 1; }
    rm -rf -- "${slug:?}"
    return
  fi

  if [ -n "$slug" ]; then
    rm -rf -- "${slug:?}"
    mkdir -p -- "$slug"
    cp -a "$build_dir/." "$slug/"
    return
  fi

  # Root deploy: wipe the root, but keep git's own files and every live branch's preview.
  local keep=".git .nojekyll CNAME" branch
  while read -r branch; do
    [ -n "$branch" ] || continue
    [ "$branch" = "$DEFAULT_BRANCH" ] && continue
    keep="$keep $("$here/branch-slug.sh" "$branch")"
  done < <(git ls-remote --heads origin | sed 's#.*refs/heads/##')

  local entry
  while IFS= read -r entry; do
    case " $keep " in *" $entry "*) continue ;; esac
    rm -rf -- "$entry"
  done < <(find . -mindepth 1 -maxdepth 1 -printf '%f\n')

  cp -a "$build_dir/." .
}

attempt() {
  rm -rf "$work/site"
  mkdir -p "$work/site"
  cd "$work/site"

  git init -q -b gh-pages
  git remote add origin "$remote"
  if git fetch -q --depth 1 origin gh-pages 2>/dev/null; then
    git reset -q --hard origin/gh-pages
  fi

  sync_tree
  touch .nojekyll   # Sphinx writes _static/ and _sources/, which Jekyll would otherwise drop.

  git add -A
  if git diff --cached --quiet; then
    echo "gh-pages is already up to date."
    return 0
  fi
  git -c user.name='github-actions[bot]' \
      -c user.email='41898282+github-actions[bot]@users.noreply.github.com' \
      commit -q -m "docs: ${GITHUB_REF_NAME:-manual} @ ${GITHUB_SHA:0:7}${slug:+ (/$slug/)}"
  git push -q origin gh-pages
}

# The workflow's concurrency group serialises deploys, so a rejected push means someone pushed to
# gh-pages by hand. Rebuilding the tree from the new remote state is safer than rebasing.
for n in 1 2 3; do
  if attempt; then exit 0; fi
  echo "push failed (attempt $n), retrying against the current gh-pages" >&2
  sleep $((n * 5))
done
exit 1
