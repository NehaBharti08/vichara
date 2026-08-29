"""Deploy the static trajectory viewer to a Hugging Face Space.

    uv run python scripts/deploy_space.py --repo-id nehabharti0802/vichara

A *static* Space, not a Docker one: Hugging Face now requires a PRO
subscription to host Docker or Gradio Spaces on free hardware. That constraint
turned out to suit this demo. The viewer was always about *displaying* a
trajectory rather than producing one, and a static page loads instantly, never
sleeps, and cannot show a cold-start error or an exhausted quota -- the three
ways a hosted agent demo usually embarrasses its author.

Only `site/` is uploaded, plus the Space README. No source, no credentials, no
container. Regenerate the payload first with scripts/export_static.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi

# Anything not needed to build or run the image. The first three lines are the
# ones that matter: a live key, whatever the file tool wrote, and trajectory
# logs would otherwise be uploaded to a public Space.
# Only the built site is uploaded. An allowlist by construction -- the folder
# contains exactly index.html and data.json -- rather than a denylist that has
# to anticipate every future secret.
SITE = "site"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="e.g. username/vichara")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    space_readme = root / "spaces" / "README.md"
    if not space_readme.exists():
        print(f"missing {space_readme}", file=sys.stderr)
        return 1

    api = HfApi()
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="space",
        space_sdk="static",
        private=args.private,
        exist_ok=True,
    )
    print(f"space ready: https://huggingface.co/spaces/{args.repo_id}", file=sys.stderr)

    site = root / SITE
    if not (site / "data.json").exists():
        print("run scripts/export_static.py first", file=sys.stderr)
        return 1

    api.upload_folder(
        folder_path=str(site),
        repo_id=args.repo_id,
        repo_type="space",
        commit_message="Deploy the static trajectory viewer",
    )
    # Last, so it wins over the repository README uploaded above.
    api.upload_file(
        path_or_fileobj=str(space_readme),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="space",
        commit_message="Space README with frontmatter",
    )
    print("uploaded. build starts automatically.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
