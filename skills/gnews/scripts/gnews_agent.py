#!/usr/bin/env python3
"""Zero-dep launcher for the /gnews Claude Code skill.

Tries to invoke the installed ``gnews-agent`` CLI. If the package isn't
installed, prints a clear install message and exits non-zero so the calling
agent surfaces the gap to the user instead of silently producing empty
output.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import textwrap


_INSTALL_HELP = textwrap.dedent("""
    gnews-agent is not installed.

    Install with:

        pip install gnews-agent

    For LLM-powered commands (`brief`, `sentiment`) also export an API key:

        export OPENAI_API_KEY=sk-...    # or ANTHROPIC_API_KEY / GROQ_API_KEY

    Once installed, this skill will work transparently — re-run the same
    command and it will be forwarded to the `gnews-agent` CLI.
""").strip()


def main(argv: list[str]) -> int:
    cli = shutil.which("gnews-agent")
    if cli is None:
        sys.stderr.write(_INSTALL_HELP + "\n")
        return 127
    # Forward every argument as-is. CLI emits JSON to stdout already.
    return subprocess.call([cli, *argv[1:]])


if __name__ == "__main__":
    sys.exit(main(sys.argv))
