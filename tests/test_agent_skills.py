"""The agent skills in .cursor/skills and .claude/skills stay identical."""

import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parents[1]
CURSOR, CLAUDE = REPO / ".cursor/skills", REPO / ".claude/skills"


def _files(root):
  return {p.relative_to(root): p.read_text() for p in root.rglob("*") if p.is_file()}


def test_copies_are_identical():
  cursor, claude = _files(CURSOR), _files(CLAUDE)
  assert cursor.keys() == claude.keys()
  for path in cursor:
    assert cursor[path] == claude[path], path


def test_frontmatter():
  for skill in CLAUDE.glob("*/SKILL.md"):
    text = skill.read_text()
    m = re.match(r"---\nname: ([a-z0-9-]{1,64})\ndescription: >-\n((?:  .*\n)+)---\n", text)
    assert m, skill
    assert m.group(1) == skill.parent.name
    assert len(" ".join(m.group(2).split())) <= 1024
    assert len(text.splitlines()) < 500
