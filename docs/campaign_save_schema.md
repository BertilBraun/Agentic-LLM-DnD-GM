# Campaign Save File Schema (Markdown)

This document specifies the structure of `.dnd-save.md` files used to persist
campaign progress. The schema is human-readable, diff-friendly, and append-only
(to minimise merge conflicts in version control).

---

## File Naming

```
<campaign_name>_<timestamp>.dnd-save.md
```

* `campaign_name` – slug (kebab-case) identifying the campaign.
* `timestamp` – UTC ISO8601 (e.g. `2024-03-16T14-05-22Z`).

Example: `lost-mine_2024-03-16T14-05-22Z.dnd-save.md`

---

## Top-level Sections

1. `# Metadata` – basic info (version, campaign name, creation time).
2. `# World State` – canonical facts about the campaign world (locations, NPCs, items).
3. `# Story Plan` – ordered list of narrative beats (GM prep notes).
4. `# Scene History` – chronological log of completed scenes.
5. `# Open Threads` – unresolved quests / hooks.

Separators (`---`) delineate sections for easy parsing.

---

### 1. Metadata (YAML)

```yaml
# Metadata
---
version: 1
campaign: "Lost Mine of Phandelver"
created: "2024-03-16T14:05:22Z"
last_played: "2024-03-20T21:48:10Z"
---
```

* `version` – schema version (integer).
* `campaign` – human-readable name.
* `created` / `last_played` – UTC ISO8601.

---

### 2. World State (Markdown List)

```
# World State
---
## Locations
- Phandalin: small frontier town rebuilt on ancient ruins.
- Wave Echo Cave: long-lost dwarven mine.

## NPCs
- Sildar Hallwinter – wounded Lord's Alliance agent, ally.
- Glasstaff (Iarno) – traitorous wizard leading Redbrands.

## Items
- Forge of Spells – dormant magical forge.
---
```

Sub-headers (`##`) group facts; bullet points are atomic entries.

---

### 3. Story Plan

Ordered list representing planned beats:

```
# Story Plan
---
1. Goblin ambush on Triboar Trail
2. Rescue Sildar from Cragmaw Hideout
3. Investigate Redbrand activity in Phandalin
---
```

---

### 4. Scene History

Each scene is an embedded collapsible details block containing summary,
transcript link, and date.

```
# Scene History
---
<details>
<summary>2024-03-20 – "Goblin Ambush"</summary>

**Summary**: Party ambushed by goblins; captured clues to Cragmaw Hideout.

**Transcript**: [[scene_logs/goblin_ambush.md]]
</details>
---
```

---

### 5. Open Threads

Bullet list of unresolved hooks–updated by MasterAgent:

```
# Open Threads
---
- Find Cragmaw Castle to rescue Gundren.
- Discover purpose of the Black Spider.
---
```

---

## Parsing Notes

* Sections are detected by level-1 `#` headings followed by `---`.
* The YAML block under Metadata is parsed with `yaml.safe_load`.
* Scene History may contain multiple `<details>` blocks.
* All other lists are treated as free-form text.

---

## Future Extensions

* **Version bump** when adding new sections.
* Attach image references for scene art.
* Encrypt spoilers for players.

</rewritten_file> 