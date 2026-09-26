#!/usr/bin/env bash
# The credential-free demo. Everything here runs against the bundled synthetic
# catalog, so a judge can clone the repo and see the whole product work without
# a re:Invent account, an API key, or an AWS account.
#
# CI runs this script verbatim (see .github/workflows/ci.yml) which is what
# keeps the README honest.
set -euo pipefail

OUT="${OUT:-out}"
RUN="${RUN:-uv run}"

step() {
	printf '\n\033[1;36m==> %s\033[0m\n' "$1"
}

mkdir -p "$OUT"

step "replan version"
$RUN replan --version

step "environment check"
$RUN replan doctor --source fixture

step "catalog statistics (synthetic re:Invent 2026 catalog)"
$RUN replan catalog stats --source fixture

step "build the local search index"
$RUN replan sync --source fixture

step "full-text search, offline"
$RUN replan search "bedrock agents" --level 300 --limit 8

step "travel time between two properties"
$RUN replan travel "Venetian | Level 2 | Hall C" "Mandalay Bay | South Pacific F"

step "generate a starter preference profile"
$RUN replan profile init --output "$OUT/profile.yaml" --force

step "rank the catalog against that profile"
$RUN replan rank --profile "$OUT/profile.yaml" --top 12

step "solve for a travel-feasible five-day agenda"
$RUN replan optimize --profile "$OUT/profile.yaml" --out "$OUT/plan.json"

step "explain one inclusion and one rejection"
$RUN replan why --plan "$OUT/plan.json" --worst

step "export calendar and printable day sheets"
$RUN replan export --plan "$OUT/plan.json" --ics "$OUT/replan-2026.ics" --html-dir "$OUT/daysheets"

step "preview the write-back to the attendee account (dry run: no writes)"
$RUN replan apply --plan "$OUT/plan.json" --source fixture

step "seat watcher, dry run"
$RUN replan watch --plan "$OUT/plan.json" --source fixture --dry-run --once

step "crew coverage across three attendees"
$RUN replan crew --profiles examples/crew --out "$OUT/crew.json"

printf '\n\033[1;32mDemo complete.\033[0m Artefacts in ./%s\n' "$OUT"
printf 'Calendar: %s/replan-2026.ics    Day sheets: %s/daysheets/\n' "$OUT" "$OUT"
