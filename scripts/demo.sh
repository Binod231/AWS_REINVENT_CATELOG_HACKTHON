#!/usr/bin/env bash
# Live AWS Events API Demo.
# Runs against the real, live AWS Events server (https://api.awsevents.com/v1).
#
# Uses public live sessions from the AWS Events API so anyone can run it immediately
# without pre-requisite credentials, verifying the live REST and MCP connectivity.
set -euo pipefail

OUT="${OUT:-out}"
RUN="${RUN:-uv run}"
EVENT="${REPLAN_EVENT_ID:-Summit-Dubai-2026}"

step() {
	printf '\n\033[1;36m==> %s\033[0m\n' "$1"
}

mkdir -p "$OUT"

step "1. replan version"
$RUN replan --version

step "2. Live AWS Events API reachability check"
$RUN replan doctor

step "3. Sync session catalog from live AWS Events API ($EVENT)"
$RUN replan sync --event "$EVENT"

step "4. Offline full-text search across cached live sessions (FTS5)"
$RUN replan search "AI" --event "$EVENT" --limit 5

step "5. Physical travel graph computation across conference venues"
$RUN replan travel "Venetian | Level 2 | Hall C" "Mandalay Bay | South Pacific F"

step "6. Generate attendee preference profile"
$RUN replan profile init --output "$OUT/profile.yaml" --force

step "7. Rank live sessions against preference profile"
$RUN replan rank --event "$EVENT" --profile "$OUT/profile.yaml" --top 6

step "8. Solve optimal conflict-free, transit-feasible agenda using OR-Tools CP-SAT"
$RUN replan optimize --event "$EVENT" --profile "$OUT/profile.yaml" --out "$OUT/plan.json"

step "9. Explainability engine: inspect top rejections and trade-offs"
$RUN replan why --plan "$OUT/plan.json" --worst

step "10. Export calendar (.ics) and printable HTML day sheets"
$RUN replan export --plan "$OUT/plan.json" --ics "$OUT/replan.ics" --html-dir "$OUT/daysheets"

step "11. Crew mode: multi-attendee coordinated scheduling with topic coverage matrix"
$RUN replan crew --event "$EVENT" --profiles examples/crew --out "$OUT/crew.json"

step "12. Live seat watcher: queue session, inspect watch list, poll live server"
$RUN replan watch add AIM101 --event "$EVENT"
$RUN replan watch list --event "$EVENT"
$RUN replan watch run --event "$EVENT" --once
$RUN replan watch remove AIM101 --event "$EVENT"

printf '\n\033[1;32m✓ Live AWS demo complete.\033[0m Artifacts in ./%s\n' "$OUT"
printf 'Calendar: %s/replan.ics    Day sheets: %s/daysheets/\n' "$OUT" "$OUT"

