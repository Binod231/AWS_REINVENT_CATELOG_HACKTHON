# Connecting MCP clients

Two distinct MCP servers are involved, and it is worth being clear about which
does what:

| Server | Who runs it | What it gives you |
|---|---|---|
| `awsevents` | AWS, remotely, at `https://api.awsevents.com/mcp` | The 12 raw catalog and schedule operations |
| `replan` | You, locally | Higher-order tools: solve an agenda, explain a decision, compute travel time, report coverage |

You can use either alone. Together, an assistant can read the catalog through
AWS's server and reason about it through ours.

## The AWS remote MCP server

Every tool maps one-to-one onto a REST operation: `ListEvents`, `GetEvent`,
`ListSessions`, `GetSession`, `GetSchedule`, `ReserveSessions`,
`CancelReservation`, `AssociateFavorites`, `DisassociateFavorite`,
`CreatePersonalTime`, `UpdatePersonalTime`, `DeletePersonalTime`.

Unlike the REST API, **every** MCP call requires sign-in, including catalog
reads that need no credentials over REST. Reading a registration-only catalog, or
touching a schedule, additionally requires being registered for that event.

### Callback ports: the part that bites

The API matches a redirect URI in full — scheme, host, port, and path — with no
wildcards, and a callback it does not recognise fails *without* telling you the
callback was the cause. Two separate sets of registered values exist:

- **Applications you build** (including `replan login`) must use loopback ports
  `8484`–`8489` with the path `/callback`. Anything else fails.
- **Installed MCP clients** have their own registered callbacks. Kiro uses
  `http://127.0.0.1:8976`; Claude Code uses `8484`.

Always set the port explicitly. Both clients otherwise pick a random one, and a
random port cannot sign in.

### Kiro

Copy the template into place and reload the window:

```bash
mkdir -p .kiro/settings
cp examples/mcp/kiro-mcp.json .kiro/settings/mcp.json
```

The template is [`examples/mcp/kiro-mcp.json`](../examples/mcp/kiro-mcp.json) and
configures both servers. The AWS half is:

```json
{
  "mcpServers": {
    "awsevents": {
      "url": "https://api.awsevents.com/mcp",
      "oauth": {
        "clientId": "7vmom55m1qstvq8i71ph127bfq",
        "redirectUri": "http://127.0.0.1:8976",
        "oauthScopes": ["openid", "email", "events/access"]
      }
    }
  }
}
```

End `redirectUri` at the port, with no trailing path. Kiro reads the port from
the last colon onward and appends `/oauth/callback` itself; give it a path and it
cannot parse the port, falls back to a random one, and sign-in fails with
`redirect_mismatch`. Reload the window after editing.

Confirm it works by asking the assistant to list events.

### Claude Code

```bash
claude mcp add --transport http --scope user awsevents \
  https://api.awsevents.com/mcp \
  --callback-port 8484 --client-id 7vmom55m1qstvq8i71ph127bfq
```

### Signing out properly

An MCP client has no redirect chain to sign you out through, and deleting a
client's configuration does **not** invalidate a token it already holds. To sign
out fully:

1. Revoke the token — `replan logout` does this against the revocation endpoint.
2. End the Builder ID session at <https://profile.aws.amazon.com>.

## What to expect from an assistant

Two failure modes are worth knowing about, because an assistant will otherwise
report success that did not happen:

**Partial success is normal.** Reserving and favoriting succeed *per session*,
not per request. A batch of ten can return HTTP 200 with four refusals inside it.
Ask the assistant to confirm from `GetSchedule` rather than trusting "done".
`replan`'s own tools do this verification automatically and report the delta.

**A listed tool can still be closed.** Per-operation quotas apply exactly as on
the REST API, and an operation can be unavailable while its tool is still
advertised. Reservations return a 409-equivalent refusal for every caller until
reserved seating opens through the API on 8 October 2026; browsing and favoriting
keep working throughout.

## Using replan against the MCP surface directly

The CLI can route every operation through AWS's MCP server instead of REST:

```bash
replan login                      # OAuth 2.0 + PKCE, ports 8484-8489
replan doctor  --surface mcp      # lists the tools the server advertises
replan catalog stats --surface mcp --event reinvent2026
```

Both surfaces are driven through one `CatalogPort` protocol and share the same
decoders, so they return identical domain objects. A single contract test suite
runs against both, which is what keeps them from drifting apart.
