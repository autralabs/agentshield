# AgentShield Cloud Platform — Design Document

> HLD + LLD for `agentshield.cloud`
> Author: AI-assisted design session
> Date: 2026-02-16
> Status: DRAFT v2 — expanded with strategy, standards, and launch-readiness guidance

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Domain Architecture](#2-domain-architecture)
3. [System Context](#3-system-context)
4. [High-Level Design](#4-high-level-design)
5. [Low-Level Design](#5-low-level-design)
6. [Data Model](#6-data-model)
7. [API Specification](#7-api-specification)
8. [Authentication & Authorization](#8-authentication--authorization)
9. [Infrastructure & Deployment](#9-infrastructure--deployment)
10. [Cost Analysis](#10-cost-analysis)
11. [Migration Plan — Wiring the Existing UI](#11-migration-plan--wiring-the-existing-ui)
12. [Security Considerations](#12-security-considerations)
13. [Open Questions](#13-open-questions)
14. [Platform Strategy and Value Model](#14-platform-strategy-and-value-model)
15. [Standards and Schema Governance](#15-standards-and-schema-governance)
16. [Launch Readiness Gaps](#16-launch-readiness-gaps)
17. [SLOs, Operations, and Incident Model](#17-slos-operations-and-incident-model)
18. [Execution Roadmap (0-90 Days)](#18-execution-roadmap-0-90-days)
19. [Company-Style Best Practices Applied](#19-company-style-best-practices-applied)

---

## 1. Executive Summary

AgentShield Cloud is the hosted complement to the open-source `pyagentshield` SDK. It receives telemetry events from SDK instances, stores them, and serves them to a Next.js dashboard for analytics, history, scan playground, and settings management.

The intended product shape is a **security control plane for AI agents**, not just a metrics dashboard. The operational loop should be:
1. **Detect** suspicious prompts/tool calls
2. **Investigate** with context (session, model, method, environment, timeline)
3. **Fix** policy/config/model choices
4. **Verify** via trend changes, replay, and alert stability

**Design principles:**
- **Minimal cost at low traffic.** Use AWS free-tier and Supabase free-tier services wherever possible. Zero idle cost is the target.
- **Scalable when needed.** SQS + Lambda + Supabase can absorb traffic spikes without re-architecture.
- **SDK sends, cloud receives.** The SDK already has a telemetry client that POSTs gzipped JSON batches. The cloud just needs to ingest, store, and query.
- **UI is done.** The Next.js dashboard exists with mock data. This design wires it to real data.
- **Privacy-first by default.** Metadata-only ingestion unless a team explicitly opts into sensitive payload capture.
- **Open-core distribution.** Keep provider flexibility in SDK; monetize cloud operations, governance, team workflows, and managed intelligence.

---

## 2. Domain Architecture

```
agentshield.cloud          → Landing page (Next.js static export, CloudFront)
dashboard.agentshield.cloud → Dashboard app (Next.js, Vercel or CloudFront)
api.agentshield.cloud       → Backend API (API Gateway + Lambda)
```

### DNS Setup (Route 53)

| Record | Type | Target |
|--------|------|--------|
| `agentshield.cloud` | A/AAAA | CloudFront distribution (landing page) |
| `dashboard.agentshield.cloud` | CNAME | Vercel deployment (or separate CloudFront) |
| `api.agentshield.cloud` | A/AAAA | API Gateway custom domain |

---

## 3. System Context

```
┌──────────────┐         ┌──────────────────────┐         ┌──────────────────┐
│              │  POST   │                      │  READ   │                  │
│ pyagentshield├────────►│  api.agentshield.cloud├────────►│  Supabase        │
│ SDK (user's  │ gzipped │  (API Gateway +      │  SQL    │  (Postgres +     │
│ app)         │ JSON    │   Lambda)             │         │   Auth + RLS)    │
│              │ Bearer  │                      │         │                  │
└──────────────┘  token  └──────────┬───────────┘         └────────▲─────────┘
                                    │                              │
                                    │ SQS (async                   │
                                    │ ingest)                      │
                                    ▼                              │
                         ┌──────────────────────┐                  │
                         │  Ingest Lambda        │   INSERT         │
                         │  (batch processor)    ├─────────────────┘
                         └──────────────────────┘

┌──────────────────────┐         ┌──────────────────┐
│ dashboard.            │  REST  │                  │
│ agentshield.cloud     ├───────►│  api.agentshield │
│ (Next.js dashboard)   │ + JWT  │  .cloud          │
│                       │        │  (read endpoints)│
└───────────────────────┘        └──────────────────┘
```

**Data flow:**
1. SDK calls `POST api.agentshield.cloud/v1/telemetry` with `Bearer <api_key>`, gzipped JSON body
2. API Gateway validates key, forwards to Ingest Lambda
3. Ingest Lambda pushes to SQS (immediate 202 response to SDK)
4. Processor Lambda reads from SQS, batch-inserts into Supabase Postgres
5. Dashboard calls read APIs (`/v1/scans`, `/v1/analytics`, etc.) authenticated via Supabase JWT
6. Read APIs query Supabase Postgres directly

---

## 4. High-Level Design

### 4.1 Component Overview

| Component | Technology | Why |
|-----------|-----------|-----|
| **Landing page** | Next.js static export → S3 + CloudFront | Zero server cost, global CDN |
| **Dashboard** | Next.js on Vercel (free tier) | Already built, Vercel free for hobby |
| **API Gateway** | AWS API Gateway HTTP API | $1/million requests, free tier 1M/month for 12 months |
| **Ingest Lambda** | Python 3.12, <128MB | Validates + enqueues, <100ms |
| **SQS Queue** | Standard queue | Free forever (1M requests/month), decouples ingest from storage |
| **Processor Lambda** | Python 3.12, <256MB | Batch reads SQS, inserts Supabase |
| **Database** | Supabase Postgres (free tier) | 500MB storage, built-in auth, RLS, REST API |
| **Auth** | Supabase Auth | Email/password + OAuth, JWT tokens, free |
| **Secrets** | AWS SSM Parameter Store | Free for standard parameters |

### 4.2 Why This Stack

**SQS as the buffer:** The SDK sends batches of up to 50 events. At low traffic, SQS stays in the perpetual free tier (1M requests/month). It absorbs spikes so the ingest Lambda can return 202 immediately without waiting for Postgres. If Supabase is briefly slow or down, events queue up instead of being lost.

**Supabase over DynamoDB:** The dashboard needs relational queries — time-series aggregation, filtering by project/environment/cleaning_method, pagination with sort. Postgres handles this natively. Supabase adds auth, RLS (row-level security), and a REST API for free. DynamoDB would need GSIs and be awkward for analytics queries.

**Lambda over ECS/Fargate:** At low traffic, Lambda costs nothing (1M free invocations/month). Even at medium traffic (100K scans/day), the cost is pennies. ECS would have a baseline cost even when idle.

**Vercel for dashboard:** The Next.js app is already deployed there. Free tier covers hobby use. If costs become an issue later, can switch to CloudFront + S3 static export.

---

## 5. Low-Level Design

### 5.1 Ingest Lambda (`ingest_handler`)

**Trigger:** API Gateway HTTP API route `POST /v1/telemetry`

```python
# pseudocode
def handler(event, context):
    # 1. API Gateway already validated the API key via Lambda authorizer
    # 2. Decompress if Content-Encoding: gzip
    body = decompress_if_needed(event["body"])
    payload = json.loads(body)

    events = payload.get("events", [])
    if not events:
        return {"statusCode": 400, "body": '{"error":"no events"}'}

    # 3. Validate + enrich each event
    api_key_id = event["requestContext"]["authorizer"]["api_key_id"]
    project_id = event["requestContext"]["authorizer"]["project_id"]

    validated = []
    for e in events[:200]:  # cap at 200 per batch
        validated.append({
            "event_id": e.get("event_id", str(uuid4())),
            "project_id": project_id,
            "api_key_id": api_key_id,
            "timestamp": e.get("timestamp", datetime.utcnow().isoformat()),
            "is_suspicious": bool(e.get("is_suspicious", False)),
            "confidence": float(e.get("confidence", 0)),
            "drift_score": e.get("drift_score"),
            "threshold": e.get("threshold"),
            "embedding_model": e.get("embedding_model", ""),
            "cleaning_method": e.get("cleaning_method", ""),
            "on_detect": e.get("on_detect", ""),
            "sdk_version": e.get("sdk_version", ""),
            "session_id": e.get("session_id", ""),
            "environment": e.get("environment"),
        })

    # 4. Send to SQS (single SendMessage with batch as body)
    sqs.send_message(
        QueueUrl=QUEUE_URL,
        MessageBody=json.dumps({"events": validated}),
    )

    return {"statusCode": 202, "body": '{"accepted":' + str(len(validated)) + '}'}
```

**Cold start optimization:** Python 3.12, no heavy deps (just `boto3` bundled by Lambda runtime, `json`, `gzip`). Expected cold start <300ms, warm <50ms.

### 5.2 API Key Authorizer Lambda

**Trigger:** API Gateway Lambda authorizer (request-based, cached 300s)

```python
def handler(event, context):
    token = extract_bearer_token(event["headers"].get("authorization", ""))
    if not token:
        raise Exception("Unauthorized")

    # Lookup API key in Supabase (cached in Lambda memory across invocations)
    key_record = lookup_api_key(token)  # SELECT from api_keys WHERE key_hash = sha256(token)
    if not key_record or key_record["revoked"]:
        raise Exception("Unauthorized")

    return {
        "isAuthorized": True,
        "context": {
            "api_key_id": key_record["id"],
            "project_id": key_record["project_id"],
            "user_id": key_record["user_id"],
        }
    }
```

**Key format:** `ask_live_<32 hex chars>` (matching the existing UI mock). Stored as SHA-256 hash in the database. Prefix `ask_live_` for production, `ask_test_` for test environments.

### 5.3 Processor Lambda (`processor_handler`)

**Trigger:** SQS queue, batch size 10, max batching window 30s

```python
def handler(event, context):
    rows = []
    for record in event["Records"]:
        message = json.loads(record["body"])
        rows.extend(message["events"])

    if not rows:
        return

    # Batch insert into Supabase Postgres via HTTP API
    # Uses service_role key (server-side only, never exposed)
    supabase.table("scan_events").insert(rows).execute()
```

**Why batch insert:** SQS delivers up to 10 messages per invocation, each containing up to 200 events. Worst case: 2000 rows per invocation. Postgres handles this easily in a single `INSERT ... VALUES` statement.

### 5.4 Dashboard Read API

**Trigger:** API Gateway HTTP API, authenticated via Supabase JWT

The dashboard calls these endpoints. The API Gateway routes them to a single Read Lambda that queries Supabase Postgres.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/scans` | GET | Paginated scan history |
| `/v1/scans/stats` | GET | Aggregate stats (total scans, threats, detection rate, avg confidence) |
| `/v1/scans/timeline` | GET | Time-series data for detection timeline chart |
| `/v1/analytics/distribution` | GET | Safe vs threat distribution |
| `/v1/analytics/sources` | GET | Scans by source |
| `/v1/analytics/drift-trends` | GET | Drift score trends over time |
| `/v1/analytics/risk-factors` | GET | Top risk factors |
| `/v1/analytics/activity-heatmap` | GET | Scans by day-of-week and hour |
| `/v1/scan` | POST | Playground scan (proxies to a hosted AgentShield instance) |
| `/v1/settings` | GET/PUT | User detection/notification settings |
| `/v1/settings/api-keys` | GET/POST/DELETE | API key management |

**Alternative — Supabase direct:** Since Supabase exposes a PostgREST API with RLS, the dashboard could call Supabase directly for reads, skipping the Read Lambda entirely. This saves Lambda costs and latency. The trade-off is coupling the UI to the Supabase schema. Recommended for MVP; add a Lambda abstraction layer later if needed.

### 5.5 Playground Scan Endpoint

`POST /v1/scan` — This is different from telemetry ingestion. The user submits text from the dashboard playground, and the backend runs an actual AgentShield scan.

**Options (pick one for MVP):**

1. **Client-side only (current mock).** Keep the `simulateScan()` function in the UI. Zero backend cost. Honest about it being a demo.
2. **Lambda with embedded SDK.** A Lambda running `pyagentshield` with `all-MiniLM-L6-v2` (local provider). Needs ~512MB memory + EFS for the model. Cold start ~5s. Cost: ~$0.01/1000 scans.
3. **Dedicated lightweight container (Fargate Spot).** Keeps the model warm, ~$3/month. Better latency after first request.

**Recommendation:** Option 1 for MVP. Option 2 when demand warrants it.

---

## 6. Data Model

### 6.1 Supabase Postgres Schema

```sql
-- =============================================
-- Auth: Managed by Supabase Auth (auth.users)
-- =============================================

-- Projects (multi-tenancy unit)
CREATE TABLE projects (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name        TEXT NOT NULL,
    owner_id    UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_projects_owner ON projects(owner_id);

-- API Keys
CREATE TABLE api_keys (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id  UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    user_id     UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    key_prefix  TEXT NOT NULL,           -- "ask_live_7f3a" (first 12 chars, for display)
    key_hash    TEXT NOT NULL UNIQUE,     -- SHA-256 of full key
    name        TEXT NOT NULL DEFAULT 'Default',
    revoked     BOOLEAN NOT NULL DEFAULT FALSE,
    last_used   TIMESTAMPTZ,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_api_keys_hash ON api_keys(key_hash) WHERE NOT revoked;
CREATE INDEX idx_api_keys_project ON api_keys(project_id);

-- Scan Events (the core table — maps directly to SDK ScanEvent)
CREATE TABLE scan_events (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id       UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    api_key_id       UUID REFERENCES api_keys(id) ON DELETE SET NULL,

    -- From ScanEvent dataclass
    event_id         TEXT NOT NULL,          -- SDK-generated UUID
    timestamp        TIMESTAMPTZ NOT NULL,   -- SDK-generated ISO timestamp
    sdk_version      TEXT NOT NULL DEFAULT '',
    session_id       TEXT NOT NULL DEFAULT '',

    -- Scan metrics
    is_suspicious    BOOLEAN NOT NULL DEFAULT FALSE,
    confidence       FLOAT NOT NULL DEFAULT 0,
    drift_score      FLOAT,
    threshold        FLOAT,

    -- Config snapshot
    embedding_model  TEXT NOT NULL DEFAULT '',
    cleaning_method  TEXT NOT NULL DEFAULT '',
    on_detect        TEXT NOT NULL DEFAULT '',

    -- User context
    environment      TEXT,                   -- "production", "staging", etc.
    source           TEXT DEFAULT 'api',     -- "api", "playground", "webhook"

    -- Ingestion metadata
    ingested_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Primary query patterns:
CREATE INDEX idx_scan_events_project_ts ON scan_events(project_id, timestamp DESC);
CREATE INDEX idx_scan_events_suspicious ON scan_events(project_id, is_suspicious, timestamp DESC);
CREATE INDEX idx_scan_events_session ON scan_events(project_id, session_id);

-- Partition by month when table exceeds ~10M rows (future optimization)
-- For now, indexes are sufficient.

-- Unique constraint to prevent duplicate ingestion
CREATE UNIQUE INDEX idx_scan_events_dedup ON scan_events(project_id, event_id);

-- Project Settings (persists dashboard settings)
CREATE TABLE project_settings (
    project_id          UUID PRIMARY KEY REFERENCES projects(id) ON DELETE CASCADE,

    -- Detection settings
    drift_threshold     FLOAT DEFAULT 0.30,
    cleaning_method     TEXT DEFAULT 'hybrid',
    confidence_threshold FLOAT DEFAULT 0.70,
    on_detect           TEXT DEFAULT 'flag',      -- "block" | "flag"

    -- Notification settings
    email_alerts        BOOLEAN DEFAULT TRUE,
    slack_enabled       BOOLEAN DEFAULT FALSE,
    slack_webhook_url   TEXT,
    alert_threshold     INT DEFAULT 5,            -- threats per hour

    -- Webhook
    webhook_url         TEXT,

    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Row Level Security
ALTER TABLE projects ENABLE ROW LEVEL SECURITY;
ALTER TABLE api_keys ENABLE ROW LEVEL SECURITY;
ALTER TABLE scan_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE project_settings ENABLE ROW LEVEL SECURITY;

-- RLS Policies
CREATE POLICY "Users see own projects"
    ON projects FOR ALL
    USING (owner_id = auth.uid());

CREATE POLICY "Users see own API keys"
    ON api_keys FOR ALL
    USING (user_id = auth.uid());

CREATE POLICY "Users see own scan events"
    ON scan_events FOR ALL
    USING (project_id IN (SELECT id FROM projects WHERE owner_id = auth.uid()));

CREATE POLICY "Users see own settings"
    ON project_settings FOR ALL
    USING (project_id IN (SELECT id FROM projects WHERE owner_id = auth.uid()));

-- Service role bypasses RLS (used by ingest Lambda)
```

### 6.2 Mapping: SDK ScanEvent → Database

| SDK Field | DB Column | Notes |
|-----------|-----------|-------|
| `event_id` | `event_id` | Dedup key |
| `timestamp` | `timestamp` | Parsed from ISO string to TIMESTAMPTZ |
| `event_type` | — | Always "scan", not stored |
| `sdk_version` | `sdk_version` | |
| `session_id` | `session_id` | |
| `is_suspicious` | `is_suspicious` | |
| `confidence` | `confidence` | |
| `drift_score` | `drift_score` | |
| `threshold` | `threshold` | |
| `embedding_model` | `embedding_model` | |
| `cleaning_method` | `cleaning_method` | |
| `on_detect` | `on_detect` | |
| `project` | — | Resolved from API key → `project_id` |
| `environment` | `environment` | |
| — | `project_id` | Added by ingest Lambda from API key lookup |
| — | `api_key_id` | Added by ingest Lambda |
| — | `source` | "api" for SDK telemetry, "playground" for UI scans |
| — | `ingested_at` | Server-side timestamp |

---

## 7. API Specification

### 7.1 Telemetry Ingestion (SDK → Cloud)

```
POST /v1/telemetry
Host: api.agentshield.cloud
Authorization: Bearer ask_live_7f3a9b2c...
Content-Type: application/json
Content-Encoding: gzip

{
  "events": [
    {
      "event_id": "550e8400-e29b-41d4-a716-446655440000",
      "timestamp": "2026-02-15T10:30:00.000Z",
      "event_type": "scan",
      "sdk_version": "0.1.2",
      "session_id": "abc123",
      "is_suspicious": true,
      "confidence": 0.94,
      "drift_score": 0.0312,
      "threshold": 0.023,
      "embedding_model": "all-MiniLM-L6-v2",
      "cleaning_method": "heuristic",
      "on_detect": "flag",
      "project": null,
      "environment": "production"
    }
  ]
}

→ 202 Accepted
{"accepted": 1}
```

### 7.2 Dashboard Read APIs

All dashboard APIs use Supabase JWT (`Authorization: Bearer <supabase_jwt>`).

#### `GET /v1/scans`
Paginated scan history. Maps to the **RecentScansTable** component.

```
Query params:
  ?project_id=<uuid>
  &page=1
  &per_page=20
  &sort=timestamp:desc
  &filter[is_suspicious]=true
  &search=<text>  (searches session_id, embedding_model, cleaning_method)

Response:
{
  "data": [
    {
      "id": "uuid",
      "event_id": "uuid",
      "timestamp": "2026-02-15T10:30:00Z",
      "is_suspicious": true,
      "confidence": 0.94,
      "drift_score": 0.0312,
      "source": "api",
      "cleaning_method": "heuristic",
      "embedding_model": "all-MiniLM-L6-v2",
      "environment": "production",
      "session_id": "abc123"
    }
  ],
  "meta": {
    "total": 1247,
    "page": 1,
    "per_page": 20,
    "total_pages": 63
  }
}
```

#### `GET /v1/scans/stats`
Aggregate stats. Maps to the **StatsCards** component.

```
Query params:
  ?project_id=<uuid>
  &period=7d  (7d | 30d | 90d)

Response:
{
  "total_scans": 1247,
  "threats_detected": 23,
  "detection_rate": 0.018,
  "avg_confidence": 0.942,
  "period_comparison": {
    "total_scans_change": 0.12,
    "threats_change": 3,
    "detection_rate_change": -0.003,
    "avg_confidence_change": 0.014
  },
  "sparkline": [4, 7, 5, 9, 6, 8, 11, 9, 13, 10, 14, 12]
}
```

#### `GET /v1/scans/timeline`
Time-series for the **DetectionTimeline** chart.

```
Query params:
  ?project_id=<uuid>
  &period=7d

Response:
{
  "data": [
    {"date": "2026-02-09", "total_scans": 178, "threats": 3},
    {"date": "2026-02-10", "total_scans": 192, "threats": 5},
    ...
  ]
}
```

#### `GET /v1/analytics/distribution`
Maps to the **ScanDistributionChart** donut.

```
Response:
{
  "safe": 847,
  "threat": 153
}
```

#### `GET /v1/analytics/sources`
Maps to the **ScansBySourceChart** bar chart.

```
Response:
{
  "data": [
    {"source": "api", "count": 512},
    {"source": "playground", "count": 234},
    {"source": "webhook", "count": 178},
    {"source": "sdk", "count": 76}
  ]
}
```

#### `GET /v1/analytics/drift-trends`
Maps to the **DriftScoreTrendsChart** line chart.

```
Query params:
  ?project_id=<uuid>
  &days=30

Response:
{
  "data": [
    {"date": "2026-01-17", "avg_drift": 0.0123, "max_drift": 0.0456},
    ...
  ]
}
```

#### `GET /v1/analytics/risk-factors`
Maps to the **TopRiskFactorsChart**.

Note: The SDK `ScanEvent` doesn't currently include risk factors. This chart requires either:
- (a) Adding a `risk_factors` field to `ScanEvent` in a future SDK version, or
- (b) Keeping it as mock data until the pattern detector is implemented.

**Recommendation:** Option (b) for MVP, then ship real `risk_factors` immediately after dashboard MVP as the first schema extension paired with pattern detector output.

#### `GET /v1/analytics/activity-heatmap`
Maps to the **DailyActivityHeatmap**.

```
Response:
{
  "data": [
    {"day": 0, "hour": 0, "count": 5},
    {"day": 0, "hour": 1, "count": 2},
    ...
  ]
}
```

SQL: `SELECT EXTRACT(DOW FROM timestamp) as day, EXTRACT(HOUR FROM timestamp) as hour, COUNT(*) as count FROM scan_events WHERE project_id = $1 AND timestamp > now() - interval '7 days' GROUP BY 1, 2`

#### `POST /v1/scan` (Playground)
Runs a real scan via hosted AgentShield. **Deferred to post-MVP.**

#### `GET/PUT /v1/settings`
Maps to the **SettingsPageContent** component.

```
GET Response:
{
  "detection": {
    "drift_threshold": 0.30,
    "cleaning_method": "hybrid",
    "confidence_threshold": 0.70,
    "on_detect": "flag"
  },
  "notifications": {
    "email_alerts": true,
    "slack_enabled": false,
    "slack_webhook_url": null,
    "alert_threshold": 5
  },
  "webhook_url": null
}

PUT Body: (same shape, partial updates allowed)
```

#### `GET/POST/DELETE /v1/settings/api-keys`
API key management for the settings API tab.

```
GET Response:
{
  "data": [
    {
      "id": "uuid",
      "name": "Default",
      "key_prefix": "ask_live_7f3a",
      "created_at": "2026-02-01T00:00:00Z",
      "last_used": "2026-02-15T10:30:00Z"
    }
  ]
}

POST (create new key):
{"name": "Production"}
→ {"id": "uuid", "key": "ask_live_full_key_shown_once", "name": "Production"}

DELETE /v1/settings/api-keys/:id
→ 204 No Content
```

---

## 8. Authentication & Authorization

### 8.1 Two Auth Contexts

| Context | Mechanism | Purpose |
|---------|-----------|---------|
| **SDK → API** | API key (`Bearer ask_live_...`) | Telemetry ingestion. Stateless. Validated by Lambda authorizer. |
| **Dashboard → API** | Supabase JWT (`Bearer eyJ...`) | Dashboard reads/writes. User identity from JWT. RLS enforced. |

### 8.2 Auth Flow (Dashboard)

1. User visits `dashboard.agentshield.cloud`
2. Supabase Auth handles signup/login (email+password, or Google/GitHub OAuth)
3. On login, Supabase returns a JWT
4. Dashboard stores JWT in cookie/localStorage
5. All API calls include `Authorization: Bearer <jwt>`
6. Supabase RLS ensures users only see their own data

### 8.3 Auth Flow (SDK Telemetry)

1. User creates project in dashboard → gets an API key
2. User sets `AGENTSHIELD_TELEMETRY__API_KEY=ask_live_...` in their environment
3. SDK sends events with `Authorization: Bearer ask_live_...`
4. API Gateway Lambda authorizer validates the key hash against `api_keys` table
5. Authorizer response is cached for 300s to minimize DB lookups

### 8.4 Onboarding Flow

1. Sign up at `dashboard.agentshield.cloud`
2. Supabase creates `auth.users` record
3. Post-signup hook (Supabase Edge Function or DB trigger) creates:
   - Default `projects` record
   - Default `api_keys` record
   - Default `project_settings` record
4. Dashboard shows the API key with copy-to-clipboard (shown once)
5. User configures SDK: `AGENTSHIELD_TELEMETRY__API_KEY=ask_live_...`

---

## 9. Infrastructure & Deployment

### 9.1 AWS Resources (IaC: SAM or CDK)

```yaml
Resources:
  # API Gateway
  HttpApi:
    Type: AWS::ApiGatewayV2::Api
    Properties:
      Name: agentshield-api
      ProtocolType: HTTP
      CorsConfiguration:
        AllowOrigins: ["https://dashboard.agentshield.cloud"]
        AllowMethods: ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
        AllowHeaders: ["Authorization", "Content-Type", "Content-Encoding"]

  # Custom domain
  ApiDomainName:
    Type: AWS::ApiGatewayV2::DomainName
    Properties:
      DomainName: api.agentshield.cloud
      DomainNameConfigurations:
        - CertificateArn: !Ref AcmCertificate
          EndpointType: REGIONAL

  # SQS Queue
  TelemetryQueue:
    Type: AWS::SQS::Queue
    Properties:
      QueueName: agentshield-telemetry
      VisibilityTimeout: 60
      MessageRetentionPeriod: 345600  # 4 days
      RedrivePolicy:
        deadLetterTargetArn: !GetAtt TelemetryDLQ.Arn
        maxReceiveCount: 3

  TelemetryDLQ:
    Type: AWS::SQS::Queue
    Properties:
      QueueName: agentshield-telemetry-dlq
      MessageRetentionPeriod: 1209600  # 14 days

  # Ingest Lambda
  IngestFunction:
    Type: AWS::Serverless::Function
    Properties:
      Handler: ingest.handler
      Runtime: python3.12
      MemorySize: 128
      Timeout: 10
      Environment:
        Variables:
          QUEUE_URL: !Ref TelemetryQueue
          SUPABASE_URL: !Ref SupabaseUrl
          SUPABASE_SERVICE_KEY: !Ref SupabaseServiceKey  # SSM reference

  # Processor Lambda
  ProcessorFunction:
    Type: AWS::Serverless::Function
    Properties:
      Handler: processor.handler
      Runtime: python3.12
      MemorySize: 256
      Timeout: 30
      Events:
        SQSEvent:
          Type: SQS
          Properties:
            Queue: !GetAtt TelemetryQueue.Arn
            BatchSize: 10
            MaximumBatchingWindowInSeconds: 30

  # Read Lambda
  ReadFunction:
    Type: AWS::Serverless::Function
    Properties:
      Handler: read_api.handler
      Runtime: python3.12
      MemorySize: 256
      Timeout: 10

  # Landing page
  LandingBucket:
    Type: AWS::S3::Bucket
    Properties:
      BucketName: agentshield-landing

  LandingDistribution:
    Type: AWS::CloudFront::Distribution
    Properties:
      DistributionConfig:
        Aliases: [agentshield.cloud]
        Origins:
          - S3OriginConfig: ...
        DefaultCacheBehavior:
          ViewerProtocolPolicy: redirect-to-https
          CachePolicyId: 658327ea-f89d-4fab-a63d-7e88639e58f6  # CachingOptimized
```

### 9.2 Deployment Pipeline

```
GitHub Actions:
  on push to main:
    1. Build Lambda packages (zip)
    2. SAM deploy to AWS
    3. Export landing page (next export)
    4. Sync to S3 + CloudFront invalidation
    5. Vercel auto-deploys dashboard from agentshield-platform repo
```

### 9.3 Environment Layout

| Env | API Domain | Dashboard | DB |
|-----|-----------|-----------|-----|
| **prod** | `api.agentshield.cloud` | `dashboard.agentshield.cloud` | Supabase prod project |
| **staging** | `api-staging.agentshield.cloud` | Vercel preview URL | Supabase staging project (or same project, different schema) |

---

## 10. Cost Analysis

### 10.1 Free Tier Baseline (Month 1-12)

| Service | Free Tier | Our Usage (low traffic) | Cost |
|---------|-----------|------------------------|------|
| API Gateway HTTP API | 1M requests/month (12 months) | ~30K requests | $0 |
| Lambda | 1M invocations + 400K GB-s | ~60K invocations | $0 |
| SQS Standard | 1M requests/month (forever) | ~30K messages | $0 |
| CloudFront | 1TB transfer (12 months) | ~5GB | $0 |
| S3 | 5GB (12 months) | ~100MB | $0 |
| Route 53 | — | 1 hosted zone + queries | ~$0.50/month |
| ACM | Free | 1 cert | $0 |
| Supabase Free | 500MB DB, 50K MAU, 500MB storage | Fits easily | $0 |

**Estimated monthly cost at low traffic: ~$0.50/month** (just Route 53)

### 10.2 Medium Traffic (~100K scans/day)

| Service | Usage | Monthly Cost |
|---------|-------|-------------|
| API Gateway | 3M requests | ~$3 |
| Lambda (ingest) | 3M invocations, 128MB, 100ms avg | ~$0.50 |
| Lambda (processor) | 300K invocations, 256MB, 200ms avg | ~$0.30 |
| Lambda (read API) | 1M invocations, 256MB, 150ms avg | ~$0.50 |
| SQS | 3M messages | ~$1.20 |
| Supabase Pro | 8GB DB, higher limits | $25 |
| CloudFront | 50GB | ~$4 |
| Route 53 | queries | ~$1 |

**Estimated monthly cost at medium traffic: ~$35/month**

### 10.3 Cost Guardrails

- **API Gateway throttling:** 100 requests/second default (configurable per API key)
- **Lambda concurrency:** Reserved concurrency = 10 for ingest, 5 for processor
- **SQS alarm:** CloudWatch alarm if DLQ depth > 0
- **Supabase:** Monitor row count; when approaching 500MB, evaluate Supabase Pro ($25/month) vs self-hosted Postgres on RDS

---

## 11. Migration Plan — Wiring the Existing UI

The dashboard currently uses hardcoded mock data. Here's the component-by-component wiring plan:

### 11.1 Add Supabase Client to Dashboard

```bash
pnpm add @supabase/supabase-js @supabase/ssr
```

Create `lib/supabase.ts`:
```typescript
import { createBrowserClient } from '@supabase/ssr'

export const supabase = createBrowserClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
)
```

### 11.2 Component → API Mapping

| Component | Current State | Wire To | Priority |
|-----------|--------------|---------|----------|
| **StatsCards** | Hardcoded `1,247`, `23`, `1.8%`, `94.2%` | `GET /v1/scans/stats` | P0 |
| **DetectionTimeline** | `generateMockData(days)` | `GET /v1/scans/timeline?period=7d` | P0 |
| **RecentScansTable** | `MOCK_SCANS` array (5 items) | `GET /v1/scans?page=1&per_page=20` | P0 |
| **ScanPlayground** | `simulateScan()` client-side | Keep client-side for MVP | P2 |
| **AnalyticsCharts** | All 5 charts use hardcoded/seeded data | `GET /v1/analytics/*` endpoints | P1 |
| **SettingsPage** | All `useState` with hardcoded defaults | `GET/PUT /v1/settings` | P1 |
| **Settings API tab** | Hardcoded API key `ask_live_7f3a...` | `GET/POST/DELETE /v1/settings/api-keys` | P0 |

### 11.3 Auth Gate

Add auth middleware to the dashboard layout:

```typescript
// app/(dashboard)/layout.tsx
import { redirect } from 'next/navigation'
import { createServerClient } from '@supabase/ssr'

export default async function DashboardLayout({ children }) {
  const supabase = createServerClient(/* ... */)
  const { data: { user } } = await supabase.auth.getUser()

  if (!user) redirect('/login')

  return <DashboardShell>{children}</DashboardShell>
}
```

Add login/signup pages under `app/(auth)/login/page.tsx` and `app/(auth)/signup/page.tsx`.

### 11.4 Fields the UI Expects vs SDK Provides

| UI Field | Available from SDK? | Resolution |
|----------|-------------------|------------|
| `is_suspicious` | Yes | Direct |
| `confidence` | Yes | Direct |
| `drift_score` | Yes | Direct |
| `source` | No (not in ScanEvent) | Default to "api" for SDK telemetry; "playground" for UI scans |
| `preview` / `full_text` | No (privacy — SDK never sends text) | Not available. Show `session_id` or "SDK scan" instead |
| `risk_factors` | No | Defer until pattern detector. Keep "Coming Soon" until detector-backed values ship |
| `cleaning_method` | Yes | Direct |
| `timestamp` | Yes | Direct |

**UI Changes Required:**
- `RecentScansTable`: Remove `preview`/`full_text`/`risk_factors` columns for SDK-sourced scans. Replace with `embedding_model`, `environment`, `session_id`.
- `ScanPlayground`: Keep as client-side demo for now. Add `source: "playground"` when saving results to DB.
- `AnalyticsCharts > TopRiskFactors`: Show "Coming Soon" in MVP, then wire to pattern detector-backed `risk_factors` right after dashboard launch.

---

## 12. Security Considerations

### 12.1 Data Privacy
- SDK never sends document text — only metrics. This is enforced at the SDK level (`ScanEvent` has no text field).
- Supabase RLS ensures users only see their own project data.
- API keys are stored as SHA-256 hashes; the raw key is shown only once at creation.

### 12.2 Transport Security
- All endpoints are HTTPS only (API Gateway enforces TLS 1.2+).
- SDK sends gzipped payloads — not for security, but compression. The Bearer token in the Authorization header provides authentication.

### 12.3 Rate Limiting
- API Gateway throttling: 100 req/s per API key (configurable).
- Ingest Lambda caps at 200 events per batch to prevent abuse.
- SQS provides natural backpressure.

### 12.4 Secrets Management
- Supabase service role key stored in AWS SSM Parameter Store (SecureString).
- Dashboard uses Supabase anon key (safe to expose — RLS protects data).
- API keys use the `ask_live_` / `ask_test_` prefix convention for easy identification.

---

## 13. Open Questions

| # | Question | Options | Recommendation |
|---|----------|---------|----------------|
| 1 | **Dashboard hosting: Vercel or CloudFront?** | Vercel free tier is easiest. CloudFront + S3 gives full AWS control. | Vercel for now. Migrate later if needed. |
| 2 | **Read API: Lambda or Supabase direct?** | Lambda adds abstraction. Supabase PostgREST is zero-cost. | Supabase direct for MVP. Add Lambda layer later. |
| 3 | **Playground scan: client-side or real scan?** | Client-side is free. Real scan needs model hosting. | Client-side for MVP. |
| 4 | **Multi-tenancy: how many projects per user?** | 1 (simpler), many (enterprise-ready). | 1 for MVP, schema supports many. |
| 5 | **SDK endpoint migration:** Current SDK default is `api.agentshield.dev`. Move to `api.agentshield.cloud`? | Keep `.dev` as alias or update SDK. | Update SDK default in next release. Keep `.dev` as a CNAME redirect. |
| 6 | **Supabase region:** Where to host? | `us-east-1` matches AWS Lambda. | `us-east-1`. |
| 7 | **Risk factors chart:** What to show before pattern detector? | Mock data or "Coming Soon". | "Coming Soon" in MVP; first post-dashboard schema addition is detector-backed `risk_factors`. |
| 8 | **IaC tool:** SAM, CDK, or Terraform? | SAM is simplest for Lambda+API Gateway. CDK for complex infra. | SAM for MVP. |

---

## 14. Platform Strategy and Value Model

### 14.1 Product Positioning

AgentShield Cloud should be positioned as a **runtime safety and reliability control plane for AI agents**.

What users buy:
- Faster incident response for prompt injection/jailbreak-style failures
- Shared governance for teams shipping agent workflows
- Evidence that guardrails are effective over time
- Lower operational burden than self-building telemetry + review tools

What users should not perceive:
- "Yet another dashboard of generic charts"
- "A hosted wrapper around open models with little additional value"

### 14.2 Open-Core Boundary (Recommended)

| Keep in OSS SDK | Keep in Cloud Platform |
|---|---|
| Multi-provider model support | Team/org controls and role management |
| Local detection + cleaning pipeline | Hosted event retention and analytics |
| Basic calibration + local thresholds | Cross-project dashboards and investigations |
| Basic telemetry emission client | Managed alerting, webhooks, incident workflow |
| Local CLI utilities | Managed policy rollouts + audit trails |

This boundary supports adoption while preserving clear platform value.

### 14.3 Core Personas and Jobs

| Persona | Primary Job | Platform Feature Priority |
|---|---|---|
| Agent developer | Validate prompts/flows are safe | Playground, scan history, confidence/drift visibility |
| Security engineer | Enforce org-level policy | Policies, alerts, auditability, key management |
| Platform owner | Scale safely across teams | Multi-project views, governance, onboarding templates |

---

## 15. Standards and Schema Governance

### 15.1 Canonical Internal Schema

Adopt a versioned internal event schema (`schema_version`) as the source of truth and treat external standards as export/mapping layers.

Required baseline fields:
- `event_id`, `timestamp`, `project_id`, `environment`
- `is_suspicious`, `confidence`, `drift_score`, `threshold`
- `embedding_model`, `cleaning_method`, `on_detect`
- `pipeline_fingerprint` (for calibration identity consistency)
- `telemetry_version`, `sdk_version`

### 15.1.1 ScanEvent v1 Freeze (SDK + Cloud Contract)

`ScanEvent` v1 should include the following additional operational fields at schema freeze time:

| Field | Type | Purpose |
|---|---|---|
| `schema_version` | `int` | Forward-compatible evolution and parser routing |
| `pipeline_fingerprint` | `str` | Calibration identity and cross-pipeline analytics |
| `source` | `str` | Distinguish `api` / `playground` / `webhook` paths |
| `latency_ms` | `float` | Operational performance visibility per scan |
| `text_length` | `int` | Volume/cost analytics without sending text |
| `detector_type` | `str` | `zedd` now; forward-compatible for pattern/ensemble detectors |

Implementation note:
- The SDK already has fingerprint construction in threshold management; emit the same fingerprint context from scan emission path.
- `embedding_model` must use the runtime-resolved provider model identifier, not only config input values.

Not included in v1 baseline:
- `risk_factors` (defer until pattern detector produces meaningful values)
- `preview` / `full_text` (remain opt-in only under privacy controls)

### 15.2 Compatibility Strategy

Use dual compatibility:
- **Internal schema** for product reliability and migration control
- **Mapping adapters** for OpenTelemetry/OpenInference-style interoperability

This prevents hard lock-in to external spec churn while preserving integration paths.

Phase 0 compatibility requirement:
- Add `schema_version` now and structure event keys so they are cleanly mappable to OTel-style attributes later (for example, stable dotted namespaces such as `attributes.model.embedding`, `attributes.pipeline.fingerprint`, `attributes.scan.latency_ms` in mapping/export layers).
- Build exporter/bridge later; do not block MVP ingest on full OTel implementation.

### 15.3 Schema Evolution Policy

- Additive fields are backward compatible
- Breaking field changes require major `schema_version` bump
- Maintain decode support for at least 2 prior versions
- Enforce contract tests between SDK emitter and ingest parser in CI

### 15.4 Privacy Data Classes

| Class | Example | Default | Controls |
|---|---|---|---|
| `P0` metadata | scores, model names, method, env | Enabled | Stored by default |
| `P1` user text | prompt or cleaned prompt | Disabled | Explicit opt-in + retention policy |
| `P2` sensitive IDs | secrets/PII fields | Blocked | Redaction + denylist validators |

---

## 16. Launch Readiness Gaps

These are the high-impact gaps identified from current SDK/UI contracts and should be closed before broad launch.

### 16.1 Endpoint Domain Consistency

Current inconsistencies:
- SDK default telemetry endpoint still points to `.dev`
- Platform settings mock includes `.io`
- Target production domain is `.cloud`

Decision:
- Set canonical production endpoint to `https://api.agentshield.cloud/v1/telemetry`
- Keep `.dev` as temporary alias during migration window
- Remove `.io` references from platform code/docs

### 16.2 Telemetry Accuracy: Effective Embedding Model

Risk:
- Telemetry can report configured model name rather than effective resolved model in some provider paths.

Decision:
- Emit the **actual runtime model identifier** used for embedding requests.
- Add regression test for provider override cases.
- SDK implementation target: emit provider-resolved model value from runtime provider object in scan emission path.

### 16.3 UI Contract vs Telemetry Payload

Risk:
- UI tables/charts expect `preview`, `full_text`, and `risk_factors` that are not emitted by SDK metadata-only telemetry.

Decision:
- For SDK data path, remove text-dependent assumptions and use metadata-safe alternatives.
- Keep `risk_factors` as "Coming Soon" until pattern detector output is in schema.
- Commit to shipping `risk_factors` immediately after dashboard MVP launch as the first post-launch schema extension tied to pattern detector release.

### 16.4 Calibration Identity Consistency

Status:
- Pipeline fingerprinting work has improved threshold keying integrity.

Follow-up:
- Ensure all read/write threshold paths use the same fingerprint builder.
- Keep migration handling explicit and documented for pre-fingerprint caches.

### 16.5 Multi-Provider Expansion Guardrails

Decision:
- Prefer OpenAI-compatible `base_url` support as default adapter pattern
- Add provider-specific implementations only where semantics differ materially
- Keep per-provider capability tests (auth, dimensions, host keying, batch behavior)

---

## 17. SLOs, Operations, and Incident Model

### 17.1 Initial SLOs (MVP)

| Area | SLO Target |
|---|---|
| Telemetry ingest availability | 99.9% monthly |
| Ingest API latency | p95 < 300 ms (202 response) |
| Queue-to-database freshness | p95 < 60 seconds |
| Dashboard query latency | p95 < 800 ms for standard filters |
| Data durability | At-least-once ingestion + idempotent dedup by (`project_id`, `event_id`) |

### 17.2 Operational Alerts

Create CloudWatch/Supabase alerts for:
- SQS DLQ depth > 0
- Ingest Lambda error rate > 1% for 5 min
- Processor retry storms / age of oldest SQS message > 120s
- API 5xx rate > 1% for 5 min
- Supabase write failures and quota limits

### 17.3 Incident Workflow

1. Alert triggers and routes to on-call channel
2. Check queue health and ingest success first
3. Verify dedup and write throughput
4. Backfill from SQS/DLQ replay path if needed
5. Publish post-incident note with customer impact and mitigation

---

## 18. Execution Roadmap (0-90 Days)

### Phase 0 (Week 1-2): Contract Hardening
- Finalize canonical domains (`.cloud`) and migration plan for `.dev`
- Freeze telemetry schema v1 with contract tests:
  - Add fields: `schema_version`, `pipeline_fingerprint`, `source`, `latency_ms`, `text_length`, `detector_type`
  - Fix runtime `embedding_model` emission to use resolved provider model identifier
  - Add SDK→ingest contract tests for serialized `ScanEvent` compatibility
- Close endpoint and field mismatches between SDK and platform UI

### Phase 1 (Week 3-6): MVP Data Plane
- Launch ingest + queue + processor + Supabase persistence
- Wire dashboard P0 views: stats, timeline, recent scans, API keys, settings
- Add baseline SLO dashboards and alerting

### Phase 2 (Week 7-10): Investigation UX
- Add richer filtering, event drill-down, and session-centric traces
- Ship webhook/slack notifications with threshold routing
- Add replay-friendly event export for debugging
- Ship `risk_factors` field and chart wiring as first post-dashboard schema enhancement (paired with pattern detector output)

### Phase 3 (Week 11-13): Monetizable Controls
- Team/org roles and policy templates
- Audit log and change history
- Managed calibration workflows and cross-project insights

---

## 19. Company-Style Best Practices Applied

These recommendations align with patterns used by successful AI tooling companies while fitting AgentShield's stage:

- **Own the control loop, not just collection.** Detection alone commoditizes quickly; remediation and verification retain value.
- **Adopt standards with a translation layer.** Keep internal schema ownership; expose standards adapters for ecosystem fit.
- **Default-deny on sensitive data.** Metadata-first ingestion reduces compliance friction and shortens sales cycles.
- **Design for low idle cost.** Serverless + queue + managed Postgres is the right early-stage cost/perf profile.
- **Ship narrow and complete first.** A reliable core workflow (detect, investigate, act) beats broad but shallow feature surfaces.
- **Instrument change safety.** Every schema or pipeline change should have compatibility tests and migration checks.
