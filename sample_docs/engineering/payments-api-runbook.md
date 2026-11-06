---
id: eng-payments-runbook
title: payments-api Runbook
acl: [group:engineering, group:sre]
owner: payments-platform
updated_at: 2026-09-14
uri: https://wiki.example.internal/eng/payments-api-runbook
---

# Overview

payments-api is the service that authorizes and captures card payments. It runs on the
`prod-east` and `prod-west` Kubernetes clusters in the `payments` namespace and is owned by
the Payments Platform team. Pager rotation: `payments-oncall`.

# Common errors

## PAY-4012: Upstream processor timeout

PAY-4012 means the card processor did not respond within 3 seconds. Usually transient.

1. Check the processor status page linked in the dashboard header.
2. If the error rate is above 2% for 10 minutes, enable the fallback processor with the
   feature flag `payments.fallback_processor=true`.
3. Open an incident in the #inc-payments channel if fallback does not bring errors under 1%.

## PAY-5001: Idempotency key conflict

Raised when a client retries a request with the same idempotency key but a different body.
This is a client bug, not a service fault. Do not page. File a ticket for the calling team.

# Secret rotation

Database and processor credentials are stored in Vault under `kv/payments/prod`.
Rotate them every 90 days or immediately after any suspected exposure.

```bash
vault kv put kv/payments/prod processor_api_key=<new-key>
kubectl -n payments rollout restart deploy/payments-api
```

After the rollout, confirm the `auth_success_rate` metric stays above 99.5% for 15 minutes.
