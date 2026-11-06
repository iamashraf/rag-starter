---
id: eng-oncall-handbook
title: On-call Handbook
acl: [group:engineering]
owner: sre
updated_at: 2026-07-02
uri: https://wiki.example.internal/eng/oncall-handbook
---

# Severity levels

| Level | Meaning | Response time |
|-------|---------|---------------|
| SEV1 | Customer-facing outage or data loss | 15 minutes, 24x7 |
| SEV2 | Major feature degraded, workaround exists | 1 hour, 24x7 |
| SEV3 | Minor issue, no customer impact | Next business day |

# Handoff

On-call shifts run Monday 10:00 to Monday 10:00 local time. The outgoing engineer writes a
handoff note in the #oncall channel listing open incidents, silenced alerts and anything
flaky. The incoming engineer acknowledges in the thread.

# Escalation

If you cannot make progress on a SEV1 within 30 minutes, escalate to the secondary on-call,
then to the engineering manager for the owning team. Never sit on a SEV1 alone.

# Compensation for on-call

On-call engineers receive a weekly stipend and one day of time off in lieu after any week
with a SEV1 page outside business hours. Log it in the HR portal under "On-call TOIL".
