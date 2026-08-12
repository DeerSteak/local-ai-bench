# Security Policy

## Reporting a vulnerability

Report suspected vulnerabilities through [GitHub private vulnerability reporting](https://github.com/DeerSteak/local-ai-bench/security/advisories/new). Do not open a public issue for an uncoordinated vulnerability report or include credentials, embargoed hardware details, private results, prompts, responses, logs, or customer information in public discussion.

The security owner is the repository maintainer, [@DeerSteak](https://github.com/DeerSteak). The private advisory is the authoritative reporting and coordination channel.

Include the affected version or commit, platform, impact, minimum reproduction, and any known workaround. Share only the evidence needed to reproduce the issue, and redact secrets and unrelated private data.

The maintainer aims to acknowledge a report within three business days, provide an initial severity assessment or request for more information within seven business days, and coordinate disclosure after a fix or containment plan exists. Active credential exposure, release compromise, or cross-user data exposure receives immediate priority. These targets may change as project staffing evolves; the private advisory remains the authoritative coordination channel.

## Supported versions

| Version | Security fixes |
|---|---|
| Latest release on `main` | Supported |
| Active `release/X.Y` branch | Supported until release or abandonment |
| `develop` and preview builds | Best effort; fixes may require updating to a newer commit |
| Older releases | Not supported unless a separate advisory says otherwise |

Security fixes follow the repository's hotfix and release reconciliation process. A fix may be withheld from public branches until coordinated disclosure is appropriate.

## Scope and safe testing

In scope are vulnerabilities in this repository's code, packaging, release automation, local APIs, archive handling, credential handling, generated-code containment, update paths, and exported artifacts. Third-party vulnerabilities in llama.cpp, ComfyUI, model files, drivers, or operating systems should normally be reported to their maintainers unless this project introduces or amplifies the issue.

Do not access other people's systems or data, disrupt shared services, publish secrets, run denial-of-service tests against infrastructure you do not own, or download multi-gigabyte assets solely to demonstrate a report. The project does not operate a bug bounty and cannot authorize testing of third-party services.

The full product threat model and data-handling requirements are documented in [Security and Privacy](docs/security-and-privacy.md).
