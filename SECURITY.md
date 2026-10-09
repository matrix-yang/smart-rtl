# Security Policy

## Reporting Security Issues

Please do not report security vulnerabilities through public GitHub issues,
pull requests, discussions, or other public channels.

Report Smart-RTL vulnerabilities privately to the maintainers through the
private security-reporting channel of the repository host or distribution from
which you obtained this code. If the issue is in unmodified upstream ACE-RTL
code, NVIDIA Product Security accepts reports at
https://www.nvidia.com/en-us/security/report-vulnerability/.

Avoid public disclosure until the relevant maintainers have completed triage
and remediation coordination.

## Supported Scope

Security triage for this repository covers the current default branch and the
latest published release, when a release exists.

In scope:

- ACE-RTL scripts and reusable agent components under `skills/ace-rtl/scripts/`
- ACE-RTL and EDA setup skill content under `skills/`
- Repository setup, dependency, and benchmark-integration logic
- Handling of API keys, license-related environment variables, prompts,
  generated candidates, evaluator reports, and run artifacts produced by this
  repository

Out of scope for this repository:

- Vulnerabilities in third-party dependencies, unless ACE-RTL uses them in an
  unsafe way
- Vulnerabilities in external simulator, synthesis, coverage, or EDA tools
- Vulnerabilities in the upstream CVDP benchmark repository or downloaded CVDP
  datasets, except where ACE-RTL integration code creates the issue
- User-provided `ACE_RTL_LLM_SCRIPT` adapters outside this repository
- Local machine, cluster, Docker daemon, license-server, or site-specific EDA
  configuration issues
- Generated files under `outputs/` unless they expose a vulnerability in
  ACE-RTL itself

## What To Include

Include enough detail for NVIDIA PSIRT and maintainers to reproduce and assess
the issue:

- Affected ACE-RTL commit, tag, or branch
- Affected file paths, command line, dataset row, or CVDP category when relevant
- Environment details, including Python version, operating system, simulator
  backend, and whether Docker or the native runner was used
- Steps to reproduce from a clean checkout
- Proof of concept or minimal trigger, if available
- Expected and observed behavior
- Potential impact, including whether secrets, private benchmark assets, host
  files, external services, or generated artifacts may be exposed or modified

Do not include live API keys, license file contents, private credentials,
private benchmark assets, or confidential third-party source code in the report.
If a secret was exposed, revoke or rotate it before submitting the report.

## Secret Handling

ACE-RTL expects credentials and license-related values to come from the runtime
environment, not from committed files.

- Keep `DEEPSEEK_API_KEY`, `NVIDIA_API_KEY`, and other provider credentials in
  the shell environment or a local untracked `.env` file.
- Keep EDA license variables such as `CDS_LIC_FILE`, `LM_LICENSE_FILE`,
  `SNPSLMD_LICENSE_FILE`, and `MGLS_LICENSE_FILE` out of committed logs,
  reports, prompts, and configuration files.
- Do not commit API keys, bearer tokens, license file contents, vendor account
  credentials, cluster credentials, or service-specific secrets.
- Before sharing logs or run directories, inspect `outputs/` for prompts,
  responses, environment summaries, command lines, and evaluator reports that
  may contain sensitive values.

## External LLM And Benchmark Data Safety

External LLM calls are disabled by default unless a user explicitly assigns a
model or configures `ACE_RTL_LLM_SCRIPT`. When using an external model or custom
adapter:

- Send only public task prompts, allowed generated target files, current
  candidates, role history, and bounded evaluator reports needed for debugging.
- Do not send private scorer files, real hidden test files, expected-output
  code, mutation definitions, private harness internals, or raw logs that reveal
  private test structure.
- Do not print or persist API keys in prompts, responses, reports, or run logs.
- Treat custom adapters as trusted local code; review them before use and run
  them with the least privileges practical for the evaluation environment.

## Dependency And Environment Updates

The top-level `requirements.txt` is the authoritative Python dependency
manifest for ACE-RTL. When updating dependencies or setup scripts:

- Prefer pinned versions and review changelogs for security-impacting behavior.
- Rebuild a clean local environment before evaluating security-sensitive
  changes.
- Keep CVDP data, Python environments, prompts, scripts, and outputs
  repository-local unless explicitly authorized.
- Record tool command names and non-secret environment variable names needed for
  reproducibility, but never record secret values or license file contents.
