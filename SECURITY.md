# Security Policy

## Supported Version

Only the latest tagged release receives fixes. TinyML_NPU is a research and teaching prototype and has not been designed or audited for safety-critical, security-critical, or production deployment.

## Reporting a Vulnerability

Use GitHub's private vulnerability reporting or Security Advisory interface for the repository. Please include the affected commit, reproduction steps, impact, and any suggested mitigation. Do not open a public issue before the maintainer has had a reasonable opportunity to assess the report.

The project does not promise a fixed response SLA. Reports that affect host scripts, generated firmware, memory-range validation, or untrusted model parsing are in scope. Vulnerabilities that require unsupported modifications outside the documented v0.1.0 configuration may be closed as out of scope with an explanation.

## Deployment Notice

The ZYBO7010 demo assumes trusted firmware, trusted bundles, physical access to the board, and a trusted JTAG host. It does not implement secure boot, model authentication, privilege isolation, or adversarial input hardening.
