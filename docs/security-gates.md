# Release security gates

`scripts/security_gate.py` performs the deterministic offline portion of release scanning against a prepared staging directory. It rejects symbolic links, prohibited credential filenames such as `hf.txt` and `.env`, files too large to inspect under the declared limit, unreadable files, and common Hugging Face, AWS, private-key, and bearer-credential patterns. Findings contain only a relative filename and category, never the matched value.

This gate scans final staged bytes rather than assuming Git ignore rules or source review kept secrets out of a package. A clean result means only that its narrow offline rules found no listed condition. It does not detect every secret, malicious binary, vulnerable dependency, unsafe model, or compromised build environment.

Stable release automation must additionally run dependency vulnerability analysis for Python and npm lock inputs, language/static analysis, malware and platform package inspection, artifact-signature/provenance verification, and a maintained secret scanner with reviewed findings. Network-backed scanners archive their database/tool versions and results. Any suppression names the finding, reason, approver, scope, and expiry; a broad path exclusion is not an acceptable substitute for inspecting shipped content.
