# Verify Local AI Bench 6.1

Download all release assets into one directory. The signed checksums authenticate the source archive, SBOM, notices, provenance, public key, and these instructions. Obtain the expected public-key fingerprint independently from the maintainer or an already trusted release; a key downloaded beside a signature does not establish trust by itself.

Expected ED25519 fingerprint: `SHA256:p6Vheb5UuDuKpSTmcoDg4xN8DTPDgItwVR8LdJ4UbZk`.

Run in that directory with OpenSSH and Python 3 installed:

```sh
ssh-keygen -lf local_ai_bench_signing.pub
python3 -c "from pathlib import Path; Path('allowed_signers').write_text('DeerSteak@users.noreply.github.com ' + Path('local_ai_bench_signing.pub').read_text())"
ssh-keygen -Y verify -f allowed_signers -I DeerSteak@users.noreply.github.com -n file -s SHA256SUMS.sig < SHA256SUMS
python3 - <<'PYVERIFY'
import hashlib
from pathlib import Path
for line in Path('SHA256SUMS').read_text().splitlines():
    expected, name = line.split('  ', 1)
    if Path(name).name != name:
        raise SystemExit(f'Invalid checksum filename: {name}')
    actual = hashlib.sha256(Path(name).read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f'Checksum mismatch: {name}')
    print(f'OK: {name}')
PYVERIFY
```

All commands must succeed. The shell commands use POSIX syntax; Windows users can run them in Git Bash with Python 3 and OpenSSH available. Do not execute extracted source before verification succeeds.

In a repository checkout, verify the signed release tag using the trusted allowed-signers file:

```sh
git fetch origin tag v6.1
git -c gpg.ssh.allowedSignersFile=/absolute/path/to/allowed_signers verify-tag v6.1
```

`provenance.json` names the exact source commit. The source archive intentionally excludes the subsequently committed public release assets. The signed tag points to the main release merge, which contains the source and those assets. Retained branch `release/6.1` preserves the release preparation and payload commits.
