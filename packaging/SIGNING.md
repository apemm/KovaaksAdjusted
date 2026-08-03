# Code signing

Windows 11's **Smart App Control** blocks unsigned binaries outright. Measured
on a real machine, launching an unsigned `kovadapt.exe` produced CodeIntegrity
event **3077** under policy `VerifiedAndReputableDesktop` (SAC's own policy),
Requested Signing Level 2 against Validated 1, with publisher and issuer both
`Unknown`. Nothing else was wrong with it.

SAC's rule is an **OR**, not an AND:

> Apps cannot be run unless they are recognized by Microsoft's app intelligence
> services, **or** they are signed with a trusted certificate.
> — [Smart App Control overview](https://learn.microsoft.com/en-us/windows/apps/develop/smart-app-control/overview)

So reputation is the alternative path *for unsigned code*, not an extra hurdle
stacked on a signature. A brand-new certificate with no reputation should clear
the block on the first signed build. Signing does **not** silence SmartScreen —
expect a dismissible "Windows protected your PC" prompt for some weeks while
reputation accrues. Signing converts a hard block into a warning.

`.github/workflows/release.yml` does the signing. It is gated on the variables
below, so until they exist it publishes an unsigned zip and logs a warning.

## What has to be true

- **Sign every PE, not just the exe.** SAC checks each module as it loads. A
  sweep of a real build: 101 binaries, 80 already validly signed (55 Qt, 22
  Python Software Foundation, 3 Microsoft) and **21 unsigned** —
  `kovadapt.exe`, twelve numpy `.pyd`, OpenBLAS, `libcrypto-3-x64.dll`,
  `libssl-3-x64.dll`, psutil, yaml and two charset_normalizer modules. Signing
  only the launcher leaves it blocked.
- **RSA, not ECC.** SAC's signature check does not support elliptic-curve
  signatures. Artifact Signing's default path is RSA, so this is automatic
  there — it is the trap if the CA ever changes.
- **Timestamp everything.** Artifact Signing certificates are valid for
  **three days**; an untimestamped signature is dead in 72 hours.
- **Sign before zipping**, and keep the build one-dir. With `--onefile` the
  payload would have to be signed inside `site-packages` before PyInstaller
  ran.
- **Never recreate the certificate profile.** It resets accumulated SmartScreen
  reputation. One stable signing identity across every release.

## One-time Azure setup (needs a person, not CI)

Requires a **paid** pay-as-you-go subscription — free, trial and sponsored
subscriptions are not supported — and Azure CLI ≥ 2.75.0.

```bash
az login
az extension add --name artifact-signing
az provider register --namespace Microsoft.CodeSigning   # legacy namespace, unchanged by the rename

az group create -n kovadapt-signing -l eastus
az artifact-signing create -n kovadaptSigning -l eastus -g kovadapt-signing --sku Basic
```

Basic is $9.99/month: 5,000 signatures/month against roughly 101 per release.

Then in the portal, **Identity validation → Individual**. The identity fields
come from the Azure *billing account*, which must have Account Type =
Individual, and the legal name and address must match a government photo ID
exactly; validation runs a live FaceCheck. Individuals are limited to the
**USA and Canada**.

> The certificate's subject is the **validated legal name** on the billing
> account — it cannot be set to "kovadapt". Here that means Windows will show
> the publisher as **Arjun Pemmasani**, on the SmartScreen prompt and in the
> file's digital-signature tab.
>
> That is no new disclosure: the same name is already the LICENSE copyright
> holder, the `authors` entry in `pyproject.toml`, and the author of every
> commit in the public history. The thing it does change is liability — a
> signature ties the binary to a verified identity, which is the point, and
> also why the PUA note below matters.

Once validated:

```bash
az artifact-signing certificate-profile create \
  -g kovadapt-signing --account-name kovadaptSigning \
  -n kovadapt-public --profile-type PublicTrust
```

## Wiring GitHub to it, without secrets

Create a user-assigned managed identity (or app registration), add a federated
credential scoped to this repository's tags —
`repo:apemm/KovaaksAdjusted:ref:refs/tags/*` — and grant it the **Code Signing
Certificate Profile Signer** role on the signing account.

Then add these as repository **Variables** (Settings → Secrets and variables →
Actions → Variables). They are identifiers, not secrets; OIDC carries the trust:

| Variable | Value |
|---|---|
| `AZURE_CLIENT_ID` | the managed identity / app registration client ID |
| `AZURE_TENANT_ID` | your tenant ID |
| `AZURE_SUBSCRIPTION_ID` | the paid subscription ID |
| `AZURE_SIGNING_ACCOUNT` | `kovadaptSigning` |
| `AZURE_CERT_PROFILE` | `kovadapt-public` |
| `AZURE_SIGNING_ENDPOINT` | `https://eus.codesigning.azure.net/` (match the account's region) |

Rehearse it with **Actions → release → Run workflow** before tagging: on a
manual run the workflow builds, signs and uploads the zip as a build artifact
without touching a release.

## Before signing under a personal legal name

kovadapt writes an HKCU Run key, tunes process priority and affinity, and reads
Game DVR, Game Mode, timer resolution and core parking. Those are
potentially-unwanted-application heuristics, and Microsoft's guidance is to not
sign files exhibiting them, with certificate revocation and account suspension
as consequences. Because the certificate carries a validated personal identity,
a bad classification is personally costly.

The current evidence is reassuring rather than alarming: the same CodeIntegrity
block above also logged event **3118** with `IsUnfriendlyFile: false` and
`DefenderStatusCode 0x0`, after a completed cloud lookup — Defender examined
this build and did not classify it as unwanted. Keep it that way: explicit
consent for the startup entry, an obvious opt-out, nothing silent.

## Renewal

Individual identity validation expires. Reminders begin 60 days out; if it
lapses, certificate renewal stops and every signing operation on the affected
profiles stops with it.
