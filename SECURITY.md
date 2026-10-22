# Security

PaperTrail is a static website. There is no server, database, login or API key
behind it, so the usual ways a web app gets taken over or leaks data (server
code like PHP, SQL injection, stolen credentials, a breached user database) do
not apply. This file lists what I did to keep the rest safe.

## How the app is protected

- **No backend to attack.** GitHub Pages serves the built files over HTTPS.
  Everything else, including parsing and the ML models, runs in the visitor's
  browser.
- **No secrets.** The repository holds no keys or tokens. The app only calls
  public APIs (DataCite, OpenAlex, Crossref, Semantic Scholar, arXiv) that do
  not need one.
- **Your papers stay on your device.** PDFs, analyses and answers are stored in
  the browser's IndexedDB. Nothing is uploaded, and there is no analytics or
  tracking.
- **Paper content can't run as code.** Text from PDFs and paper metadata is
  rendered by React as plain text, never as HTML. Links that come from that
  data are only shown if they are ordinary `http(s)` links, so a crafted
  `javascript:` link is dropped.
- **Content Security Policy.** The built page only runs its own scripts (plus
  one hashed inline script that sets the theme), blocks plugins and frames, and
  only makes HTTPS requests.
- **Locked dependencies.** `package-lock.json` pins every package, CI installs
  with `npm ci`, and Dependabot opens a pull request when an npm package or
  GitHub Action has a known vulnerability.
- **Least-privilege CI.** Workflows can only read the repository. Only the
  Pages deploy job may publish, and it can't push code.

## Known limits

- `npm audit` reports moderate advisories in `onnxruntime-node` and its
  dependencies, which transformers.js pulls in for Node.js. They run only when
  I use the evaluation script locally and are never part of the website. I
  pinned `sharp` to a patched release to clear the high-severity ones.
- GitHub Pages does not let a site set its own response headers, so the policy
  is delivered as a `<meta>` tag. That covers everything above except blocking
  other sites from framing the page; PaperTrail has no account or action worth
  clickjacking.

## Reporting a problem

Please report anything you find privately through
[GitHub's vulnerability reporting](https://github.com/Gaganpraveen/papertrail/security/advisories/new)
rather than in a public issue.
