# Security policy

## Report a problem privately

Do not open a public issue, discussion or pull request for a security problem.

Report it privately in one of these two ways:

1. On GitHub: open the repository's **Security** tab and click
   **Report a vulnerability**
   (<https://github.com/mczielinski/ob-analytics/security/advisories/new>).
   Only the maintainer can read the report. If the tab has no
   **Report a vulnerability** button, use email.
2. By email to the maintainer: mczielinski@gmail.com. Put "ob-analytics
   security" in the subject.

Include:

- the ob-analytics version (`pip show ob-analytics`), or the commit if you
  use `main`;
- what an attacker can do, and what they need first (a file you open, a key
  you set, a venue that sends a message);
- the steps or a small file that shows the problem.

The maintainer answers within 14 days. When the fix is ready, the maintainer
publishes it in a release and a GitHub security advisory, and names you in the
advisory unless you ask not to be named.

## Supported versions

Report a problem in `main` or in the newest release on PyPI. A fix goes into
`main` and then into a new release. The package does not publish fixes for
older releases, so upgrade to get the fix.

## What counts

ob-analytics reads files that other people make, and it holds API keys for
live capture. These are security problems:

- A file that makes ob-analytics run code, or read or write a file outside the
  folder it was given. This covers every file the package reads: the source
  formats (Bitstamp, LOBSTER, Databento, depth CSV), saved Parquet runs, and a
  capture's `meta.json` and `manifest.json`.
- An API key that reaches a file, a log line or an error message. The
  [Use an API key](https://mczielinski.github.io/ob-analytics/howto/api-keys/#what-ob-analytics-does-with-a-key)
  page says what the package does to keep a key out of them.
- A key that goes to a host other than the venue it is for.

These are not security problems; open a normal issue for them:

- a wrong number, a crash, or a slow run on a file you made yourself;
- a large file that uses all the memory (the package keeps a run in memory;
  see [Scale and chunking](https://mczielinski.github.io/ob-analytics/scale-and-chunking/));
- a problem in a dependency (NumPy, pandas, ccxt, cryptofeed, …) that
  ob-analytics does not make worse. Report that to the dependency.
