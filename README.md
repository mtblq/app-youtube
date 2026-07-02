# app-template

One application of the RSS Summarizer. Reads its feed list from the private
feeds repo, AI-translates/rewrites new items, and publishes a single feed.

- Edit **feeds**: the matching file in the private repo.
- Edit **prompt / output_html / model**: `config.yml`.
- Edit **schedule**: the `cron:` line in `.github/workflows/aggregate.yml`.

Secrets required: `FEEDS_TOKEN`, `OPENAI_API_KEY`.
Output: `feed.xml` on this repo's GitHub Pages site.
See README-ARCHITECTURE.md for full setup.
