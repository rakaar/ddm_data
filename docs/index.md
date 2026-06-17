# DDM Data Result Book

This MkDocs book is for date-wise research results from this repository.

Use `docs/results/YYYY-MM-DD.md` for daily result pages. Generated figures should live under `docs/assets/results/YYYY-MM-DD/`, and each entry should include:

- the figure
- a short caption
- the path of the Python file or notebook that generated it

Run the local docs server with:

```bash
.venv/bin/python -m mkdocs serve
```

The MkDocs config binds the dev server to `0.0.0.0:8000`, so it is reachable from the lab network and through Tailscale while the server is running.

Current access URLs:

- Local machine: <http://127.0.0.1:8000/>
- Lab network: <http://10.40.49.28:8000/>
- Tailscale IP: <http://100.88.161.67:8000/>
- Tailscale MagicDNS: <http://lavos.tailbc8bf8.ts.net:8000/>

Current server process:

```bash
tmux attach -t result-book
```

Stop it with:

```bash
tmux kill-session -t result-book
```
