# Financial preprocessing research

This folder groups the final outputs of the research/preprocessing part.

The preprocessing pipeline builds alternative datasets using:

- Time bars.
- Count bars, used as a daily proxy for tick bars.
- Volume bars.
- Dollar bars.
- Fractional differentiation sequence summaries when available.

The generated `.npz` training sequences are intentionally not committed because they are heavy; they can be regenerated with:

```bash
bash model/preprocessing/run_build_sequences.sh
```
