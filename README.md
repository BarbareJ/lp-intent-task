# LP Visitor Intent Segmentation

This is my take-home for the data scientist role — a behavior-based intent segmentation for landing-page visitors, plus a read on the variant B question and the appointment-quality question.

## What's here

- `writeup.md` / `lp_intent_writeup.pdf` — the actual write-up. Start here if you just want the analysis and the recommendations.
- `lp_intent.py` — the script that produces everything else. Runs end to end from the raw CSV to every table and chart referenced in the write-up.
- `lp_intent.ipynb` — the same analysis as a notebook, if that's easier to skim on GitHub. Outputs are saved, so it renders with results already visible.
- `lp_sessions.csv` — the data, unchanged from what was provided.
- `outputs/` — every table behind the write-up, as plain CSVs, so any number in there can be checked against its source.
- `figures/` — the charts.
- `requirements.txt` — what you need to run the script.

## Running it

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python lp_intent.py --input lp_sessions.csv --output-dir outputs --figure-dir figures
```

That regenerates every file in `outputs/` and `figures/` from scratch.

## A few decisions worth knowing about going in

The main segmentation only uses pre-form behavior — duration, scroll depth, clicks, sections viewed, time to first scroll. Nothing about the form itself, and nothing about channel or device. That was a deliberate call: the ask was for a *behavior*-based segmentation, and if I'd let campaign/device into the main score, "high intent" could quietly turn into a proxy for "came from Google on desktop" instead of reflecting what someone actually did on the page. I did fit a version with that extra context in, and a version that includes form-behavior fields too, but only to benchmark against — not as the segmentation itself. The form-behavior version scores almost perfectly, which is exactly why those fields are excluded from the real thing: they're basically a delayed readout of the outcome, not an early signal.

Every tier score comes from out-of-fold predictions, meaning a session is only ever scored by a model version that didn't train on it. Otherwise the lift numbers would be flattering the model on data it already knows.

For variant B, I didn't stop at the raw comparison. B looks slightly worse on a straight conversion comparison, but that gap isn't significant, and B happens to be getting a lot more mobile traffic than A — which converts worse regardless of the page. I checked whether B might have been a phased rollout rather than a real concurrent split (it isn't — its weekly share is flat the whole time), then adjusted for the pre-existing traffic mix, and B comes out ahead once you do that. That whole chain of reasoning is in the write-up rather than just the headline number.

On lead quality: I kept "how likely is someone to convert" and "how good is the lead once they do" as two separate questions, since `appointment_set` only exists for people who actually converted. High intent still produces the most leads and appointments by volume, but medium/low-intent leads book appointments at a higher rate once they convert — which is the kind of thing that should change how budget conversations get framed, even if it doesn't change who gets prioritized for immediate conversion.

The real-time scoring section goes a bit further than the prompt technically asks for — it's optional, but I built a small working scorer and simulated it against real sessions rather than just describing the idea, since it seemed more useful to show than to tell.
