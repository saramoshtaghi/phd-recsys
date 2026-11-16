#!/bin/bash
cd /home/moshtasa/Research/phd-recsys || exit 1
while kill -0 2552727 2>/dev/null; do sleep 30; done
python -c 'import json; from pathlib import Path; s=json.loads(Path("First_paper_material/notebook/shopping_reviewer_status.json").read_text()); assert s["stage"]=="all measurements complete",s' || exit 1
python First_paper_material/notebook/render_shopping_reviewer.py || exit 1
git add -- First_paper_material/figures/first_reviewer/shopping First_paper_material/notebook/shopping_reviewer_components.csv First_paper_material/notebook/shopping_reviewer_figures_manifest.json First_paper_material/notebook/shopping_reviewer_status.json
git commit -m 'Add measured Shopping training and prediction figures' -- First_paper_material/figures/first_reviewer/shopping First_paper_material/notebook/shopping_reviewer_components.csv First_paper_material/notebook/shopping_reviewer_figures_manifest.json First_paper_material/notebook/shopping_reviewer_status.json && git push
