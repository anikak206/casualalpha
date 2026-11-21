PYTHON ?= python

.PHONY: run phase1 test clean

run: phase1 phase2 phase3 phase4 phase5 phase6

phase1:
	$(PYTHON) -m src.data

phase2:
	$(PYTHON) -m src.causal

phase3:
	$(PYTHON) -m src.regimes

phase4:
	$(PYTHON) -m src.phase4

phase5:
	$(PYTHON) -m src.sentiment

phase6:
	$(PYTHON) -m src.report

test:
	$(PYTHON) -m pytest -q

clean:
	rm -rf data/processed results/*.csv results/*.png
