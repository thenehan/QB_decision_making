PYTHON ?= python
DATA_DIR ?= .
OUTPUT_DIR ?= expected_yards_output_all_weeks
SMOKE_OUTPUT_DIR ?= expected_yards_output_smoke
GRAPH_OUTPUT_DIR ?= graph_transformer_all_weeks
GRAPH_SMOKE_OUTPUT_DIR ?= graph_transformer_smoke
WEEKS ?= 1 2 3 4 5 6 7 8 9
SMOKE_WEEKS ?= 1
MAX_PLAYS ?=
SMOKE_MAX_PLAYS ?= 200
MIN_THROWS ?= 1
GRAPH_EPOCHS ?= 20
GRAPH_BATCH_SIZE ?= 64
GRAPH_LEARNING_RATE ?= 0.0005
GRAPH_PATIENCE ?= 5
GRAPH_TASK ?= all
INTERCEPTION_POS_WEIGHT_CAP ?= 10.0
TEST_WEEKS ?=
ACCEPTABLE_YARDS_GAP ?= 1.0
ACCEPTABLE_YARDS_PCT ?= 0.20
ACCEPTABLE_EPA_GAP ?= 0.25

.PHONY: help install smoke analyze-smoke expected-yards analyze graph-smoke graph-train graph-infer graph-decisions compare-errors visuals clean-smoke

help:
	@echo "QB decision making expected-yards workflow"
	@echo ""
	@echo "Targets:"
	@echo "  install          Install Python dependencies"
	@echo "  smoke            Run a capped expected-yards pipeline for quick validation"
	@echo "  analyze-smoke    Build analysis tables for the smoke output"
	@echo "  expected-yards   Run the expected-yards pipeline"
	@echo "  analyze          Build analysis tables for OUTPUT_DIR"
	@echo "  graph-smoke      Train a small graph transformer prototype"
	@echo "  graph-train      Train the graph transformer on WEEKS"
	@echo "  graph-infer      Run graph transformer inference from GRAPH_OUTPUT_DIR"
	@echo "  graph-decisions  Make graph decision summary"
	@echo "  compare-errors   Create a tree-vs-graph model error table and PNG"
	@echo "  visuals          Make project graphics"
	@echo "  clean-smoke      Remove smoke-test outputs"
	@echo ""
	@echo "Common variables:"
	@echo "  DATA_DIR=$(DATA_DIR)"
	@echo "  OUTPUT_DIR=$(OUTPUT_DIR)"
	@echo "  WEEKS=$(WEEKS)"
	@echo "  MAX_PLAYS=$(MAX_PLAYS)"

install:
	$(PYTHON) -m pip install -r requirements.txt

smoke:
	$(PYTHON) expected_receiver_yards.py \
		--data-dir "$(DATA_DIR)" \
		--output-dir "$(SMOKE_OUTPUT_DIR)" \
		--weeks $(SMOKE_WEEKS) \
		--max-plays $(SMOKE_MAX_PLAYS) \
		--acceptable-yards-gap $(ACCEPTABLE_YARDS_GAP) \
		--acceptable-yards-pct $(ACCEPTABLE_YARDS_PCT) \
		--acceptable-epa-gap $(ACCEPTABLE_EPA_GAP)

analyze-smoke:
	$(PYTHON) analyze_expected_yards.py \
		--data-dir "$(DATA_DIR)" \
		--output-dir "$(SMOKE_OUTPUT_DIR)" \
		--min-throws 1

expected-yards:
	$(PYTHON) expected_receiver_yards.py \
		--data-dir "$(DATA_DIR)" \
		--output-dir "$(OUTPUT_DIR)" \
		--weeks $(WEEKS) \
		$(if $(MAX_PLAYS),--max-plays $(MAX_PLAYS),) \
		$(if $(TEST_WEEKS),--test-weeks $(TEST_WEEKS),) \
		--acceptable-yards-gap $(ACCEPTABLE_YARDS_GAP) \
		--acceptable-yards-pct $(ACCEPTABLE_YARDS_PCT) \
		--acceptable-epa-gap $(ACCEPTABLE_EPA_GAP)

analyze:
	$(PYTHON) analyze_expected_yards.py \
		--data-dir "$(DATA_DIR)" \
		--output-dir "$(OUTPUT_DIR)" \
		--min-throws $(MIN_THROWS)

graph-smoke:
	$(PYTHON) spatiotemporal_graph_transformer.py \
		--data-dir "$(DATA_DIR)" \
		--output-dir "$(GRAPH_SMOKE_OUTPUT_DIR)" \
		--weeks $(SMOKE_WEEKS) \
		--max-plays $(SMOKE_MAX_PLAYS) \
		--epochs $(GRAPH_EPOCHS) \
		--batch-size $(GRAPH_BATCH_SIZE) \
		--learning-rate $(GRAPH_LEARNING_RATE) \
		--early-stop-patience $(GRAPH_PATIENCE) \
		--interception-pos-weight-cap $(INTERCEPTION_POS_WEIGHT_CAP) \
		--prediction-task $(GRAPH_TASK)

graph-train:
	$(PYTHON) spatiotemporal_graph_transformer.py \
		--data-dir "$(DATA_DIR)" \
		--output-dir "$(GRAPH_OUTPUT_DIR)" \
		--weeks $(WEEKS) \
		$(if $(MAX_PLAYS),--max-plays $(MAX_PLAYS),) \
		--epochs $(GRAPH_EPOCHS) \
		--batch-size $(GRAPH_BATCH_SIZE) \
		--learning-rate $(GRAPH_LEARNING_RATE) \
		--early-stop-patience $(GRAPH_PATIENCE) \
		--interception-pos-weight-cap $(INTERCEPTION_POS_WEIGHT_CAP) \
		--prediction-task $(GRAPH_TASK)

graph-infer:
	$(PYTHON) spatiotemporal_graph_transformer.py \
		--mode infer \
		--data-dir "$(DATA_DIR)" \
		--output-dir "$(GRAPH_OUTPUT_DIR)" \
		--weeks $(WEEKS) \
		$(if $(MAX_PLAYS),--max-plays $(MAX_PLAYS),)

graph-decisions:
	$(PYTHON) graph_decision_summary.py \
		--graph-dir "$(GRAPH_OUTPUT_DIR)" \
		--acceptable-yards-gap $(ACCEPTABLE_YARDS_GAP) \
		--acceptable-yards-pct $(ACCEPTABLE_YARDS_PCT)

compare-errors:
	$(PYTHON) compare_model_errors.py \
		--expected-yards-dir "$(OUTPUT_DIR)" \
		--graph-dir "$(GRAPH_OUTPUT_DIR)"

visuals:
	$(PYTHON) make_visuals.py \
		--output-dir "$(OUTPUT_DIR)"

clean-smoke:
	rm -rf "$(SMOKE_OUTPUT_DIR)" "$(GRAPH_SMOKE_OUTPUT_DIR)"
