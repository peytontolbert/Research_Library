# Simple make targets for running experiments and preprocessing corpora.
# Usage (from repo root or models/):
#   make -C models train.M1
#   make -C models eval.M1
#   make -C models preprocess.pdfs
#   make -C models preprocess.repos
# Or inside models/: make train.M1

EXP_DIR := experiments
PYTHONPATH := $(abspath ..)

.PHONY: $(patsubst $(EXP_DIR)/%.json,train.%,$(wildcard $(EXP_DIR)/*.json))
.PHONY: $(patsubst $(EXP_DIR)/%.json,eval.%,$(wildcard $(EXP_DIR)/*.json))

train.%:
	@file=$(firstword $(wildcard $(EXP_DIR)/$(shell echo $* | tr 'A-Z' 'a-z')*.json)); \
	if [ -z "$$file" ]; then echo "No experiment for $*"; exit 1; fi; \
	PYTHONPATH=$(PYTHONPATH) python -m models.cli --experiment $$file --mode train

eval.%:
	@file=$(firstword $(wildcard $(EXP_DIR)/$(shell echo $* | tr 'A-Z' 'a-z')*.json)); \
	if [ -z "$$file" ]; then echo "No experiment for $*"; exit 1; fi; \
	PYTHONPATH=$(PYTHONPATH) python -m models.cli --experiment $$file --mode eval

.PHONY: preprocess.pdfs preprocess.repos
preprocess.pdfs:
	# Set years/limits via env or edit below; max-files 0 means all PDFs.
	PYTHONPATH=$(PYTHONPATH) python -m models.scripts.preprocess_pdfs --max-files 0

preprocess.repos:
	PYTHONPATH=$(PYTHONPATH) python -m models.scripts.preprocess_repos --max-files 5000

.PHONY: preprocess.align
preprocess.align:
	PYTHONPATH=$(PYTHONPATH) python -m models.scripts.preprocess_alignment --max-papers 5000 --max-repos 5000 --top-k 5000 --negatives 5000

.PHONY: preprocess.concepts
preprocess.concepts:
	PYTHONPATH=$(PYTHONPATH) python -m models.scripts.extract_concepts --limit-repos 0 --out exports/repo_concepts.jsonl

.PHONY: preprocess.graphs
preprocess.graphs:
	PYTHONPATH=$(PYTHONPATH) python -m scripts.library_repo_graph_export --library-root /data/repositories --export-root /data/repository_library/exports

.PHONY: build.corpus
build.corpus:
	# Build unified corpora from repos, structured PDFs, and alignment pairs.
	PYTHONPATH=$(PYTHONPATH) python -m models.scripts.build_corpus --repos-chunks-dir exports/repos_chunks --pdfs-structured-dir exports/pdfs_structured --alignment-path exports/paper_repo_align.jsonl --out-dir exports/corpus
