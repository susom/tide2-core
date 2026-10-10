.PHONY: docs docs-serve docker docker-push setup-hooks

-include .env
export

setup-hooks:
	uv tool install pre-commit
	uv tool install nbstripout
	pre-commit install && pre-commit install --hook-type commit-msg
	nbstripout --install

# Pre-import pyarrow to prevent segfault caused by native library init order
# conflict when presidio_anonymizer triggers a deep import chain:
# presidio_anonymizer → ahds_surrogate → presidio_analyzer → transformers
# → sklearn → pandas → pyarrow (segfault during pyarrow native init)
docs:
	python -c "import pyarrow; import pdoc, pdoc.render, pathlib; \
		pdoc.render.configure(docformat='google'); \
		pdoc.pdoc('tide2', output_directory=pathlib.Path('docs/'))"

docs-serve:
	python -c "import pyarrow; from pdoc.web import DocServer, open_browser; \
		server = DocServer(('localhost', 8080), ['tide2']); \
		open_browser('http://localhost:8080'); \
		server.serve_forever()"

# DOCKER_REGISTRY and DOCKER_IMAGE come from .env (DOCKER_IMAGE_GPU is still read as a fallback).
IMAGE := $(or $(DOCKER_IMAGE),$(DOCKER_IMAGE_GPU),tide2)
TAG ?= dev
IMAGE_REF := $(if $(DOCKER_REGISTRY),$(DOCKER_REGISTRY)/,)$(IMAGE):$(TAG)

docker:
	docker build --platform linux/amd64 -t $(IMAGE_REF) .

docker-push: docker
	docker push $(IMAGE_REF)
