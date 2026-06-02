.PHONY: docs docs-serve docker docker-cpu docker-gpu test-docker deploy setup-hooks clean-kubernetes work-pool

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

REGISTRY := $(DOCKER_REGISTRY)
IMAGE_GPU := $(DOCKER_IMAGE_GPU)

test-docker:
	mkdir -p .buildx-cache-gpu
	docker buildx build -f Dockerfile --target test \
		--cache-from type=local,src=.buildx-cache-gpu \
		--cache-to type=local,dest=.buildx-cache-gpu,mode=max \
		.

docker-gpu:
	mkdir -p .buildx-cache-gpu
	docker buildx build --platform linux/amd64 -f Dockerfile --target production-gpu \
		--cache-from type=local,src=.buildx-cache-gpu \
		--cache-to type=local,dest=.buildx-cache-gpu,mode=max \
		--push \
		-t $(REGISTRY)/$(IMAGE_GPU):dev .

docker: docker-gpu
	uv lock
