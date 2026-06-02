.PHONY: docs docs-serve docs-deploy docker docker-cpu docker-gpu test-docker deploy setup-hooks clean-kubernetes work-pool

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

# Build docs, copy into a gh-pages worktree in /tmp, commit and push.
# Uses git worktree so you never leave your current branch.
GHPAGES_DIR := $(shell mktemp -d)/gh-pages
docs-deploy: docs
	git worktree add $(GHPAGES_DIR) gh-pages
	rm -rf $(GHPAGES_DIR)/*
	cp -r docs/* $(GHPAGES_DIR)/
	cd $(GHPAGES_DIR) && \
		git add --all && \
		git diff --cached --quiet || \
		(git commit --no-verify -m "Update documentation" && git push origin gh-pages)
	git worktree remove --force $(GHPAGES_DIR)

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
