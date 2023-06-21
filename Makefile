SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c
MAKEFLAGS = --warn-undefined-variables

.PHONY: nvitop generate train

requirements.txt: requirements.in
	.venv/bin/python3.10 -m piptools compile \
		--pip-args="-f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html" \
		--resolver=backtracking

.venv: requirements.txt
	python3.10 -m venv .venv
	.venv/bin/python3.10 -m pip install --upgrade pip setuptools wheel build
	.venv/bin/python3.10 -m pip install pip-tools
	.venv/bin/python3.10 -m piptools sync
	touch .venv

nvitop: .venv
	.venv/bin/python3.10 -m nvitop

generate: .venv
	.venv/bin/python3.10 -m gpt2 "Alan Turing theorized that computers would one day become"

train: .venv
	.venv/bin/python3.10 -m train
