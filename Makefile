SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c
MAKEFLAGS = --warn-undefined-variables

.PHONY: nvitop generate train

.venv:
	python3.10 -m venv .venv
	.venv/bin/python3.10 -m pip install --upgrade pip setuptools wheel build pip-tools
	touch .venv

requirements.txt: .venv requirements.in
	.venv/bin/python3.10 -m piptools compile \
		--pip-args="-f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"
	.venv/bin/python3.10 -m piptools sync

nvitop: requirements.txt
	.venv/bin/python3.10 -m nvitop

generate: requirements.txt
	.venv/bin/python3.10 -m gpt2 "Alan Turing theorized that computers would one day become"

train: requirements.txt
	.venv/bin/python3.10 -m train
