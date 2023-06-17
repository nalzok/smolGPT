.venv:
	python3.10 -m venv .venv
	.venv/bin/python3.10 -m pip install --upgrade pip
	.venv/bin/python3.10 -m pip install pip-tools
	.venv/bin/python3.10 -m piptools sync

generate: .venv
	.venv/bin/python3.10 -m gpt2 "Alan Turing theorized that computers would one day become"
