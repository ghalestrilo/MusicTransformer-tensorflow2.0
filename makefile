server-transformer:
	pyenv local 3.7.4; \
	python server.py

server-remi:
	pyenv local 3.6.12; \
	python server.py

client:
	osc-repl server.yaml

all: server