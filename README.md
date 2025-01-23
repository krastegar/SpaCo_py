# PythonProject

# Setup for developement:

- Setup a python 3.x venv (usually in `.venv`)
- pip-compile is buggy for newer versions of python use docker to generate requirements.txt and requirements.dev.txt
- need to have your requirements.in and requirements.dev.in in appropriate directory
` sudo docker run --rm -v $(pwd):/app -w /app python:3.8-slim     /bin/bash -c "pip install pip-tools && pip-compile --output-file=requirements.txt requirements.in && pip-compile --output-file=requirements.dev.txt requirements.dev.in" `

- Install dev requirements: 
` sudo docker run --rm -v $(pwd):/app -w /app python:3.8-slim     /bin/bash -c " pip install -r requirements.txt && pip install -r requirements.txt" `



## Update versions

`pip-compile --output-file=requirements.dev.txt requirements.dev.in --upgrade`
`pip-compile --output-file=requirements.txt requirements.in --upgrade`

# Run `pre-commit` locally.

`pre-commit run --all-files`
