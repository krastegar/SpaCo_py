# PythonProject

# Setup for developement:

- Setup a python 3.x venv (usually in `.venv`)
- pip-compile is buggy for newer versions of python use UV python package manager to generate requirements.txt and requirements.dev.txt
- Download UV: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- need to have your requirements.in and requirements.dev.in in appropriate directory
` uv pip compile ./requirements.in    --universal --output-file ./requirements.txt`
` uv pip compile ./requirements.dev.in --universal --output-file ./requirements.dev.txt`

- Install dev requirements: 
` uv add --dev $(grep -vE '^\s*#|^\s*$|;' requirements.dev.txt) `

- Install requirements:
` uv add $(grep -vE '^\s*#|^\s*$|;' requirements.txt) `


## Update versions

`uv pip compile --output-file=requirements.dev.txt requirements.dev.in --upgrade`
`uv pip compile --output-file=requirements.txt requirements.in --upgrade`

# Run `pre-commit` locally.

`pre-commit run --all-files`
