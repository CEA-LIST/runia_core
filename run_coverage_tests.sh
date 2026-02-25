#!/bin/bash
# Run the tests with coverage
coverage run --source=runia_core/ -m unittest discover tests "*test*.py"
# Generate the coverage report
coverage report
# Optionally, generate an HTML report
coverage html -d tools/coverage/html
coverage xml -o tools/coverage/coverage.xml
# Generate badge
genbadge coverage -i tools/coverage/coverage.xml -o tools/coverage/coverage-badge.svg