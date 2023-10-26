echo "Running flake8..."
flake8 --per-file-ignores="__init__.py:F401" --max-line-length 120 mlutils/

echo "Running tests with coverage..."
coverage run -m --source=.  pytest  --verbose tests/

echo "Producing coverage report..."
coverage report -m > coverage_report.txt
echo "Done..."


