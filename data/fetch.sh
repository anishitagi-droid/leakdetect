#!/usr/bin/env bash
# Download the example datasets from their original public sources and verify them against the checksums of
# the exact files the numbers in the README were measured on. The data is not redistributed in this repo.
set -euo pipefail
cd "$(dirname "$0")"

fetch() { curl -fsSL "$2" -o "$1" && echo "downloaded $1"; }
fetch CreditCard.csv      https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/AER/CreditCard.csv
fetch titanic_seaborn.csv https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv
fetch pd_speech.csv       https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/modeldata/pd_speech.csv
fetch stocks.csv          https://raw.githubusercontent.com/vega/vega-datasets/main/data/stocks.csv
fetch Orthodont.csv       https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/nlme/Orthodont.csv
fetch ChickWeight.csv     https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/ChickWeight.csv
fetch VerbAgg.csv         https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/lme4/VerbAgg.csv

if command -v sha256sum >/dev/null; then sha256sum -c SHA256SUMS; else shasum -a 256 -c SHA256SUMS; fi
