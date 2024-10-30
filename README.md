# Setup

## Python Environment

Either run (this requires Anaconda):

```
conda env create -f environment.yml
```

Or

```
pip install -r requirements.txt
```

## Google Cloud

Follow this guide to setup Google Cloud.

https://cloud.google.com/docs/authentication/provide-credentials-adc#google-idp

## Export Anaconda environment

```
conda env export --no-builds --file environment.yml
```
