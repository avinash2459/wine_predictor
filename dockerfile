#!/bin/bash
#!/usr/bin/env python
FROM python:latest

RUN mkdir /wine_predictor_Docker

WORKDIR /wine_predictor_Docker

COPY requirements.txt .
RUN pip install -r requirements.txt 

COPY savedData .
COPY ValidationDataset.csv .
COPY runmodel.py .

CMD python runmodel.py
