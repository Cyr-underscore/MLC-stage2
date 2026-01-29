# API Python (FastAPI exemple)
from fastapi import FastAPI, Request
from pydantic import BaseModel
import json
import datetime

class MouseData(BaseModel):
    x: int
    y: int
    timestamp: int
    url: str
    viewport: dict
    element: dict

