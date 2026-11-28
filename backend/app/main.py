from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import config, errors
from app.routers import health, inspect, models, results

app = FastAPI(title="Argus")
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)
errors.install(app)
for module in (health, models, inspect, results):
    app.include_router(module.router)
