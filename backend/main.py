from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="How To Not Kill Your Indoor Plants",
    description="API for indoor plant care tracking",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Welcome to How To Not Kill Your Indoor Plants API"}


@app.get("/health")
async def health():
    return {"status": "healthy"}
