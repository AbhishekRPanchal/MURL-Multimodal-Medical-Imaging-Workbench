if __name__ == "__main__":
    import uvicorn
    from api.app import app
    print("Starting MURL server on 127.0.0.1:8000...", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
