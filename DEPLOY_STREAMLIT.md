# Deploy to Streamlit Community Cloud

This project is ready to deploy as a Streamlit app.

## Required files

- `app.py` is the Streamlit entrypoint.
- `requirements.txt` contains the Python dependencies.
- `.streamlit/config.toml` contains visual theme settings.

## Steps

1. Create a new GitHub repository, for example `RIS_VLC_Simulation`.
2. Upload/push this project folder to that repository.
3. Open Streamlit Community Cloud: https://share.streamlit.io/
4. Click `Create app`.
5. Select:
   - Repository: your GitHub repo
   - Branch: `main`
   - Main file path: `app.py`
6. Open `Advanced settings` and choose Python `3.12` or `3.11`.
7. Click `Deploy`.

After deployment, Streamlit will provide a public URL like:

```text
https://your-app-name.streamlit.app
```

You can share that URL with anyone.

## Notes

- Streamlit Community Cloud uses a `*.streamlit.app` URL.
- You can choose a memorable Streamlit subdomain during deployment.
- For a fully custom domain such as `example.com`, deploy to another host or VPS instead.
