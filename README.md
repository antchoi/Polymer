<img src="resource/logo.svg" alt="Polymer logo" width="500" />

Polymer, an ready-to-use & multi-purpose AI inference server.

 * [Development documentation](https://www.notion.so/antchoi/Inferencer-Guide-24d62c5f3ad9425e9e128e85fc28052f)


1. Copy .env from .env.template file and Change environment variables if needed:

   ```
   $ cp .env.template .env
   ```

2. Run start script for running the server

   ```
   $ ./scripts/start-server.sh
   ```

   If server is ready, following page will be available:

   - **Swagger UI**: http://[SERVER IP or DOMAIN]:[SERVER HTTP PORT]/docs

   ***

   If you want to stop server, run stop script:

   ```
   $ ./scripts/stop-server.sh
   ```
