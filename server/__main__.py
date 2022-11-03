"""Main module."""
import sys

import uvicorn
from dependency_injector.wiring import Provide, inject
from fastapi import FastAPI
from loguru import logger

from server.application import Application
from server.config.Settings import Settings
from server.controller import router


@inject
def main(
    setting: Settings = Provide[Application.setting],
) -> None:
    Settings.print(setting)

    server = FastAPI(
        root_path=setting.server.base_path, root_path_in_servers=True, debug=True
    )
    server.include_router(router)

    uvicorn.run(
        server,
        host=setting.server.listen_host,
        port=setting.server.http_port,
        access_log=False,
    )


if __name__ == "__main__":
    setting: Settings = Settings()

    logger.remove()
    logger.add(sys.stdout, colorize=True, level=setting.log.level)

    application = Application(setting=setting)
    main()
