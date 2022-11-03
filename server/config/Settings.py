import logging
import re

import pytz
import torch
from loguru import logger
from prettytable import PrettyTable
from pydantic import BaseSettings, Field, validator


class LogSettings(BaseSettings):
    level: str = Field(default="INFO", env="LOG_LEVEL")

    @validator("level")
    def validate_level(value: str):
        _value = value.strip().upper() if type(value) == str else None
        if not _value in logging._nameToLevel.keys():
            raise ValueError(f"Invalid log level: {value}")
        return _value


class ServerSettings(BaseSettings):
    http_port: int = Field(default=80, env="SERVER_HTTP_PORT")

    base_path: str = Field(default="/", env="SERVER_BASE_PATH")

    listen_host: str = Field(default="0.0.0.0", env="SERVER_LISTEN_HOST")

    devices: str = Field(default=None, env="SERVER_ENGINE_DEVICES")

    @validator("devices")
    def validate_devices(value: str):
        raw = str(value).strip().lower().replace("cuda:", "").replace("gpu:", "")
        if not raw:
            raise ValueError(f"No engine device specified")

        pat1 = re.compile("^[0-9]+$")
        pat2 = re.compile("^[0-9]+-[0-9]+$")
        pat3 = re.compile("^cpu$")

        gpu_total = torch.cuda.device_count()
        avail = [x for x in range(gpu_total)]
        avail.append("cpu")

        ans = set()
        values = raw.split(",")
        for v in values:
            if pat1.match(v):
                ans.add(int(v))
            elif pat2.match(v):
                t = [int(x) for x in v.split("-")]
                for i in range(t[0], t[1] + 1):
                    ans.add(i)
            elif pat3.match(v):
                ans.add(v)
            else:
                raise ValueError(f"Invalid device id: {v}")

        ret = []
        for v in ans:
            if v in avail:
                ret.append(v)
            else:
                raise ValueError(f"Invalid device id: {v}")
        return ret

    worker_num: int = Field(default=1, env="SERVER_WORKER_NUM")
    worker_queue_size: int = Field(default=-1, env="SERVER_WORKER_QUEUE_SIZE")

    worker_batch_size: int = Field(default=32, env="SERVER_WORKER_BATCH_SIZE")

    timezone: str = Field(default="UTC", env="SERVER_TIMEZONE")

    @validator("timezone")
    def validate_timezone(value: str):
        _value = value.strip().upper() if type(value) == str else None

        for tz in pytz.all_timezones_set:
            if tz.upper() == _value:
                return tz

        raise ValueError(f"Invalid server timezone: {value}")


class Settings(BaseSettings):
    log: LogSettings = LogSettings()
    server: ServerSettings = ServerSettings()

    @staticmethod
    def print(settings):
        table = PrettyTable()
        table.field_names = ["Log Variable", "Value"]
        table.align["Value"] = "l"
        for key, value in dict(settings.log).items():
            table.add_row([key, value])
        logger.info("\n{}", table)

        table = PrettyTable()
        table.field_names = ["Server Variable", "Value"]
        table.align["Value"] = "l"
        for key, value in dict(settings.server).items():
            table.add_row([key, value])
        logger.info("\n{}", table)
