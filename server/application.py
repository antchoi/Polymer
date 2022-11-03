"""Containers module."""
from dependency_injector import containers, providers

from server.engine.Container import EngineContainer


class Application(containers.DeclarativeContainer):

    wiring_config = containers.WiringConfiguration(modules=["__main__", ".controller"])

    setting = providers.Dependency()

    engine_container = providers.Container(EngineContainer, setting=setting)
