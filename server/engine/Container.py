"""Engine containers module."""
from dependency_injector import containers, providers

from server.engine.detection.Engine import DetectionEngine
from server.engine.detection.WorkerManager import DetectionWorkerManager
from server.engine.SR.Engine import SREngine
from server.engine.SR.WorkerManager import SRWorkerManager


class EngineContainer(containers.DeclarativeContainer):

    setting = providers.Dependency()

    sr_worker_manager = providers.Singleton(SRWorkerManager, setting=setting)

    sr_engine = providers.Factory(
        SREngine, setting=setting, worker_manager=sr_worker_manager
    )

    detection_worker_manager = providers.Singleton(
        DetectionWorkerManager, setting=setting
    )

    detection_engine = providers.Factory(
        DetectionEngine, setting=setting, worker_manager=detection_worker_manager
    )
