#TODO ek olarak su an komutlar/status string geliyo onu enum yap ki sadece belirli seyler yapabilsinler

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional
import time
from contextlib import asynccontextmanager, suppress
from enum import Enum # Enum import edildi

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Robot kontrolcüsü ve anlık pozisyon için importlar
from main_func import IntegratedRobotController
import numpy as np

# --- Global State ---
# Robot kontrolcüsü nesnesi, uygulama başlangıcında oluşturulacak
robot_controller: Optional[IntegratedRobotController] = None
# Robotun mevcut fiziksel pozisyonunu (X, Y, Z) saklayan değişken
current_position: np.ndarray = np.array([38.0, 0.0, 0.0])
# Gripper'ın mevcut açısını saklayan değişken
current_gripper_angle: float = 35.0
# --------------------

# Robot komutları için Enum
class RobotCommand(str, Enum): # str eklenerek OpenAPI uyumluluğu sağlandı
    X_UP = "xup"
    X_DOWN = "xdown"
    Y_UP = "yup"
    Y_DOWN = "ydown"
    Z_UP = "zup"
    Z_DOWN = "zdown"
    GRIP_OPEN = "gripopen"
    GRIP_CLOSE = "gripclose"
    # Buraya başka komutlar eklenebilir

# Görev durumları için Enum
class TaskStatus(str, Enum): # str eklenerek OpenAPI uyumluluğu sağlandı
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# Bir taskin neler icerebilecegi hakkinda yazdim sen de devam edersin artik

@dataclass
class RobotTask:
    id: str
    command: str
    payload: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)
    # Simdilik ornekler birakiyorum burayi da enum yap
    status: TaskStatus = TaskStatus.QUEUED   # queued | running | completed | failed  
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    done_event: asyncio.Event = field(default_factory=asyncio.Event)


# request / response 

class CommandRequest(BaseModel):
    command: RobotCommand
    payload: Dict[str, Any] | None = None


class EnqueueResponse(BaseModel):
    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    created_at: datetime
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class QueueInfoResponse(BaseModel):
    queue_length: int
    total_tasks: int
    running: Optional[str] = None  # su anda calisan görevin job_idsi

    queued_count: int
    running_count: int
    completed_count: int
    failed_count: int

    queued_ids: list[str] = []


# ROBOT MAIN LOGIC

async def run_robot_command(command: RobotCommand, payload: Dict[str, Any]) -> Dict[str, Any]:
    global current_position, robot_controller, current_gripper_angle
    if robot_controller is None:
        raise RuntimeError("Robot controller is not initialized.")

    print(f"🏃‍♀️ Komut alındı: {command.value}")

    # Gripper komutları
    if command == RobotCommand.GRIP_OPEN:
        current_gripper_angle += 1.0
        current_gripper_angle = min(current_gripper_angle, 180.0)
        robot_controller.move_servos({8: current_gripper_angle}, smooth=False)
        print(f"✅ Gripper açıldı. Yeni açı: {current_gripper_angle}°")
        return {
            "status": "ok",
            "command": command.value,
            "new_angle": current_gripper_angle,
        }
    elif command == RobotCommand.GRIP_CLOSE:
        current_gripper_angle -= 1.0
        current_gripper_angle = max(current_gripper_angle, 0.0)
        robot_controller.move_servos({8: current_gripper_angle}, smooth=False)
        print(f"✅ Gripper kapatıldı. Yeni açı: {current_gripper_angle}°")
        return {
            "status": "ok",
            "command": command.value,
            "new_angle": current_gripper_angle,
        }

    # Pozisyon komutları
    step = 1.0  # 1 cm'lik adımlar
    new_position = current_position.copy()

    if command == RobotCommand.X_UP:
        new_position[0] += step
    elif command == RobotCommand.X_DOWN:
        new_position[0] -= step
    elif command == RobotCommand.Y_UP:
        new_position[1] += step
    elif command == RobotCommand.Y_DOWN:
        new_position[1] -= step
    elif command == RobotCommand.Z_UP:
        new_position[2] += step
    elif command == RobotCommand.Z_DOWN:
        new_position[2] -= step
    
    print(f"🎯 Hedef pozisyon hesaplandı: {new_position.tolist()}")

    # Bilek orientasyonunu ayarla
    orientation = None
    if robot_controller.settings.get('use_orientation'):
        orient_setting = robot_controller.settings.get('default_orientation', 'horizontal')
        if orient_setting == 'vertical':
            orientation = [0, 180, 0]
        else:
            orientation = [0, 90, 0]

    # Robotu yeni pozisyona hareket ettir
    success, _ = robot_controller.move_to_position(
        x=new_position[0],
        y=new_position[1],
        z=new_position[2],
        orientation=orientation,
        path_type=robot_controller.settings.get('movement_type', 'joint')
    )

    if success:
        # Global pozisyonu güncelle
        current_position = new_position
        print(f"✅ Hareket başarılı. Yeni pozisyon: {current_position.tolist()}")
        return {
            "status": "ok",
            "command": command.value,
            "new_position": current_position.tolist(),
            "info": f"Robot başarıyla {new_position.tolist()} konumuna hareket etti.",
        }
    else:
        print(f"❌ Hareket başarısız. Robot {current_position.tolist()} pozisyonunda kaldı.")
        raise RuntimeError(f"Inverse kinematics çözümü bulunamadı veya hareket tamamlanamadı. Hedef: {new_position.tolist()}")


# Tek worker cunku fifo yapmak istiyorz queue istiyorz tek worker olcak
async def worker(queue: "asyncio.Queue[RobotTask]", tasks: Dict[str, RobotTask]):
    while True:
        task = await queue.get()
        task.status = TaskStatus.RUNNING

        try:
            task.result = await run_robot_command(task.command, task.payload)
            task.status = TaskStatus.COMPLETED
        except Exception as e:
            task.error = str(e)
            task.status = TaskStatus.FAILED
        finally:
            task.done_event.set()
            queue.task_done()

#Worker calisma bicimi
@asynccontextmanager
async def lifespan(app: FastAPI):
    global robot_controller, current_position, current_gripper_angle

    # --- startup kısmı ---
    print("🤖 Robot kontrolcüsü başlatılıyor...")
    robot_controller = IntegratedRobotController()

    print("\n" + "="*60)
    print("🚀 UYGULAMA BAŞLATILIYOR 🚀")
    print(f"✅ Varsayılan ayarlar yüklendi. Robot başlangıç pozisyonuna gidiyor...")
    print(f"🎯 Hedef: {current_position.tolist()}")
    
    # Bilek orientasyonunu ayarla ve kullanıcıyı bilgilendir
    orientation = None
    if robot_controller.settings.get('use_orientation'):
        orient_setting = robot_controller.settings.get('default_orientation', 'horizontal')
        if orient_setting == 'vertical':
            orientation = [0, 180, 0]
            print("🦾 Bilek orientasyonu KORUNACAK: Dikey (0, 180, 0)")
        else:
            orientation = [0, 90, 0]
            print("🦾 Bilek orientasyonu KORUNACAK: Yatay (0, 90, 0)")
    else:
        print("🦾 Bilek orientasyonu SERBEST.")
    print("="*60)
    
    # Robotu başlangıç pozisyonuna götür
    robot_controller.move_to_position(
        x=current_position[0],
        y=current_position[1],
        z=current_position[2],
        orientation=orientation,
        path_type=robot_controller.settings.get('movement_type', 'joint'),
        smooth=robot_controller.settings.get('smooth_movement', False)
    )
    print("✅ Robot başlangıç pozisyonunda.")

    # Gripper'ı başlangıç pozisyonuna ayarla
    print(f"🦾 Gripper başlangıç açısına ayarlanıyor: {current_gripper_angle}°")
    robot_controller.move_servos({8: current_gripper_angle}, smooth=False)
    print("✅ Gripper ayarlandı.")

    # FastAPI görev kuyruğunu ve worker'ı başlat
    app.state.task_queue: asyncio.Queue[RobotTask] = asyncio.Queue()
    app.state.tasks: Dict[str, RobotTask] = {}

    app.state.worker_task = asyncio.create_task(
        worker(app.state.task_queue, app.state.tasks)
    )

    try:
        # Uygulama çalışırken burası bloklanır
        yield
    finally:
        # --- shutdown kısmı ---
        print("\n👋 Sunucu kapatılıyor... Robot ana pozisyonuna dönüyor...")
        if robot_controller:
            robot_controller.home_position() # Robotu güvenli bir pozisyona al
        
        app.state.worker_task.cancel()
        with suppress(asyncio.CancelledError):
            await app.state.worker_task
        print("✅ Sunucu kapatıldı.")


app = FastAPI(
    title="Robot Kol Sunucusu",
    lifespan=lifespan,
)


# API endpointler

@app.post("/commands", response_model=EnqueueResponse)
async def enqueue_command(cmd: CommandRequest):
    job_id = str(uuid.uuid4())
    task = RobotTask(
        id=job_id,
        command=cmd.command,
        payload=cmd.payload or {},
    )

    app.state.tasks[job_id] = task
    await app.state.task_queue.put(task)

    return EnqueueResponse(job_id=job_id, status=task.status)


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    task: RobotTask | None = app.state.tasks.get(job_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobStatusResponse(
        job_id=task.id,
        status=task.status,
        created_at=task.created_at,
        result=task.result,
        error=task.error,
    )


@app.get("/queue", response_model=QueueInfoResponse)
async def get_queue_info():
    q: asyncio.Queue = app.state.task_queue

    running_id: Optional[str] = None
    queued_count = 0
    running_count = 0
    completed_count = 0
    failed_count = 0
    queued_ids: list[str] = []

    for t in app.state.tasks.values():
        if t.status == TaskStatus.QUEUED:
            queued_count += 1
            queued_ids.append(t.id)
        elif t.status == TaskStatus.RUNNING:
            running_count += 1
            if running_id is None:
                running_id = t.id
        elif t.status == TaskStatus.COMPLETED:
            completed_count += 1
        elif t.status == TaskStatus.FAILED:
            failed_count += 1

    return QueueInfoResponse(
        queue_length=q.qsize(),
        total_tasks=len(app.state.tasks),
        running=running_id,
        queued_count=queued_count,
        running_count=running_count,
        completed_count=completed_count,
        failed_count=failed_count,
        queued_ids=queued_ids,
    )

