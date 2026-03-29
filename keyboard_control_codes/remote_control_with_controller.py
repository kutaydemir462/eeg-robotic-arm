import asyncio
import math
import time
import numpy as np
from evdev import InputDevice, categorize, ecodes
import json
import os

# Robot kontrolcüsünü projemize dahil ediyoruz
from main_func import IntegratedRobotController

# --- Ana Yapılandırma ---
MOTION_DEVICE_PATH = '/dev/input/event10'
BUTTON_DEVICE_PATH = '/dev/input/event9'
STATE_FILE = "robot_state.json"  # Pozisyonu kaydetmek için dosya adı
ALPHA = 0.98

# Hareket Kontrol Yapılandırması
Y_PITCH_UP_THRESHOLD, Y_PITCH_DOWN_THRESHOLD = 30.0, -30.0
X_ROLL_UP_THRESHOLD, X_ROLL_DOWN_THRESHOLD = 50.0, 140.0
Y_ACTION_COOLDOWN, X_ACTION_COOLDOWN, Z_ACTION_COOLDOWN = 1.0, 1.0, 1.0
GRIPPER_ACTION_COOLDOWN = 0.2
STEP_SIZE_CM = 1.0

# --- Global Durum Değişkenleri ---
robot_controller: IntegratedRobotController | None = None
# Varsayılan başlangıç değerleri, eğer kayıt dosyası yoksa kullanılır
target_position = np.array([38.0, 0.0, 0.0])
target_gripper_angle = 35.0

robot_lock = asyncio.Lock()
pitch, roll = 0.0, 0.0
y_tilt_state, x_tilt_state = 'neutral', 'neutral'
last_y_action_time, last_x_action_time, last_z_action_time, last_gripper_action_time = 0, 0, 0, 0
setting_button_press_time = None # Oryantasyon butonu için basılma zamanı

EVENT_MAP = {
    'ABS_X': 'accel_x', 'ABS_Y': 'accel_y', 'ABS_Z': 'accel_z',
    'ABS_RX': 'gyro_x', 'ABS_RY': 'gyro_y', 'ABS_RZ': 'gyro_z',
}
sensor_state = {key: 0 for key in EVENT_MAP.values()}

# --- Durum Kaydetme/Yükleme Fonksiyonları ---

def save_state():
    """Mevcut hedef pozisyonu ve gripper açısını dosyaya kaydeder."""
    state = {
        'target_position': target_position.tolist(),
        'target_gripper_angle': target_gripper_angle
    }
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=4)
    print(f"\n💾 Durum kaydedildi: {STATE_FILE}")

def load_state():
    """Kaydedilmiş hedef pozisyonu ve gripper açısını dosyadan yükler."""
    global target_position, target_gripper_angle
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r') as f:
                state = json.load(f)
                target_position = np.array(state['target_position'])
                target_gripper_angle = state['target_gripper_angle']
                print(f"✅ Kaydedilmiş durum yüklendi: Pozisyon={target_position.tolist()}, Gripper={target_gripper_angle}°")
        except (json.JSONDecodeError, KeyError):
            print(f"⚠️ Kayıt dosyası ({STATE_FILE}) bozuk, varsayılan değerler kullanılıyor.")
    else:
        print("ℹ️ Kayıt dosyası bulunamadı, varsayılan değerler kullanılıyor.")

# --- Robot Komut Fonksiyonları ---

def get_orientation_from_settings():
    """Ayarları okuyarak hedef oryantasyonu belirler."""
    orientation = None
    if robot_controller and robot_controller.settings.get('use_orientation'):
        orient_setting = robot_controller.settings.get('default_orientation', 'horizontal')
        orientation = [0, 135, 0] if orient_setting == 'vertical' else [0, 90, 0]
    return orientation

async def move_robot_step(axis: str, direction: int):
    """Robotu belirli bir eksende bir adım hareket ettirir."""
    global target_position
    async with robot_lock:
        print(f"\nKomut: Eksen={axis}, Yön={direction}")
        new_position = target_position.copy()
        
        if axis == 'x': new_position[0] += STEP_SIZE_CM * direction
        elif axis == 'y': new_position[1] += STEP_SIZE_CM * direction
        elif axis == 'z': new_position[2] += STEP_SIZE_CM * direction
        
        orientation = get_orientation_from_settings()
        orient_str = f"Oryantasyon: {orientation}" if orientation else "Oryantasyon: Serbest"
        print(f"Yeni hedef: {new_position.tolist()}, {orient_str}")

        success, _ = robot_controller.move_to_position(
            x=new_position[0], y=new_position[1], z=new_position[2],
            orientation=orientation, path_type='linear',
        )
        if success:
            target_position = new_position
            print(f"✅ Hareket başarılı. Mevcut pozisyon: {target_position.tolist()}")
        else:
            print(f"❌ Hareket başarısız. Pozisyon: {target_position.tolist()}")

async def move_gripper_step(direction: int):
    """Gripper'ı bir adım hareket ettirir."""
    global target_gripper_angle
    async with robot_lock:
        print(f"\nKomut: Gripper, Yön={direction}")
        gripper_step = 1.0  # Adım boyutu 1'e düşürüldü
        target_gripper_angle += gripper_step * direction
        target_gripper_angle = np.clip(target_gripper_angle, 0, 180)
        
        print(f"Yeni gripper hedefi: {target_gripper_angle:.1f}°")
        robot_controller.move_servos({8: target_gripper_angle}, smooth=False)
        print("✅ Gripper komutu gönderildi.")

async def toggle_orientation_and_reposition():
    """Oryantasyon ayarını değiştirir ve robotu yeni oryantasyona taşır."""
    global target_position
    if not robot_controller:
        return

    async with robot_lock:
        current_orientation_mode = robot_controller.settings.get('default_orientation', 'horizontal')
        
        if current_orientation_mode == 'horizontal':
            new_orientation_mode = 'vertical'  # Corresponds to 135 degrees
            print("\n📐 Oryantasyon 'Dikey' (135°) olarak değiştirildi.")
        else:
            new_orientation_mode = 'horizontal'
            print("\n📐 Oryantasyon 'Yatay' olarak değiştirildi.")
            
        robot_controller.settings['default_orientation'] = new_orientation_mode
        
        # Get the new orientation values from the settings
        new_orientation = get_orientation_from_settings()
        
        print(f"🤖 Yeni oryantasyona geçmek için mevcut pozisyona ({target_position.tolist()}) hareket ediliyor...")
        
        # Re-move to the current target position with the new orientation
        success, _ = robot_controller.move_to_position(
            x=target_position[0], y=target_position[1], z=target_position[2],
            orientation=new_orientation, path_type='linear',
        )

        if success:
            print("✅ Oryantasyon değiştirme başarılı.")
        else:
            print("❌ Oryantasyon değiştirme sırasında hareket başarısız oldu.")

# --- Kontrolcü Olay İşleyicileri (Değişiklik yok) ---

async def process_button_events(device):
    global last_z_action_time, last_gripper_action_time, setting_button_press_time

    # TODO: Kullanıcıdan doğru düğme kodunu alıp buraya yazın. Şimdilik BTN_START kullanılıyor.
    SETTING_BUTTON = ecodes.BTN_START

    async for event in device.async_read_loop():
        if event.type != ecodes.EV_KEY:
            continue

        # Oryantasyon değiştirme butonu için basılı tutma mantığı
        if event.code == SETTING_BUTTON:
            if event.value == 1:  # Düğmeye basıldı
                setting_button_press_time = time.time()
            elif event.value == 0:  # Düğme bırakıldı
                if setting_button_press_time is not None:
                    hold_duration = time.time() - setting_button_press_time
                    if hold_duration > 1.0:
                        asyncio.create_task(toggle_orientation_and_reposition())
                setting_button_press_time = None
            continue  # Bu butonu diğer eylemler için kullanma

        # Diğer butonlar için sadece basılma anını işle
        if event.value == 1:
            current_time = time.time()
            # Z Ekseni Hareketi
            if current_time - last_z_action_time > Z_ACTION_COOLDOWN:
                if event.code == ecodes.BTN_NORTH:
                    asyncio.create_task(move_robot_step('z', 1))
                    last_z_action_time = current_time
                elif event.code == ecodes.BTN_SOUTH:
                    asyncio.create_task(move_robot_step('z', -1))
                    last_z_action_time = current_time
            
            # Gripper Hareketi
            if current_time - last_gripper_action_time > GRIPPER_ACTION_COOLDOWN:
                if event.code == ecodes.BTN_EAST:
                    asyncio.create_task(move_gripper_step(1))
                    last_gripper_action_time = current_time
                elif event.code == ecodes.BTN_WEST:
                    asyncio.create_task(move_gripper_step(-1))
                    last_gripper_action_time = current_time

async def process_motion_events(device):
    global pitch, roll, y_tilt_state, x_tilt_state, last_y_action_time, last_x_action_time
    last_update_time = time.time()
    def get_pitch_roll(state):
        x, y, z = state['accel_x'], state['accel_y'], state['accel_z']
        roll_rad = math.atan2(y, z)
        pitch_rad = math.atan2(-x, math.sqrt(y*y + z*z))
        return math.degrees(pitch_rad), math.degrees(roll_rad)
    async for event in device.async_read_loop():
        if event.type != ecodes.EV_ABS: continue
        event_code_name = ecodes.bytype[event.type].get(event.code, 'UNKNOWN')
        state_key = EVENT_MAP.get(event_code_name)
        if not state_key: continue
        sensor_state[state_key] = event.value
        current_time = time.time()
        if 'gyro' in state_key:
            dt = current_time - last_update_time
            if dt == 0: continue
            last_update_time = current_time
            accel_pitch, accel_roll = get_pitch_roll(sensor_state)
            GYRO_SCALE = 1 / 1000.0
            gyro_x, gyro_y = sensor_state['gyro_x'] * GYRO_SCALE, sensor_state['gyro_y'] * GYRO_SCALE
            pitch = ALPHA * (pitch + gyro_y * dt) + (1.0 - ALPHA) * accel_pitch
            roll = ALPHA * (roll + gyro_x * dt) + (1.0 - ALPHA) * accel_roll
            if current_time - last_y_action_time > Y_ACTION_COOLDOWN:
                if pitch > Y_PITCH_UP_THRESHOLD and y_tilt_state != 'up':
                    asyncio.create_task(move_robot_step('y', 1)); y_tilt_state = 'up'; last_y_action_time = current_time
                elif pitch < Y_PITCH_DOWN_THRESHOLD and y_tilt_state != 'down':
                    asyncio.create_task(move_robot_step('y', -1)); y_tilt_state = 'down'; last_y_action_time = current_time
            if Y_PITCH_DOWN_THRESHOLD < pitch < Y_PITCH_UP_THRESHOLD: y_tilt_state = 'neutral'
            if current_time - last_x_action_time > X_ACTION_COOLDOWN:
                if roll < X_ROLL_UP_THRESHOLD and x_tilt_state != 'up':
                    asyncio.create_task(move_robot_step('x', 1)); x_tilt_state = 'up'; last_x_action_time = current_time
                elif roll > X_ROLL_DOWN_THRESHOLD and x_tilt_state != 'down':
                    asyncio.create_task(move_robot_step('x', -1)); x_tilt_state = 'down'; last_x_action_time = current_time
            if X_ROLL_UP_THRESHOLD < roll < X_ROLL_DOWN_THRESHOLD: x_tilt_state = 'neutral'

async def print_status():
    """Durumu sürekli olarak ekrana basar."""
    while True:
        pos = target_position; grip = target_gripper_angle
        print(f"Hedef -> X:{pos[0]:.1f} Y:{pos[1]:.1f} Z:{pos[2]:.1f} | Gripper:{grip:.0f}°   (Pitch:{pitch:.1f} Roll:{roll:.1f})", end='\r')
        await asyncio.sleep(0.1)

async def main():
    global robot_controller, last_z_action_time, last_y_action_time, last_x_action_time, last_gripper_action_time
    
    # Başlangıçta durumu yükle
    load_state()
    
    print("--- Entegre Robot Kontrolü Başlatılıyor ---")
    try:
        robot_controller = IntegratedRobotController()
        motion_device = InputDevice(MOTION_DEVICE_PATH)
        button_device = InputDevice(BUTTON_DEVICE_PATH)
        orientation = get_orientation_from_settings()
        
        print(f"\n🤖 Robot, kaydedilmiş veya varsayılan pozisyona gidiyor...")
        robot_controller.move_to_position(
            x=target_position[0], y=target_position[1], z=target_position[2],
            orientation=orientation, path_type='linear'
        )
        robot_controller.move_servos({8: target_gripper_angle}, smooth=False)
        orient_str = f"Oryantasyon: {orientation}" if orientation else "Oryantasyon: Serbest"
        print(f"✅ Robot başlangıçta. Hedef: {target_position.tolist()}, Gripper: {target_gripper_angle}°, {orient_str}")

    except FileNotFoundError as e:
        print(f"\n❌ HATA: Kontrolcü cihazı bulunamadı: {e}")
        return
    except Exception as e:
        print(f"\n❌ HATA: Başlatma sırasında bir sorun oluştu: {e}")
        return
        
    print("\n🎮 Kontrolcü ile robotu yönetebilirsiniz. Çıkmak ve durumu kaydetmek için Ctrl+C.")
    now = time.time()
    last_z_action_time, last_y_action_time, last_x_action_time, last_gripper_action_time = now, now, now, now
    await asyncio.gather(
        process_motion_events(motion_device),
        process_button_events(button_device),
        print_status()
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        save_state() # Program kapatılırken durumu kaydet
        print("\n\n👋 Program sonlandırılıyor...")
    except Exception as e:
        print(f"\n❌ Beklenmedik bir hata oluştu: {e}")
        save_state() # Hata durumunda da kaydetmeyi dene
