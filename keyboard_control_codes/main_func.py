import numpy as np
from roboticstoolbox import DHRobot, RevoluteDH
from spatialmath import SE3
import roboticstoolbox as rtb
from adafruit_servokit import ServoKit
import time
import threading
import math

class IntegratedRobotController:
    def __init__(self):
        # Servo kontrolcüsü başlat
        self.kit = ServoKit(channels=16)

        # Servo yapılandırması
        self.servo_ids = [0, 1, 2, 4, 5, 6, 8]
        for ch in self.servo_ids:
            if ch == 1:
                self.kit.servo[ch].set_pulse_width_range(450, 2775)
            elif ch in {0, 2, 4, 6}:
                self.kit.servo[ch].set_pulse_width_range(450, 2825)
            elif ch == 5:
                self.kit.servo[ch].set_pulse_width_range(500, 2800)
            else:
                self.kit.servo[ch].set_pulse_width_range(500, 2700)

            if ch == 5:
                self.kit.servo[ch].actuation_range = 270
            else:
                self.kit.servo[ch].actuation_range = 180

        self.home_offsets = [90, 135, 62, 100, 120.0, 173]
        self.servo_home_angles = {
            6: self.home_offsets[0],
            5: self.home_offsets[1],
            4: self.home_offsets[2],
            2: self.home_offsets[3],
            1: self.home_offsets[4],
            0: self.home_offsets[5]
        }
        self.last_joint_angles = np.array(self.home_offsets, dtype=float)

        # Robot kinematiği için DH parametreleri
        self.robot = self.create_robot()

        # Varsayılan ayarlar
        self.settings = {
            'movement_type': 'linear',
            'smooth_movement': True,
            'step_delay': 0.01,
            'auto_visualize': True,
            'use_orientation': True,
            'default_orientation': 'horizontal',
            'steps_per_cm': 5,
            'x_axis_gain': 1.0,
            'x_axis_offset': 0.0,
            'y_axis_gain': 1.0,
            'y_axis_offset': 0.0,
            'z_axis_gain': 1.0,
            'z_axis_offset': 0.0,
            'x_from_y_coupling': 0.0,
            'y_from_x_coupling': 0.0,
            'xy_axis_rotation_deg': 0.0
        }

        print("🔧 Entegre robot kontrolcüsü hazır!")
        print(f"📋 Kullanılabilir servolar: {self.servo_ids}")
        print("🤖 Kinematik hesaplama sistemi aktif!")
        print("📐 Servo 5: 0-270°, Diğerleri: 0-180°")

    def create_robot(self):
        """DH Tablosuna göre 6-DOF robot modeli oluşturur"""
        links = [
            RevoluteDH(d=0, a=0, alpha=-np.pi/2, offset=0),           # Link 1
            RevoluteDH(d=0, a=23, alpha=0, offset=-np.pi/2),         # Link 2
            RevoluteDH(d=-7, a=0, alpha=-np.pi/2, offset=0),         # Link 3
            RevoluteDH(d=23, a=0, alpha=np.pi/2, offset=0),          # Link 4
            RevoluteDH(d=7, a=0, alpha=np.pi/2, offset=np.pi),       # Link 5 (180° offset)
            RevoluteDH(d=22, a=0, alpha=0, offset=0)                 # Link 6
        ]

        servo_ranges = [
            (0, 180),   # θ1
            (0, 270),   # θ2
            (0, 180),   # θ3
            (0, 180),   # θ4
            (0, 180),   # θ5
            (0, 180)    # θ6
        ]

        for i, link in enumerate(links):
            servo_min, servo_max = servo_ranges[i]
            offset = self.home_offsets[i]

            if i < 3:
                min_q = -(servo_max - offset)
                max_q = -(servo_min - offset)
            else:
                min_q = servo_min - offset
                max_q = servo_max - offset

            # Radyana çevir
            link.qlim = [min_q * np.pi / 180, max_q * np.pi / 180]

        return DHRobot(links, name="6DOF_Robot")

    def kinematics_to_servo_angles(self, theta_degrees):
        """
        Kinematik hesaplama açılarını servo açılarına dönüştürür

        Kinematik sıralama: [θ1(Servo6), θ2(Servo5), θ3(Servo4), θ4(Servo2), θ5(Servo1), θ6(Servo0)]
        Servo sıralama: {0: θ6, 1: θ5, 2: θ4, 4: θ3, 5: θ2, 6: θ1}
        """
        if len(theta_degrees) != 6:
            raise ValueError("6 eklem açısı gerekli")

        servo_angles = {
            6: theta_degrees[0],  # θ1 → Servo 6
            5: theta_degrees[1],  # θ2 → Servo 5
            4: theta_degrees[2],  # θ3 → Servo 4
            2: theta_degrees[3],  # θ4 → Servo 2
            1: theta_degrees[4],  # θ5 → Servo 1
            0: theta_degrees[5]   # θ6 → Servo 0
        }

        return servo_angles

    def _joint_deg_to_robot_q(self, joint_angles_deg):
        """Convert joint angles with offsets into robot configuration space radians."""
        joint_angles = np.array(joint_angles_deg, dtype=float)
        actual_angles = joint_angles - np.array(self.home_offsets)
        actual_angles[0] = -actual_angles[0]
        actual_angles[1] = -actual_angles[1]
        actual_angles[2] = -actual_angles[2]
        return actual_angles * np.pi / 180

    def _robot_q_to_joint_deg(self, joint_angles_rad):
        """Convert robot configuration space radians back to joint angles with offsets."""
        joint_angles = np.array(joint_angles_rad, dtype=float).copy()
        joint_angles[0] = -joint_angles[0]
        joint_angles[1] = -joint_angles[1]
        joint_angles[2] = -joint_angles[2]
        return joint_angles * 180 / np.pi + np.array(self.home_offsets)

    def _xy_transform_matrix(self):
        """Fiziksel XY koordinatlarını model koordinatlarına dönüştüren lineer matris."""
        x_gain = self.settings.get('x_axis_gain', 1.0)
        # y_gain = self.settings.get('y_axis_gain', 1.0) # Eski ayar
        y_gain = 1/3  # Fiziksel Y eksenindeki ~3 kat fazla hareketi telafi etmek için eklendi.
        theta = np.deg2rad(self.settings.get('xy_axis_rotation_deg', 0.0))
        c_xy = self.settings.get('x_from_y_coupling', 0.0)
        c_yx = self.settings.get('y_from_x_coupling', 0.0)

        # Coupling -> Scaling -> Rotation sıralaması
        coupling = np.array([[1.0, c_xy],
                             [c_yx, 1.0]])
        scaling = np.diag([x_gain, y_gain])
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        rotation = np.array([[cos_t, -sin_t],
                             [sin_t, cos_t]])

        return rotation @ scaling @ coupling

    def _apply_axis_corrections(self, position):
        """Fiziksel koordinatı model koordinatına dönüştürür (lineer dönüşüm + ofset)."""
        pos = np.array(position, dtype=float).copy()
        z_gain = self.settings.get('z_axis_gain', 1.0)
        x_offset = self.settings.get('x_axis_offset', 0.0)
        y_offset = self.settings.get('y_axis_offset', 0.0)
        z_offset = self.settings.get('z_axis_offset', 0.0)

        xy_model = self._xy_transform_matrix() @ np.array([pos[0], pos[1]])
        corrected_xy = xy_model + np.array([x_offset, y_offset])
        corrected_z = z_gain * pos[2] + z_offset
        return np.array([corrected_xy[0], corrected_xy[1], corrected_z])

    def _restore_axis_corrections(self, position):
        """Model koordinatlarını fiziksel koordinata geri çevirir."""
        pos = np.array(position, dtype=float).copy()
        x_offset = self.settings.get('x_axis_offset', 0.0)
        y_offset = self.settings.get('y_axis_offset', 0.0)
        z_offset = self.settings.get('z_axis_offset', 0.0)
        z_gain = self.settings.get('z_axis_gain', 1.0)

        shifted_xy = np.array([pos[0] - x_offset, pos[1] - y_offset])
        transform = self._xy_transform_matrix()
        det = np.linalg.det(transform)
        if abs(det) < 1e-6:
            raise ValueError("Eksen dönüşüm matrisi singüler! Lütfen kalibrasyon parametrelerini kontrol edin.")

        inv_transform = np.linalg.inv(transform)
        restored_xy = inv_transform @ shifted_xy

        if abs(z_gain) < 1e-6:
            raise ValueError("Z ekseni kazancı 0 olamaz.")

        restored_z = (pos[2] - z_offset) / z_gain
        return np.array([restored_xy[0], restored_xy[1], restored_z])

    def forward_kinematics(self, theta_degrees):
        """Forward kinematik hesaplaması"""
        # Robot home pozisyonundaki gerçek açıları (offset değerleri)
        theta_rad = self._joint_deg_to_robot_q(theta_degrees)

        # Forward kinematics hesapla
        T = self.robot.fkine(theta_rad)

        # Pozisyon bilgisi
        position = self._restore_axis_corrections(T.t)

        # Orientasyon bilgisi (RPY)
        rpy = T.rpy()
        rpy_degrees = rpy * 180 / np.pi

        return T, position, rpy_degrees

    def inverse_kinematics(self, x, y, z, orientation=None):
        """Inverse kinematics hesaplaması"""
        # Fiziksel koordinatı robot modeline göre düzelt
        target_coords = self._apply_axis_corrections([x, y, z])

        # Hedef pose oluştur
        if orientation is None:
            target_pose = SE3.Trans(*target_coords)
            mask = [1, 1, 1, 0, 0, 0]
        else:
            roll, pitch, yaw = np.array(orientation) * np.pi / 180
            target_pose = SE3.Trans(*target_coords) * SE3.RPY(roll, pitch, yaw)
            mask = [1, 1, 1, 1, 1, 1]

        # IK çöz
        if self.last_joint_angles is not None:
            q0 = self._joint_deg_to_robot_q(self.last_joint_angles)
        else:
            q0 = np.zeros(6)

        sol = self.robot.ikine_LM(target_pose, q0=q0, mask=mask)

        if sol.success:
            joint_angles_deg = self._robot_q_to_joint_deg(sol.q)
            self.last_joint_angles = np.array(joint_angles_deg, dtype=float)

            T_check = self.robot.fkine(sol.q)
            physical_pose = self._restore_axis_corrections(T_check.t)
            error = np.linalg.norm(physical_pose - np.array([x, y, z]))
            return True, joint_angles_deg, error, T_check
        else:
            return False, None, None, None

    def get_current_position(self):
        """Mevcut end-effector pozisyonunu hesapla"""
        try:
            # Mevcut servo açılarını al
            current_servo_angles = []
            servo_mapping = [6, 5, 4, 2, 1, 0]  # θ1→Servo6, θ2→Servo5, etc.

            for i, servo_id in enumerate(servo_mapping):
                try:
                    angle = self.kit.servo[servo_id].angle
                    if angle is None:
                        angle = self.servo_home_angles.get(servo_id, 90)
                    current_servo_angles.append(angle)
                except:
                    # Hata durumunda varsayılan değer
                    current_servo_angles.append(self.servo_home_angles.get(servo_id, 90))

            # Forward kinematics ile pozisyonu hesapla
            T, position, rpy = self.forward_kinematics(current_servo_angles)
            return position

        except Exception as e:
            print(f"⚠️ Mevcut pozisyon okunamadı: {e}")
            return np.array([0, 0, 30])  # Varsayılan pozisyon

    def move_to_position(self, x, y, z, orientation=None, smooth=False, duration=1.0, path_type="joint"):
        """
        Hedef pozisyona git (IK + Servo hareket)

        Parametreler:
        x, y, z: Hedef pozisyon koordinatları (cm)
        orientation: [roll, pitch, yaw] derece (opsiyonel)
        smooth: Yumuşak hareket
        duration: Hareket süresi
        path_type: "joint" (eklem interpolasyonu) veya "linear" (doğrusal yol)
        """
        print(f"\n🎯 Pozisyona hareket: X={x:.2f}, Y={y:.2f}, Z={z:.2f}")
        print(f"📍 Hareket tipi: {'Doğrusal yol' if path_type == 'linear' else 'Eklem interpolasyonu'}")

        if not self.settings.get('use_orientation', False):
            orientation = None

        if path_type == "linear":
            return self._move_to_position_linear(x, y, z, orientation, duration, smooth)
        else:
            return self._move_to_position_joint(x, y, z, orientation, smooth, duration)

    def _move_to_position_joint(self, x, y, z, orientation=None, smooth=False, duration=1.0):
        """Eklem interpolasyonu ile hareket (orijinal yöntem)"""
        # IK hesapla
        success, joint_angles, error, T = self.inverse_kinematics(x, y, z, orientation)

        if not success:
            print("❌ Bu pozisyon için çözüm bulunamadı!")
            return False, None

        print(f"✅ IK çözümü bulundu (hata: {error:.4f} cm)")

        # Servo açılarına dönüştür
        servo_angles = self.kinematics_to_servo_angles(joint_angles)

        # Servo'ları hareket ettir
        self.move_servos(servo_angles, smooth=smooth, duration=duration)

        return True, joint_angles

    def _move_to_position_linear(self, x, y, z, orientation=None, duration=1.0, smooth=False):
        """Doğrusal yol ile hareket (pozisyon interpolasyonu)"""
        # Mevcut pozisyonu al
        current_pos = self.get_current_position()
        target_pos = np.array([x, y, z])

        print(f"📍 Başlangıç pozisyonu: X={current_pos[0]:.2f}, Y={current_pos[1]:.2f}, Z={current_pos[2]:.2f}")

        # Güvenlik kontrolü - hedef pozisyon IK çözümünü kontrol et
        test_success, test_angles, test_error, test_T = self.inverse_kinematics(x, y, z, orientation)
        if not test_success:
            print("❌ Hedef pozisyon için IK çözümü bulunamadı!")
            return False, None

        # Mevcut pozisyon için IK çözümü
        current_success, current_angles, current_error, current_T = self.inverse_kinematics(
            current_pos[0], current_pos[1], current_pos[2], orientation
        )

        if current_success:
            # Eklem açıları arasındaki maksimum farkı hesapla
            angle_diffs = np.abs(np.array(test_angles) - np.array(current_angles))
            max_angle_diff = np.max(angle_diffs)

            print(f"🔍 Maksimum eklem açısı farkı: {max_angle_diff:.1f}°")

            # Eğer açı farkı çok büyükse (60°'den fazla), güvenli hareket öner
            if max_angle_diff > 60:
                print(f"⚠️ UYARI: Büyük açı farkı tespit edildi ({max_angle_diff:.1f}°)")
                print("🛡️ Güvenlik için ara pozisyon öneriliyor...")

                # Z koordinatını yükseltilmiş ara pozisyon öner
                safe_z = max(current_pos[2], target_pos[2]) + 5  # 5cm yüksek
                print(f"💡 Önerilen ara pozisyon: X={x:.1f}, Y={y:.1f}, Z={safe_z:.1f}")

                choice = input("Güvenli hareket yapmak ister misiniz? (y/n/cancel): ").strip().lower()
                if choice == 'cancel':
                    print("❌ Hareket iptal edildi")
                    return False, None
                elif choice == 'y':
                    # İlk ara pozisyona git (Z yüksek)
                    print("\n🛡️ 1. Adım: Güvenli yüksekliğe çıkılıyor...")
                    mid_success = self._execute_linear_movement(current_pos, np.array([x, y, safe_z]), orientation, smooth)
                    if not mid_success:
                        return False, None

                    # Sonra hedef pozisyona git
                    print("\n🛡️ 2. Adım: Hedef pozisyona iniliyor...")
                    return self._execute_linear_movement(np.array([x, y, safe_z]), target_pos, orientation, smooth)

        # Normal doğrusal hareket
        return self._execute_linear_movement(current_pos, target_pos, orientation, smooth)

    def _execute_linear_movement(self, start_pos, target_pos, orientation, smooth=False):
        """Gerçek doğrusal hareket işlemini yapar"""
        # Mesafeyi hesapla
        distance = np.linalg.norm(target_pos - start_pos)
        print(f"📏 Hareket mesafesi: {distance:.2f} cm")

        if distance < 0.1:  # 1mm'den küçükse hareket etme
            print("✅ Robot zaten pozisyonda!")
            return True, None

        # Adım sayısını hesapla (ayarlardan al)
        steps = max(5, int(distance * self.settings['steps_per_cm']))  # Minimum 5 adım
        step_delay = self.settings['step_delay']  # Ayarlardan al

        print(f"🔄 Adım sayısı: {steps}, Adım süresi: {step_delay:.3f}s")
        print(f"⏱️ Toplam hareket süresi: {steps * step_delay:.2f} saniye")
        if smooth:
            print("✨ Yumuşatılmış doğrusal hareket aktif!")

        # Doğrusal interpolasyon
        for step in range(steps + 1):
            t = step / steps
            
            if smooth:
                # Harekete yumuşak bir başlangıç ve bitiş vermek için easing uygula
                eased_t = (1 - math.cos(t * math.pi)) / 2
                current_step_pos = start_pos + eased_t * (target_pos - start_pos)
            else:
                # Mevcut ara pozisyonu hesapla
                current_step_pos = start_pos + t * (target_pos - start_pos)

            # IK hesapla
            success, joint_angles, error, T = self.inverse_kinematics(
                current_step_pos[0], current_step_pos[1], current_step_pos[2], orientation
            )

            if not success:
                print(f"⚠️ Adım {step}/{steps} için IK çözümü bulunamadı, atlanıyor...")
                continue

            # Servo açılarına dönüştür
            servo_angles = self.kinematics_to_servo_angles(joint_angles)
            
            # İlk adımda, konfigürasyon değişikliği gerekip gerekmediğini kontrol et
            if step == 0:
                # Mevcut donanım açılarını oku
                current_hw_angles = {}
                can_read_all = True
                for servo_id in servo_angles.keys():
                    try:
                        angle = self.kit.servo[servo_id].angle
                        if angle is None:
                            can_read_all = False
                            break
                        current_hw_angles[servo_id] = angle
                    except Exception:
                        can_read_all = False
                        break

                # Sadece tüm açıları okuyabildiysek farkı kontrol et
                if can_read_all:
                    max_angle_diff = 0
                    for servo_id, target_angle in servo_angles.items():
                        diff = abs(target_angle - current_hw_angles[servo_id])
                        if diff > max_angle_diff:
                            max_angle_diff = diff
                    
                    if max_angle_diff < 2.0:
                        self._move_servos_direct(servo_angles) # Fark küçük, direkt hareket
                    else:
                        # Fark büyük, yumuşak geçiş
                        print(f"🔄 Başlangıç konfigürasyonuna yumuşak geçiş (maks fark: {max_angle_diff:.1f}°)...")
                        self.move_servos(servo_angles, smooth=True, duration=0.5)
                else:
                    # Açıları okuyamazsak, güvenli tarafta kal ve yumuşak geçiş yap
                    print("⚠️ Mevcut açılar okunamadı, güvenli yumuşak geçiş yapılıyor...")
                    self.move_servos(servo_angles, smooth=True, duration=0.5)
            else:
                self._move_servos_direct(servo_angles)

            # Progress göster (her 10 adımda bir veya son adım)
            if step % max(1, steps // 10) == 0 or step == steps:
                progress = int((step / steps) * 100)
                print(f"📊 İlerleme: %{progress} - X={current_step_pos[0]:.1f}, Y={current_step_pos[1]:.1f}, Z={current_step_pos[2]:.1f}")

            if step < steps:
                time.sleep(step_delay)

        print("✅ Hareket tamamlandı!")
        return True, None

    def move_with_joint_angles(self, theta_degrees, smooth=False, duration=1.0):
        """
        Doğrudan eklem açılarıyla hareket

        Parametreler:
        theta_degrees: [θ1, θ2, θ3, θ4, θ5, θ6] eklem açıları
        smooth: Yumuşak hareket
        duration: Hareket süresi
        """
        print(f"\n🎯 Eklem açılarıyla hareket")

        # Forward kinematics ile pozisyonu hesapla
        T, position, rpy = self.forward_kinematics(theta_degrees)

        print(f"📍 Hedef pozisyon: X={position[0]:.2f}, Y={position[1]:.2f}, Z={position[2]:.2f}")

        # Servo açılarına dönüştür
        servo_angles = self.kinematics_to_servo_angles(theta_degrees)

        # Servo'ları hareket ettir
        self.move_servos(servo_angles, smooth=smooth, duration=duration)

        self.last_joint_angles = np.array(theta_degrees, dtype=float)

        return T, position, rpy

    def move_servos(self, servo_angles, smooth=False, duration=1.0):
        """Servo'ları hareket ettir"""
        print(f"\n🎯 Servo hareketi başlatılıyor... {'(Yumuşak)' if smooth else '(Hızlı)'}")

        if smooth:
            self._move_servos_smooth(servo_angles, duration)
        else:
            self._move_servos_direct(servo_angles)

    def _move_servos_direct(self, servo_angles):
        """Servo'ları direkt hareket ettir"""
        for servo_id, angle in servo_angles.items():
            if servo_id not in self.servo_ids:
                print(f"⚠️ Servo {servo_id} tanımlı değil, atlanıyor.")
                continue

            # Açı sınırı kontrolü
            if servo_id == 5:
                max_angle = 270
            else:
                max_angle = 180

            if angle < 0 or angle > max_angle:
                print(f"⚠️ Servo {servo_id} için geçersiz açı: {angle}° (0-{max_angle}° arası olmalı)")
                continue

            try:
                self.kit.servo[servo_id].angle = angle
                print(f"✅ Servo {servo_id}: {angle:.1f}°")
            except Exception as e:
                print(f"❌ Servo {servo_id} hatası: {e}")

        print("🏁 Servo hareketi tamamlandı!\n")

    def _move_servos_smooth(self, servo_angles, duration):
        """Servo'ları yumuşak geçişle hareket ettir"""
        # Mevcut açıları al ve hedef açıları hazırla
        start_angles = {}
        target_angles = {}

        for servo_id, target_angle in servo_angles.items():
            if servo_id not in self.servo_ids:
                continue

            # Açı sınırı kontrolü
            if servo_id == 5:
                max_angle = 270
            else:
                max_angle = 180

            if target_angle < 0 or target_angle > max_angle:
                print(f"⚠️ Servo {servo_id} için geçersiz açı: {target_angle}°")
                continue

            # Mevcut açıyı al
            try:
                current_angle = self.kit.servo[servo_id].angle
                if current_angle is None:
                    current_angle = self.servo_home_angles.get(servo_id, 90)

                start_angles[servo_id] = current_angle
                target_angles[servo_id] = target_angle
                print(f"📍 Servo {servo_id}: {current_angle:.1f}° → {target_angle:.1f}°")
            except Exception as e:
                print(f"❌ Servo {servo_id} mevcut açı okunamadı: {e}")
                continue

        if not start_angles:
            print("❌ Hareket edilecek servo bulunamadı!")
            return

        # Adım sayısı hesapla
        max_angle_diff = 0
        for servo_id in start_angles:
            angle_diff = abs(target_angles[servo_id] - start_angles[servo_id])
            max_angle_diff = max(max_angle_diff, angle_diff)

        steps = max(10, min(100, int(max_angle_diff * 1.5)))
        step_delay = self.settings['step_delay']  # Ayarlardan al

        print(f"🔄 Maksimum açı farkı: {max_angle_diff:.1f}°, Adım sayısı: {steps}")

        def smooth_movement():
            try:
                for step in range(steps + 1):
                    t = step / steps
                    eased_t = (1 - math.cos(t * math.pi)) / 2

                    for servo_id in start_angles:
                        start_angle = start_angles[servo_id]
                        target_angle = target_angles[servo_id]
                        current_position = start_angle + (target_angle - start_angle) * eased_t

                        try:
                            self.kit.servo[servo_id].angle = current_position
                        except Exception as e:
                            print(f"⚠️ Servo {servo_id} hareket hatası: {e}")

                    if step < steps:
                        time.sleep(step_delay)

                print("\n✅ Yumuşak hareket tamamlandı:")
                for servo_id, target_angle in target_angles.items():
                    print(f"  Servo {servo_id}: {target_angle:.1f}°")
                print()

            except Exception as e:
                print(f"❌ Yumuşak hareket hatası: {e}")

        movement_thread = threading.Thread(target=smooth_movement)
        movement_thread.daemon = True
        movement_thread.start()
        movement_thread.join()

    def home_position(self):
        """Robotu tanımlanmış ana pozisyona (38, 0, 0) götürür."""
        print("🏠 Ana pozisyona (38, 0, 0) gidiliyor...")
        
        home_x, home_y, home_z = 38.0, 0.0, 0.0
        
        # Ayarlardan hareket tipini alarak hareket et
        path_type = self.settings.get('movement_type', 'joint')
        smooth = self.settings.get('smooth_movement', False) if path_type == 'joint' else False

        success, _ = self.move_to_position(
            home_x, home_y, home_z,
            orientation=None,  # Orientasyon serbest bırakılabilir veya varsayılan ayarlanabilir
            smooth=smooth,
            path_type=path_type
        )
        
        if success:
            print(f"✅ Ana pozisyona ulaşıldı: X={home_x}, Y={home_y}, Z={home_z}")
        else:
            print(f"❌ Ana pozisyona ulaşılamadı!")

    def get_current_angles(self):
        """Mevcut servo açılarını göster"""
        print("📊 Mevcut servo açıları:")
        for servo_id in self.servo_ids:
            try:
                angle = self.kit.servo[servo_id].angle
                if angle is not None:
                    print(f"  Servo {servo_id}: {angle:.1f}°")
                else:
                    print(f"  Servo {servo_id}: Belirsiz")
            except:
                print(f"  Servo {servo_id}: Okunamıyor")

    def print_forward_results(self, theta_input, T, position, rpy_degrees):
        """Forward kinematics sonuçlarını yazdır"""
        print("=" * 60)
        print("6-DOF ROBOT FORWARD KINEMATICS")
        print("=" * 60)

        print(f"\nGirdi Açıları (derece):")
        servo_names = ["(Servo 6)", "(Servo 5)", "(Servo 4)", "(Servo 2)", "(Servo 1)", "(Servo 0)"]
        for i, angle in enumerate(theta_input):
            print(f"  θ{i+1} {servo_names[i]} = {angle:8.2f}°")

        print(f"\nEnd-Effector Pozisyonu:")
        print(f"  X = {position[0]:8.3f} cm")
        print(f"  Y = {position[1]:8.3f} cm")
        print(f"  Z = {position[2]:8.3f} cm")

        print(f"\nEnd-Effector Orientasyonu (RPY):")
        print(f"  Roll  = {rpy_degrees[0]:8.3f}°")
        print(f"  Pitch = {rpy_degrees[1]:8.3f}°")
        print(f"  Yaw   = {rpy_degrees[2]:8.3f}°")

    def visualize_robot(self, joint_angles_deg, target_pos=None, title="Robot Visualization"):
        """Robot'u matplotlib ile görselleştir"""
        actual_angles = np.array(joint_angles_deg) - np.array(self.home_offsets)
        actual_angles[0] = -actual_angles[0]
        actual_angles[1] = -actual_angles[1]
        actual_angles[2] = -actual_angles[2]

        q = actual_angles * np.pi / 180

        try:
            print(f"\n📊 {title}...")
            self.robot.plot(q, backend='pyplot')

            if target_pos is not None:
                print(f"🎯 Hedef pozisyon: X={target_pos[0]:.2f}, Y={target_pos[1]:.2f}, Z={target_pos[2]:.2f}")

            print("✅ Görselleştirme başarılı!")
            print("🔍 Pencereyi kapatmak için [X] butonuna tıklayın")
            return True
        except Exception as e:
            print(f"⚠️ Görselleştirme hatası: {e}")
            return False



    def show_current_settings(self):
        """Mevcut ayarları göster"""
        print("\n📋 Mevcut Ayarlar:")
        print(f"  Hareket Tipi: {'Doğrusal yol' if self.settings['movement_type'] == 'linear' else 'Eklem interpolasyonu'}")
        print(f"  Yumuşak Hareket: {'Aktif' if self.settings['smooth_movement'] else 'Pasif'}")
        print(f"  Her Adım Süresi: {self.settings['step_delay']} saniye")
        print(f"  Otomatik Görselleştirme: {'Aktif' if self.settings['auto_visualize'] else 'Pasif'}")

        if self.settings['use_orientation']:
            if self.settings['default_orientation'] == 'vertical':
                print(f"  Varsayılan Orientasyon: Dikey (0,135,0)")
            else:
                print(f"  Varsayılan Orientasyon: Yatay (0,90,0)")
        else:
            print("  Bilek Orientasyonu: Kapalı (sadece pozisyon)")

        print(f"  Doğrusal Hareket Adım/cm: {self.settings['steps_per_cm']}")
        print(f"  X Kalibrasyonu: gain={self.settings['x_axis_gain']:.4f}, offset={self.settings['x_axis_offset']:.2f} cm")
        print(f"  Y Kalibrasyonu: gain={self.settings['y_axis_gain']:.4f}, offset={self.settings['y_axis_offset']:.2f} cm")
        print(f"  Z Kalibrasyonu: gain={self.settings['z_axis_gain']:.4f}, offset={self.settings['z_axis_offset']:.2f} cm")
        print(f"  XY Düzlemi Dönmesi: {self.settings['xy_axis_rotation_deg']:.2f}°")
        print(f"  X←Y Coupling: {self.settings['x_from_y_coupling']:.4f} cm/cm")
        print(f"  Y←X Coupling: {self.settings['y_from_x_coupling']:.4f} cm/cm")

    def configure_axis_calibration(self):
        """XY düzlemindeki dönme ve eksen ölçek/ofsetlerini ayarla."""
        print("\n" + "="*50)
        print("EKSENKALİBRASYONU")
        print("="*50)
        print(f"Mevcut ayarlar:")
        print(f"  X → gain={self.settings['x_axis_gain']:.4f}, offset={self.settings['x_axis_offset']:.2f} cm")
        print(f"  Y → gain={self.settings['y_axis_gain']:.4f}, offset={self.settings['y_axis_offset']:.2f} cm")
        print(f"  Z → gain={self.settings['z_axis_gain']:.4f}, offset={self.settings['z_axis_offset']:.2f} cm")
        print(f"  XY Rotasyonu → {self.settings['xy_axis_rotation_deg']:.2f}° (pozitif saat yönü tersi)")
        print(f"  Coupling X←Y → {self.settings['x_from_y_coupling']:.4f} cm/cm")
        print(f"  Coupling Y←X → {self.settings['y_from_x_coupling']:.4f} cm/cm")

        def update_gain(key, label):
            raw = input(f"{label} gain (Enter=değiştirme): ").strip()
            if not raw:
                return
            try:
                value = float(raw)
                if abs(value) < 1e-4:
                    print("❌ Gain 0'a çok yakın olamaz!")
                    return
                self.settings[key] = value
                print(f"✅ {label} gain güncellendi!")
            except ValueError:
                print("❌ Geçersiz değer!")

        def update_offset(key, label):
            raw = input(f"{label} offset (cm) (Enter=değiştirme): ").strip()
            if not raw:
                return
            try:
                value = float(raw)
                self.settings[key] = value
                print(f"✅ {label} offset güncellendi!")
            except ValueError:
                print("❌ Geçersiz değer!")

        update_gain('x_axis_gain', 'X')
        update_offset('x_axis_offset', 'X')
        # update_gain('y_axis_gain', 'Y') # Orantısız Y ekseni hareketine neden olduğu için devre dışı bırakıldı.
        update_offset('y_axis_offset', 'Y')
        update_gain('z_axis_gain', 'Z')
        update_offset('z_axis_offset', 'Z')

        coupling_xy = input("Y artışı başına X düzeltmesi (cm/cm, Enter=değiştirme): ").strip()
        if coupling_xy:
            try:
                self.settings['x_from_y_coupling'] = float(coupling_xy)
                print("✅ X←Y coupling güncellendi!")
            except ValueError:
                print("❌ Geçersiz değer!")

        coupling_yx = input("X artışı başına Y düzeltmesi (cm/cm, Enter=değiştirme): ").strip()
        if coupling_yx:
            try:
                self.settings['y_from_x_coupling'] = float(coupling_yx)
                print("✅ Y←X coupling güncellendi!")
            except ValueError:
                print("❌ Geçersiz değer!")

        rot_input = input("XY düzlemi dönme açısı (derece, Enter=değiştirme): ").strip()
        if rot_input:
            try:
                value = float(rot_input)
                self.settings['xy_axis_rotation_deg'] = value
                print("✅ Rotasyon açısı güncellendi!")
            except ValueError:
                print("❌ Geçersiz değer!")

        print("🔧 Güncel eksen kalibrasyonu kaydedildi.")

    def change_settings(self):
        """Ayarları değiştir"""
        while True:
            print("\n" + "="*50)
            print("AYARLAR MENÜSÜ")
            print("="*50)
            self.show_current_settings()
            print("\nDeğiştirmek istediğiniz ayar:")
            print("1. Hareket Tipi")
            print("2. Yumuşak Hareket")
            print("3. Adım Süresi")
            print("4. Otomatik Görselleştirme")
            print("5. Bilek Orientasyonu Kullanımı")
            print("6. Varsayılan Orientasyon")
            print("7. Doğrusal Hareket Adım Sayısı")
            print("8. Eksen Kalibrasyonu")
            print("9. Ana Menüye Dön")

            choice = input("\nSeçiminizi yapın (1-9): ").strip()

            if choice == '1':
                print("\nHareket Tipi:")
                print("1. Eklem interpolasyonu")
                print("2. Doğrusal yol")
                movement_choice = input("Seçin (1/2): ").strip()
                self.settings['movement_type'] = "linear" if movement_choice == "2" else "joint"
                print("✅ Hareket tipi güncellendi!")

            elif choice == '2':
                smooth_choice = input("Yumuşak hareket? (y/n): ").strip().lower()
                self.settings['smooth_movement'] = smooth_choice == 'y'
                print("✅ Yumuşak hareket ayarı güncellendi!")

            elif choice == '3':
                step_delay_input = input("Yeni adım süresi (saniye): ").strip()
                try:
                    new_delay = float(step_delay_input)
                    if new_delay > 0:
                        self.settings['step_delay'] = new_delay
                        print("✅ Adım süresi güncellendi!")
                    else:
                        print("❌ Adım süresi pozitif olmalı!")
                except ValueError:
                    print("❌ Geçersiz değer!")

            elif choice == '4':
                viz_choice = input("Otomatik görselleştirme? (y/n): ").strip().lower()
                self.settings['auto_visualize'] = viz_choice == 'y'
                print("✅ Görselleştirme ayarı güncellendi!")

            elif choice == '5':
                orient_use_choice = input("Pozisyonlarda bilek orientasyonu korunsun mu? (y/n): ").strip().lower()
                self.settings['use_orientation'] = orient_use_choice == 'y'
                if self.settings['use_orientation']:
                    print("✅ Bilek orientasyonu aktif")
                else:
                    print("✅ Bilek orientasyonu kapatıldı (sadece pozisyon)")

            elif choice == '6':
                print("\nVarsayılan Orientasyon:")
                print("1. Yatay (0,90,0)")
                print("2. 45 derece (0,135,0)")
                if not self.settings['use_orientation']:
                    print("⚠️ Önce bilek orientasyonunu etkinleştirmelisiniz.")
                else:
                    orient_choice = input("Seçin (1/2): ").strip()
                    if orient_choice == '2':
                        self.settings['default_orientation'] = 'vertical'
                        print("✅ Dikey orientasyon seçildi")
                    elif orient_choice == '1':
                        self.settings['default_orientation'] = 'horizontal'
                        print("✅ Yatay orientasyon seçildi")
                    else:
                        print("❌ Geçersiz seçim!")

            elif choice == '7':
                steps_input = input("Yeni adım sayısı (cm başına): ").strip()
                try:
                    new_steps = int(steps_input)
                    if new_steps < 1:
                        print("❌ Adım sayısı en az 1 olmalı!")
                    else:
                        self.settings['steps_per_cm'] = new_steps
                        print("✅ Doğrusal hareket adım sayısı güncellendi!")
                except ValueError:
                    print("❌ Geçersiz değer!")
            elif choice == '8':
                self.configure_axis_calibration()

            elif choice == '9':
                break

            else:
                print("❌ Geçersiz seçim!")

def main_menu():
    """Ana menü ve kullanıcı arayüzü"""
    controller = IntegratedRobotController()

    # Başlangıç ayarları yap
    # controller.setup_initial_settings()

    while True:
        print("\n" + "="*60)
        print("ENTEGRe ROBOT KONTROL SISTEMİ")
        print("="*60)
        print("1. Pozisyona Git (Inverse Kinematics)")
        print("2. Eklem Açılarıyla Git (Forward Kinematics)")
        print("3. Manuel Servo Kontrolü")
        print("4. Forward Kinematics Hesapla")
        print("5. Mevcut Açıları Göster")
        print("6. Ana Pozisyona Git")
        print("7. Ayarlar")
        print("8. Çıkış")
        print("="*60)

        try:
            choice = input("\nSeçiminizi yapın (1-8): ").strip()

            if choice == '1':
                # Pozisyona git (IK)
                print("\n" + "="*50)
                print("POZISYONA GİT (INVERSE KINEMATICS)")
                print("="*50)

                try:
                    coord_input = input("X,Y,Z koordinatları girin (örnek: 25,15,30): ")
                    coords = [float(x.strip()) for x in coord_input.split(',')]

                    if len(coords) != 3:
                        print("Hata: 3 koordinat değeri girmelisiniz!")
                        continue

                    x, y, z = coords

                    if controller.settings['use_orientation']:
                        if controller.settings['default_orientation'] == 'vertical':
                            orientation = [0, 135, 0]
                        else:
                            orientation = [0, 90, 0]
                    else:
                        orientation = None

                    # Ayarlardaki değerleri kullan
                    path_type = controller.settings['movement_type']
                    smooth = controller.settings['smooth_movement']
                    step_delay = controller.settings['step_delay']

                    print(f"\n📋 Kullanılacak ayarlar:")
                    print(f"  Hareket tipi: {'Doğrusal yol' if path_type == 'linear' else 'Eklem interpolasyonu'}")
                    print(f"  Yumuşak hareket: {'Aktif' if smooth else 'Pasif'}")
                    print(f"  Adım süresi: {step_delay} saniye")
                    if orientation is None:
                        print("  Bilek orientasyonu: Serbest (yalnızca pozisyon)")
                    else:
                        orient_name = 'Dikey (0,135,0)' if orientation == [0, 135, 0] else 'Yatay (0,90,0)'
                        print(f"  Orientasyon: {orient_name}")

                    success, angles = controller.move_to_position(x, y, z, orientation=orientation, smooth=smooth, duration=1.0, path_type=path_type)

                    if success:
                        print(f"✅ Hedef pozisyona başarıyla ulaşıldı!")

                        # Otomatik görselleştirme veya sor
                        if controller.settings['auto_visualize']:
                            controller.visualize_robot(angles, [x, y, z], "Pozisyon Kontrolü")
                        else:
                            viz_choice = input("\n🎨 Robotu görselleştirmek ister misiniz? (y/n): ").strip().lower()
                            if viz_choice == 'y':
                                controller.visualize_robot(angles, [x, y, z], "Pozisyon Kontrolü")

                except ValueError:
                    print("Hata: Sayısal değerler girin")
                except Exception as e:
                    print(f"Hata: {e}")

            elif choice == '2':
                # Eklem açılarıyla git (FK)
                print("\n" + "="*50)
                print("EKLEM AÇILARIYLA GİT (FORWARD KINEMATICS)")
                print("="*50)

                try:
                    angles_input = input("6 eklem açısını virgülle ayırarak girin (örnek: 90,135,62,95,123,165): ")
                    user_angles = [float(x.strip()) for x in angles_input.split(',')]

                    if len(user_angles) != 6:
                        print("Hata: 6 açı değeri girmelisiniz!")
                        continue

                    # Ayarlardaki değerleri kullan
                    smooth = controller.settings['smooth_movement']
                    step_delay = controller.settings['step_delay']

                    print(f"\n📋 Kullanılacak ayarlar:")
                    print(f"  Yumuşak hareket: {'Aktif' if smooth else 'Pasif'}")
                    print(f"  Adım süresi: {step_delay} saniye")

                    T, position, rpy = controller.move_with_joint_angles(user_angles, smooth=smooth, duration=1.0)
                    controller.print_forward_results(user_angles, T, position, rpy)

                    # Otomatik görselleştirme veya sor
                    if controller.settings['auto_visualize']:
                        controller.visualize_robot(user_angles, position.tolist(), "Eklem Açısı Kontrolü")
                    else:
                        viz_choice = input("\n🎨 Robotu görselleştirmek ister misiniz? (y/n): ").strip().lower()
                        if viz_choice == 'y':
                            controller.visualize_robot(user_angles, position.tolist(), "Eklem Açısı Kontrolü")

                except ValueError:
                    print("Hata: Sayısal değerler girin")
                except Exception as e:
                    print(f"Hata: {e}")

            elif choice == '3':
                # Manuel servo kontrolü
                print("\n" + "="*50)
                print("MANUEL SERVO KONTROLÜ")
                print("="*50)

                try:
                    servo_id = int(input("Servo ID girin (0,1,2,4,5,6): "))
                    max_angle = 270 if servo_id == 5 else 180
                    angle = float(input(f"Açı girin (0-{max_angle}°): "))

                    if servo_id not in controller.servo_ids:
                        print(f"⚠️ Servo {servo_id} tanımlı değil.")
                        continue

                    # Ayarlardaki değerleri kullan
                    smooth = controller.settings['smooth_movement']
                    step_delay = controller.settings['step_delay']

                    print(f"\n📋 Kullanılacak ayarlar:")
                    print(f"  Yumuşak hareket: {'Aktif' if smooth else 'Pasif'}")
                    print(f"  Adım süresi: {step_delay} saniye")

                    servo_angles = {servo_id: angle}
                    controller.move_servos(servo_angles, smooth=smooth, duration=1.0)

                except ValueError:
                    print("Hata: Geçersiz değer girdiniz!")
                except Exception as e:
                    print(f"Hata: {e}")

            elif choice == '4':
                # Sadece forward kinematics hesapla
                print("\n" + "="*50)
                print("FORWARD KINEMATICS HESAPLAMA")
                print("="*50)

                try:
                    angles_input = input("6 eklem açısını virgülle ayırarak girin (örnek: 90,135,62,95,123,165): ")
                    user_angles = [float(x.strip()) for x in angles_input.split(',')]

                    if len(user_angles) != 6:
                        print("Hata: 6 açı değeri girmelisiniz!")
                        continue

                    T, position, rpy = controller.forward_kinematics(user_angles)
                    controller.print_forward_results(user_angles, T, position, rpy)

                    # Otomatik görselleştirme veya sor
                    if controller.settings['auto_visualize']:
                        controller.visualize_robot(user_angles, position.tolist(), "Forward Kinematics")
                    else:
                        viz_choice = input("\n🎨 Robotu görselleştirmek ister misiniz? (y/n): ").strip().lower()
                        if viz_choice == 'y':
                            controller.visualize_robot(user_angles, position.tolist(), "Forward Kinematics")

                except ValueError:
                    print("Hata: Sayısal değerler girin")
                except Exception as e:
                    print(f"Hata: {e}")

            elif choice == '5':
                controller.get_current_angles()

            elif choice == '6':
                controller.home_position()

            elif choice == '7':
                controller.change_settings()

            elif choice == '8':
                print("\n👋 Program sonlandırılıyor...")
                break

            else:
                print("❌ Geçersiz seçim! 1-8 arası bir değer girin.")

        except KeyboardInterrupt:
            print("\n\n👋 Program sonlandırılıyor...")
            break
        except Exception as e:
            print(f"❌ Beklenmeyen hata: {e}")

if __name__ == '__main__':
    main_menu()
