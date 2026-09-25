# ══════════════════════════════════════════════════════════
# 🔬 V8 机器人 MuJoCo 物理仿真验证
#
# 功能:
#   - 将STL模型导入MuJoCo物理世界
#   - 验证质量、惯量、碰撞体
#   - 运行动力学仿真(重力/碰撞/稳定性)
#   - 输出物理属性报告
#
# ══════════════════════════════════════════════════════════

import os
import sys
import numpy as np
import trimesh
import mujoco
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

MODELS_DIR = "v8_3d_models"
ASSEMBLY_DIR = "v8_robot_assembly"
OUTPUT_DIR = "v8_mujoco_verify"


def create_mujoco_xml_from_stl(stl_path: str, output_path: str,
                                mass_kg: float = 5.0) -> str:
    """
    从STL文件创建MuJoCo XML模型
    
    Args:
        stl_path: STL文件路径
        output_path: 输出XML路径
        mass_kg: 总质量(kg)
    
    Returns:
        XML文件路径
    """
    # 加载STL获取尺寸信息
    mesh = trimesh.load(stl_path, force='mesh')
    
    # 计算包围盒
    bounds = mesh.bounds
    extents = mesh.extents
    center = mesh.centroid
    volume = mesh.volume
    
    # 估算密度 (假设铝合金)
    if volume > 0:
        density = mass_kg / volume
    else:
        density = 2700  # 铝合金默认密度 kg/m3
    
    log.info(f"模型尺寸: {extents*1000} mm")
    log.info(f"模型体积: {volume*1e6:.1f} cm³")
    log.info(f"估算密度: {density:.0f} kg/m³")
    
    # 创建MuJoCo XML (使用绝对路径)
    stl_abs = os.path.abspath(stl_path)
    stl_dir = os.path.dirname(stl_abs)
    stl_name = os.path.basename(stl_path)

    # 复制STL到输出目录 (避免路径问题)
    import shutil
    stl_copy = os.path.join(OUTPUT_DIR, stl_name)
    if not os.path.exists(stl_copy) or os.path.getsize(stl_copy) != os.path.getsize(stl_abs):
        shutil.copy2(stl_abs, stl_copy)
        log.info(f"STL已复制到: {stl_copy}")

    # 使用输出目录作为meshdir
    mesh_dir = os.path.abspath(OUTPUT_DIR)

    # 计算尺寸 (转换为米)
    size_x = extents[0] / 2
    size_y = extents[1] / 2
    size_z = extents[2] / 2

    # 使用内联box几何体避免文件路径问题
    xml_content = f'''<mujoco model="V8_Robot_Assembly">
  <compiler angle="radian"/>

  <option timestep="0.001" gravity="0 0 -9.81" iterations="50" tolerance="1e-6"/>

  <default>
    <joint armature="0.01" damping="1" limited="true"/>
    <geom condim="3" contype="1" conaffinity="1" friction="0.8 0.005 0.0001" solref="0.02 1"/>
  </default>

  <asset>
    <material name="aluminum" rgba="0.7 0.7 0.75 1.0" shininess="0.8"/>
    <material name="motor_red" rgba="0.85 0.2 0.15 1.0" shininess="0.6"/>
  </asset>

  <worldbody>
    <!-- 地面 -->
    <geom name="ground" type="plane" size="2 2 0.1" rgba="0.5 0.52 0.55 1"/>

    <!-- 光源 -->
    <light pos="0.5 0.5 2" dir="-0.3 -0.3 -1" diffuse="1 1 1"/>

    <!-- 机器人主体 (自由体) - 使用box近似 -->
    <body name="robot_chassis" pos="0 0 {center[2] + size_z + 0.05}">
      <freejoint name="chassis_free"/>

      <!-- 主碰撞体: 包围盒 -->
      <geom name="main_body" type="box" size="{size_x} {size_y} {size_z}"
            mass="{mass_kg}" material="aluminum"/>

      <!-- 质心标记 -->
      <site name="com" size="0.01" rgba="1 0 0 0.8" type="sphere"/>

      <!-- 底部4个接触点 -->
      <site name="contact_fl" pos="{size_x*0.8} {size_y*0.8} {-size_z}" size="0.005" rgba="0 1 0 0.8"/>
      <site name="contact_fr" pos="{size_x*0.8} {-size_y*0.8} {-size_z}" size="0.005" rgba="0 1 0 0.8"/>
      <site name="contact_bl" pos="-{size_x*0.8} {size_y*0.8} {-size_z}" size="0.005" rgba="0 1 0 0.8"/>
      <site name="contact_br" pos="-{size_x*0.8} {-size_y*0.8} {-size_z}" size="0.005" rgba="0 1 0 0.8"/>
    </body>
  </worldbody>

  <sensor>
    <accelerometer name="imu_accel" site="com"/>
    <gyro name="imu_gyro" site="com"/>
    <framepos name="robot_pos" objtype="body" objname="robot_chassis"/>
    <framequat name="robot_quat" objtype="body" objname="robot_chassis"/>
  </sensor>

</mujoco>'''

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(xml_content)
    
    log.info(f"MuJoCo XML已生成: {output_path}")
    return output_path


def run_physics_verification(xml_path: str, duration: float = 3.0) -> dict:
    """
    运行物理仿真验证
    
    Args:
        xml_path: MuJoCo XML路径
        duration: 仿真时长(秒)
    
    Returns:
        验证结果字典
    """
    log.info(f"\n加载MuJoCo模型: {xml_path}")
    
    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
    except Exception as e:
        log.error(f"加载失败: {e}")
        return {'success': False, 'error': str(e)}
    
    results = {
        'success': True,
        'model_info': {},
        'simulation': {},
        'stability': {},
        'physics': {}
    }
    
    # 模型基本信息
    results['model_info'] = {
        'nq': model.nq,
        'nv': model.nv,
        'nbody': model.nbody,
        'njnt': model.njnt,
        'nu': model.nu,
        'nsensor': model.nsensor,
        'timestep': model.opt.timestep,
        'gravity': list(model.opt.gravity),
    }
    
    log.info(f"\n模型信息:")
    log.info(f"  自由度(nq): {model.nq}")
    log.info(f"  速度维度(nv): {model.nv}")
    log.info(f"  刚体数量: {model.nbody}")
    log.info(f"  关节数量: {model.njnt}")
    log.info(f"  传感器数: {model.nsensor}")
    log.info(f"  时间步长: {model.opt.timestep*1000:.2f} ms")
    
    # 获取质量属性
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot_chassis")
    if body_id >= 0:
        mass = model.body_mass[body_id]
        inertia = model.body_inertia[body_id]
        
        results['physics']['mass_kg'] = mass
        results['physics']['inertia'] = list(inertia)
        results['physics']['com_local'] = list(model.body_pos[body_id])
        
        log.info(f"\n物理属性:")
        log.info(f"  质量: {mass:.3f} kg")
        log.info(f"  惯量张量(Ixx,Iyy,Izz): [{inertia[0]:.6f}, {inertia[1]:.6f}, {inertia[2]:.6f}]")
    
    # 运行仿真
    n_steps = int(duration / model.opt.timestep)
    log.info(f"\n运行仿真 ({duration}s, {n_steps}步)...")
    
    # 初始化
    mujoco.mj_resetData(model, data)
    
    # 设置初始位置 (稍微离地)
    data.qpos[2] = 0.20  # 初始高度
    
    # 数据记录
    positions = []
    velocities = []
    energies = []
    contact_forces = []
    
    for step in range(n_steps):
        mujoco.mj_step(model, data)
        
        # 记录数据 (每10步记录一次)
        if step % 10 == 0:
            pos = data.qpos[:3].copy()
            vel = data.qvel[:3].copy()
            
            # 计算能量
            ke = 0.5 * np.dot(data.qvel, data.qvel * model.body_mass[body_id]) if body_id >= 0 else 0
            pe = mass * 9.81 * pos[2]
            
            positions.append(pos.copy())
            velocities.append(vel.copy())
            energies.append({'KE': ke, 'PE': pe, 'total': ke + pe})
            
            # 接触力
            ncon = data.ncon
            total_force = 0
            for i in range(ncon):
                contact = data.contact[i]
                force = np.zeros(6)
                mujoco.mj_contactForce(model, data, i, force)
                total_force += np.linalg.norm(force[:3])
            contact_forces.append(total_force)
    
    # 分析结果
    positions = np.array(positions)
    velocities = np.array(velocities)
    
    final_pos = positions[-1]
    min_height = np.min(positions[:, 2])
    max_height = np.max(positions[:, 2])
    height_range = max_height - min_height
    
    max_velocity = np.max(np.linalg.norm(velocities, axis=1))
    final_velocity = np.linalg.norm(velocities[-1])
    
    avg_contact_force = np.mean(contact_forces) if contact_forces else 0
    max_contact_force = np.max(contact_forces) if contact_forces else 0
    
    # 稳定性判断
    is_stable = height_range < 0.05 and final_velocity < 0.1
    landed_properly = min_height < 0.03  # 接近地面
    
    results['simulation'] = {
        'duration_s': duration,
        'steps': n_steps,
        'final_position': list(final_pos),
        'min_height_m': float(min_height),
        'max_height_m': float(max_height),
        'height_range_m': float(height_range),
        'max_velocity_ms': float(max_velocity),
        'final_velocity_ms': float(final_velocity),
    }
    
    results['stability'] = {
        'is_stable': bool(is_stable),
        'landed_properly': bool(landed_properly),
        'avg_contact_force_N': float(avg_contact_force),
        'max_contact_force_N': float(max_contact_force),
        'oscillation_count': int(np.sum(np.diff(np.abs(np.diff(positions[:, 2]))) > 0.001)),
    }
    
    log.info(f"\n仿真结果:")
    log.info(f"  最终位置: ({final_pos[0]:.3f}, {final_pos[1]:.3f}, {final_pos[2]:.3f}) m")
    log.info(f"  高度范围: {min_height:.3f} ~ {max_height:.3f} m (变化{height_range*1000:.1f}mm)")
    log.info(f"  最大速度: {max_velocity:.3f} m/s")
    log.info(f"  最终速度: {final_velocity:.3f} m/s")
    log.info(f"  平均接触力: {avg_contact_force:.1f} N")
    log.info(f"  最大接触力: {max_contact_force:.1f} N")
    log.info(f"\n稳定性评估:")
    log.info(f"  着陆正确: {'YES' if landed_properly else 'NO'}")
    log.info(f"  稳定: {'YES' if is_stable else 'NO'}")
    
    return results


def generate_verification_report(results: dict, output_dir: str) -> str:
    """生成验证报告"""
    os.makedirs(output_dir, exist_ok=True)
    
    lines = []
    lines.append("=" * 70)
    lines.append("V8 机器人 MuJoCo 物理仿真验证报告")
    lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)
    
    if not results.get('success'):
        lines.append(f"\n错误: {results.get('error', '未知')}")
        report_text = "\n".join(lines)
        path = os.path.join(output_dir, "verification_report.txt")
        with open(path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        return path
    
    # 模型信息
    mi = results.get('model_info', {})
    lines.append("\n【模型信息】")
    lines.append(f"  自由度:     {mi.get('nq', 'N/A')}")
    lines.append(f"  速度维度:   {mi.get('nv', 'N/A')}")
    lines.append(f"  刚体数量:   {mi.get('nbody', 'N/A')}")
    lines.append(f"  关节数量:   {mi.get('njnt', 'N/A')}")
    lines.append(f"  时间步长:   {mi.get('timestep', 'N/A')} ms")
    lines.append(f"  重力:       {mi.get('gravity', 'N/A')}")
    
    # 物理属性
    phys = results.get('physics', {})
    lines.append("\n【物理属性】")
    lines.append(f"  质量:       {phys.get('mass_kg', 'N/A'):.3f} kg")
    inertia = phys.get('inertia', [0, 0, 0])
    lines.append(f"  惯量张量:   Ixx={inertia[0]:.6f}, Iyy={inertia[1]:.6f}, Izz={inertia[2]:.6f}")
    
    # 仿真结果
    sim = results.get('simulation', {})
    lines.append("\n【仿真结果】")
    lines.append(f"  仿真时长:   {sim.get('duration_s', 'N/A'):.1f} s ({sim.get('steps', 'N/A')} 步)")
    fp = sim.get('final_position', [0, 0, 0])
    lines.append(f"  最终位置:   ({fp[0]:.4f}, {fp[1]:.4f}, {fp[2]:.4f}) m")
    lines.append(f"  高度范围:   {sim.get('min_height_m', 0):.4f} ~ {sim.get('max_height_m', 0):.4f} m")
    lines.append(f"  高度变化:   {sim.get('height_range_m', 0)*1000:.2f} mm")
    lines.append(f"  最大速度:   {sim.get('max_velocity_ms', 0):.4f} m/s")
    lines.append(f"  最终速度:   {sim.get('final_velocity_ms', 0):.4f} m/s")
    
    # 稳定性
    stab = results.get('stability', {})
    lines.append("\n【稳定性评估】")
    lines.append(f"  正确着陆:   {'PASS' if stab.get('landed_properly') else 'FAIL'}")
    lines.append(f"  稳定状态:   {'PASS' if stab.get('is_stable') else 'FAIL'}")
    lines.append(f"  平均接触力: {stab.get('avg_contact_force_N', 0):.1f} N")
    lines.append(f"  最大接触力: {stab.get('max_contact_force_N', 0):.1f} N")
    lines.append(f"  振荡次数:   {stab.get('oscillation_count', 0)}")
    
    # 结论
    lines.append("\n" + "=" * 70)
    all_pass = stab.get('landed_properly', False) and stab.get('is_stable', False)
    if all_pass:
        lines.append("结论: PASS - 组装体物理属性正常，仿真稳定")
    else:
        lines.append("结论: WARN - 建议检查装配关系或调整参数")
    lines.append("=" * 70)
    
    report_text = "\n".join(lines)
    path = os.path.join(output_dir, "verification_report.txt")
    with open(path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print("\n" + report_text)
    return path


def main():
    log.info("=" * 60)
    log.info("V8 机器人 MuJoCo 物理仿真验证")
    log.info("=" * 60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 检查组装体STL
    stl_path = os.path.join(ASSEMBLY_DIR, "v8_robot_complete.stl")
    if not os.path.exists(stl_path):
        log.error(f"找不到组装体STL: {stl_path}")
        log.error("请先运行 v8_robot_assembler.py 生成组装体")
        return 1
    
    # 创建MuJoCo XML
    xml_path = os.path.join(OUTPUT_DIR, "v8_robot_verify.xml")
    create_mujoco_xml_from_stl(stl_path, xml_path, mass_kg=5.0)
    
    # 运行仿真验证
    results = run_physics_verification(xml_path, duration=3.0)
    
    # 生成报告
    report_path = generate_verification_report(results, OUTPUT_DIR)
    
    log.info(f"\n报告已保存: {report_path}")
    log.info(f"输出目录: {os.path.abspath(OUTPUT_DIR)}")
    
    return 0


if __name__ == "__main__":
    exit(main())
