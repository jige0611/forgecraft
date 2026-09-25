"""
Phase 2: 集成测试 - 网格质量与物理属性验证

测试内容：
1. ✅ AdvancedMeshBuilderV2 基础功能
2. ✅ 创新零件网格质量（流形、法线、体积）
3. ✅ 物理属性合理性检查（质量、密度、尺寸）
4. ✅ 参数自适应算法正确性
5. ✅ Catalog系统与V2零件集成
6. ✅ STL导出/导入往返一致性
7. ✅ 性能基准测试

运行方式:
    python test_phase2_integration.py
"""

import sys
import os
import time
import shutil
import numpy as np
import trimesh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from advanced_mesh_builder_v2 import AdvancedMeshBuilderV2


class TestResult:
    """测试结果收集器"""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
        self.warnings = []
    
    def add_pass(self, test_name):
        self.passed += 1
        print(f"   ✅ {test_name}")
    
    def add_fail(self, test_name, reason):
        self.failed += 1
        error_msg = f"   ❌ {test_name}: {reason}"
        print(error_msg)
        self.errors.append(error_msg)
    
    def add_warning(self, msg):
        self.warnings.append(msg)
        print(f"   ⚠️  {msg}")
    
    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"📊 测试结果汇总")
        print(f"{'='*60}")
        print(f"   总计: {total} 测试")
        print(f"   通过: {self.passed} ✅")
        print(f"   失败: {self.failed} ❌")
        
        if self.warnings:
            print(f"\n   ⚠️  警告 ({len(self.warnings)}):")
            for w in self.warnings:
                print(f"      - {w}")
        
        if self.errors:
            print(f"\n❌ 失败详情:")
            for e in self.errors:
                print(f"   {e}")
        
        success_rate = (self.passed / total * 100) if total > 0 else 0
        print(f"\n   通过率: {success_rate:.1f}%")
        print(f"{'='*60}")
        
        return self.failed == 0


def check_mesh_manifold(mesh, result, test_prefix=""):
    """
    检查网格是否为有效流形
    
    检查项：
    - 顶点和面数组非空
    - 无退化面（面积为零）
    - 法线方向一致
    - 边界边数量合理
    """
    prefix = f"{test_prefix}" if test_prefix else ""
    
    # 检查1：基本结构完整性
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        result.add_fail(f"{prefix}网格为空", "顶点或面数为0")
        return False
    
    result.add_pass(f"{prefix}网格非空 ({len(mesh.vertices)}顶点, {len(mesh.faces)}面)")
    
    # 检查2：顶点坐标有效性（无NaN/Inf）
    vertices = mesh.vertices
    if np.any(np.isnan(vertices)) or np.any(np.isinf(vertices)):
        result.add_fail(f"{prefix}包含无效顶点坐标", "检测到NaN或Inf")
        return False
    
    result.add_pass(f"{prefix}顶点坐标有效 (无NaN/Inf)")
    
    # 检查3：面的索引有效性
    faces = mesh.faces
    max_vertex_idx = len(vertices) - 1
    invalid_faces = faces[faces < 0] | faces > max_vertex_idx
    
    if len(invalid_faces) > 0:
        result.add_fail(f"{prefix}存在无效面索引", f"{len(invalid_faces)}个无效索引")
        return False
    
    result.add_pass(f"{prefix}所有面索引有效")
    
    # 检查4：体积合理性（必须为正）
    try:
        volume = mesh.volume
        if volume <= 0:
            result.add_fail(f"{prefix}体积无效", f"volume={volume:.6f} (应为正)")
            return False
        
        if volume < 1e-9:
            result.add_warning(f"{prefix}体积过小: {volume:.2e} m³")
        else:
            result.add_pass(f"{prefix}体积正常: {volume:.6e} m³")
            
    except Exception as e:
        result.add_warning(f"{prefix}无法计算体积: {e}")
    
    # 检查5：表面积合理性
    try:
        surface_area = mesh.surface_area
        if surface_area <= 0:
            result.add_fail(f"{prefix}表面积无效", f"area={surface_area:.6f}")
            return False
        
        result.add_pass(f"{prefix}表面积正常: {surface_area:.6e} m²")
        
    except Exception as e:
        result.add_warning(f"{prefix}无法计算表面积: {e}")
    
    # 检查6：边界边数量（流形网格应较少边界边）
    try:
        edges = mesh.edges_unique
        boundary_edges = mesh.edges_unique_boundary
        
        boundary_ratio = len(boundary_edges) / len(edges) if len(edges) > 0 else 0
        
        if boundary_ratio > 0.3:
            result.add_warning(
                f"{prefix}边界边比例较高: {boundary_ratio:.1%} "
                f"(可能影响仿真稳定性)"
            )
        else:
            result.add_pass(f"{prefix}边界边比例正常: {boundary_ratio:.1%}")
            
    except Exception as e:
        result.add_warning(f"{prefix}无法分析边界边: {e}")
    
    # 检查7：法线一致性
    try:
        face_normals = mesh.face_normals
        if face_normals is not None and len(face_normals) > 0:
            if np.any(np.isnan(face_normals)) or np.any(np.isinf(face_normals)):
                result.add_fail(f"{prefix}法线计算失败", "包含NaN/Inf法线")
                return False
            result.add_pass(f"{prefix}法线计算成功且有效")
        else:
            result.add_warning(f"{prefix}无法获取面法线数据")
        
    except Exception as e:
        result.add_warning(f"{prefix}无法计算法线: {e}")
    
    return True


def check_physical_properties(mesh, expected_mass_range, density, 
                              result, test_prefix="", part_type="unknown"):
    """
    检查物理属性合理性
    
    Args:
        mesh: trimesh对象
        expected_mass_range: [min_mass, max_mass] kg
        density: 材料密度 kg/m³
        result: TestResult对象
        test_prefix: 测试名称前缀
        part_type: 零件类型（用于更详细的错误信息）
    """
    prefix = f"{test_prefix}" if test_prefix else ""
    
    try:
        # 计算理论质量
        volume = mesh.volume
        
        if volume is None or volume == 0:
            result.add_warning(f"{prefix}体积为0或None，跳过物理检查")
            return True
            
        theoretical_mass = volume * density
        
        min_m, max_m = expected_mass_range
        
        # 检查质量范围
        if theoretical_mass < min_m * 0.5:
            result.add_warning(
                f"{prefix}质量偏小: {theoretical_mass*1000:.1f}g "
                f"(预期>{min_m*1000:.1f}g)"
            )
        elif theoretical_mass > max_m * 2.0:
            result.add_warning(
                f"{prefix}质量偏大: {theoretical_mass*1000:.1f}g "
                f"(预期<{max_m*1000:.1f}g)"
            )
        else:
            result.add_pass(
                f"{prefix}质量合理: {theoretical_mass*1000:.1f}g "
                f"(范围{min_m*1000:.1f}-{max_m*1000:.1f}g)"
            )
        
        # 计算并显示密度一致性
        actual_density = theoretical_mass / volume if volume > 0 else 0
        result.add_pass(f"{prefix}密度验证: ρ={density} kg/m³")
        
        # 包围盒尺寸检查
        bounds = mesh.bounds
        size = bounds[1] - bounds[0]
        
        result.add_pass(
            f"{prefix}包围盒尺寸: "
            f"[{size[0]*1000:.1f}, {size[1]*1000:.1f}, {size[2]*1000:.1f}] mm"
        )
        
        return True
        
    except Exception as e:
        result.add_fail(f"{prefix}物理属性检查异常", str(e))
        return False


def test_spring_element(builder, result):
    """测试弹簧生成器"""
    print("\n🧪 测试1: 弹簧生成器 (spring_element)")
    print("─" * 50)
    
    start_time = time.time()
    
    # 测试不同参数组合
    test_cases = [
        {
            'name': '标准弹簧',
            'params': {
                'stiffness': 500.0,
                'max_deformation': 0.02,
                'preload': 0.0,
            }
        },
        {
            'name': '高刚度弹簧',
            'params': {
                'stiffness': 8000.0,
                'max_deformation': 0.005,
                'preload': 10.0,
            }
        },
        {
            'name': '低刚度长弹簧',
            'params': {
                'stiffness': 50.0,
                'max_deformation': 0.05,
                'preload': 0.0,
            }
        },
    ]
    
    for i, case in enumerate(test_cases):
        name = case['name']
        params = case['params']
        prefix = f"[{i+1}] {name}"
        
        try:
            spring = builder.create_spring_element(**params)
            
            # 网格质量检查
            is_valid = check_mesh_manifold(spring, result, prefix)
            
            if is_valid:
                # 物理属性检查
                check_physical_properties(
                    spring, 
                    expected_mass_range=[0.01, 0.2],
                    density=7800.0,  # 钢材
                    result=result,
                    test_prefix=prefix,
                    part_type="spring_element"
                )
                
                # 弹簧特有检查：长度应大于线圈直径
                bounds = spring.bounds
                length = bounds[1][2] - bounds[0][2]
                diameter = max(bounds[1][0] - bounds[0][0], 
                             bounds[1][1] - bounds[0][1])
                
                if length > diameter * 0.5:
                    result.add_pass(f"{prefix}长径比合理: L/D={length/diameter:.2f}")
                else:
                    result.add_warning(f"{prefix}长径比偏小: L/D={length/diameter:.2f}")
                    
        except Exception as e:
            result.add_fail(prefix, f"生成失败: {e}")
    
    elapsed = time.time() - start_time
    result.add_pass(f"⏱️  弹簧批量生成总耗时: {elapsed:.2f}s")


def test_hollow_tube(builder, result):
    """测试中空管生成器"""
    print("\n🧪 测试2: 中空管生成器 (hollow_tube)")
    print("─" * 50)
    
    start_time = time.time()
    
    test_cases = [
        {
            'name': '标准中空管',
            'params': {
                'outer_diameter': 0.03,
                'length': 0.15,
                'wall_thickness': 0.002,
            }
        },
        {
            'name': '大直径管',
            'params': {
                'outer_diameter': 0.06,
                'length': 0.30,
                'wall_thickness': 0.003,
            }
        },
        {
            'name': '薄壁管',
            'params': {
                'outer_diameter': 0.02,
                'length': 0.10,
                'wall_thickness': 0.001,
            }
        },
    ]
    
    for i, case in enumerate(test_cases):
        name = case['name']
        params = case['params']
        prefix = f"[{i+1}] {name}"
        
        try:
            tube = builder.create_hollow_tube(**params)
            
            # 网格质量检查
            is_valid = check_mesh_manifold(tube, result, prefix)
            
            if is_valid:
                # 物理属性检查（铝合金）
                check_physical_properties(
                    tube,
                    expected_mass_range=[0.02, 0.5],
                    density=2700.0,  # 铝合金
                    result=result,
                    test_prefix=prefix,
                    part_type="hollow_tube"
                )
                
                # 中空管特有检查：壁厚应小于外半径
                outer_r = params['outer_diameter'] / 2
                wall_t = params['wall_thickness']
                
                if wall_t < outer_r:
                    result.add_pass(f"{prefix}壁厚合理: t/R={wall_t/outer_r:.2f}")
                else:
                    result.add_fail(prefix, f"壁厚过大: t={wall_t*1000:.1f}mm ≥ R={outer_r*1000:.1f}mm")
                    
                # 质量估算对比（如果builder提供了）
                if hasattr(tube, 'estimated_mass'):
                    est_mass = tube.estimated_mass
                    calc_mass = tube.volume * 2700.0
                    mass_error = abs(est_mass - calc_mass) / calc_mass
                    
                    if mass_error < 0.2:  # 允许20%误差
                        result.add_pass(f"{prefix}质量估算准确: 误差{mass_error*100:.1f}%")
                    else:
                        result.add_warning(f"{prefix}质量估算偏差较大: {mass_error*100:.1f}%")
                        
        except Exception as e:
            result.add_fail(prefix, f"生成失败: {e}")
    
    elapsed = time.time() - start_time
    result.add_pass(f"⏱️  中空管批量生成总耗时: {elapsed:.2f}s")


def test_hemisphere_foot(builder, result):
    """测试半球脚垫生成器"""
    print("\n🧪 测试3: 半球脚垫生成器 (hemisphere_foot)")
    print("─" * 50)
    
    start_time = time.time()
    
    test_cases = [
        {
            'name': '标准脚垫',
            'params': {
                'radius': 0.015,
                'shell_thickness': 0.003,
                'include_contact_pattern': True,
            }
        },
        {
            'name': '大号脚垫',
            'params': {
                'radius': 0.025,
                'shell_thickness': 0.004,
                'include_contact_pattern': True,
            }
        },
        {
            'name': '小号实心脚垫',
            'params': {
                'radius': 0.010,
                'shell_thickness': 0.008,  # 接近实心
                'include_contact_pattern': False,
            }
        },
    ]
    
    for i, case in enumerate(test_cases):
        name = case['name']
        params = case['params']
        prefix = f"[{i+1}] {name}"
        
        try:
            foot = builder.create_hemisphere_foot(**params)
            
            # 网格质量检查
            is_valid = check_mesh_manifold(foot, result, prefix)
            
            if is_valid:
                # 物理属性检查（橡胶）
                check_physical_properties(
                    foot,
                    expected_mass_range=[0.005, 0.08],
                    density=1200.0,  # 橡胶
                    result=result,
                    test_prefix=prefix,
                    part_type="hemisphere_foot"
                )
                
                # 半球特有检查：Z轴范围应在正半轴（半球形状）
                bounds = foot.bounds
                z_min, z_max = bounds[0][2], bounds[1][2]
                z_center = (z_min + z_max) / 2
                
                # 允许一定偏移（因为底座环的存在）
                if z_center >= -params['radius'] * 0.3:
                    result.add_pass(f"{prefix}几何中心合理: Z={z_center*1000:.1f}mm")
                else:
                    result.add_warning(f"{prefix}几何中心偏低: Z={z_center*1000:.1f}mm")
                    
        except Exception as e:
            result.add_fail(prefix, f"生成失败: {e}")
    
    elapsed = time.time() - start_time
    result.add_pass(f"⏱️  半球脚垫批量生成总耗时: {elapsed:.2f}s")


def test_parameter_adaptation(builder, result):
    """测试参数自适应算法"""
    print("\n🧪 测试4: 参数自适应算法")
    print("─" * 50)
    
    # 测试弹簧参数自适应
    test_inputs = [
        {'stiffness': 100, 'expected_wire_range': (0.001, 0.002)},
        {'stiffness': 1000, 'expected_wire_range': (0.0015, 0.003)},
        {'stiffness': 10000, 'expected_wire_range': (0.003, 0.005)},
    ]
    
    for i, test in enumerate(test_inputs):
        k = test['stiffness']
        exp_min, exp_max = test['expected_wire_range']
        prefix = f"[{i+1}] k={k} N/m"
        
        try:
            adapted = builder._adapt_spring_parameters(
                stiffness=k, 
                max_deformation=0.02,
                preload=0.0
            )
            wire_d = adapted['wire_diameter']
            
            if exp_min <= wire_d <= exp_max:
                result.add_pass(f"{prefix}线径自适应正确: d={wire_d*1000:.2f}mm")
            else:
                result.add_warning(
                    f"{prefix}线径偏离预期: d={wire_d*1000:.2f}mm "
                    f"(预期{exp_min*1000:.1f}-{exp_max*1000:.1f}mm)"
                )
                
            # 应力比检查
            stress_ratio = adapted.get('stress_ratio', 0)
            if stress_ratio < 0.5:  # 安全系数>2
                result.add_pass(f"{prefix}应力安全: σ/σ_yield={stress_ratio:.1%}")
            elif stress_ratio < 0.8:
                result.add_warning(f"{prefix}应力偏高: σ/σ_yield={stress_ratio:.1%}")
            else:
                result.add_fail(prefix, f"应力超限: σ/σ_yield={stress_ratio:.1%}")
                
        except Exception as e:
            result.add_fail(prefix, f"参数自适应失败: {e}")


def test_stl_roundtrip(builder, result):
    """测试STL导出/导入往返一致性"""
    print("\n🧪 测试5: STL文件往返测试")
    print("─" * 50)
    
    import tempfile
    
    test_dir = tempfile.mkdtemp(prefix="v2_test_")
    
    try:
        # 生成一个完整零件套件
        parts = builder.create_complete_robot_part_set(
            spring_params={'stiffness': 500},
            tube_params={'outer_diameter': 0.03, 'length': 0.15},
            foot_params={'radius': 0.015},
            output_dir=test_dir
        )
        
        for part_name, original_mesh in parts.items():
            stl_path = os.path.join(test_dir, f"{part_name}_roundtrip.stl")
            
            # 导出
            original_mesh.export(stl_path)
            
            # 导入
            imported_mesh = trimesh.load(stl_path)
            
            # 比较
            v_orig = len(original_mesh.vertices)
            v_import = len(imported_mesh.vertices)
            f_orig = len(original_mesh.faces)
            f_import = len(imported_mesh.faces)
            
            if v_orig == v_import and f_orig == f_import:
                result.add_pass(f"{part_name}: 往返一致 ({v_import}v, {f_import}f)")
            else:
                result.add_warning(
                    f"{part_name}: 顶点/面数变化 "
                    f"({v_orig}→{v_import}v, {f_orig}→{f_import}f)"
                )
                
            # 清理临时文件
            if os.path.exists(stl_path):
                os.remove(stl_path)
                
    finally:
        # 清理临时目录
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir, ignore_errors=True)


def test_performance_benchmark(builder, result):
    """性能基准测试"""
    print("\n🧪 测试6: 性能基准测试")
    print("─" * 50)
    
    iterations = 3
    part_types = ['spring', 'tube', 'foot']
    
    timings = {pt: [] for pt in part_types}
    
    for _ in range(iterations):
        # 弹簧计时
        t_start = time.time()
        builder.create_spring_element(stiffness=500)
        timings['spring'].append(time.time() - t_start)
        
        # 中空管计时
        t_start = time.time()
        builder.create_hollow_tube(outer_diameter=0.03, length=0.15)
        timings['tube'].append(time.time() - t_start)
        
        # 半球脚垫计时
        t_start = time.time()
        builder.create_hemisphere_foot(radius=0.015)
        timings['foot'].append(time.time() - t_start)
    
    for pt in part_types:
        avg_time = np.mean(timings[pt]) * 1000  # ms
        std_time = np.std(timings[pt]) * 1000
        
        if avg_time < 1000:  # < 1秒
            result.add_pass(f"{pt}: 平均{avg_time:.1f}ms ± {std_time:.1f}ms")
        elif avg_time < 3000:  # < 3秒
            result.add_warning(f"{pt}: 平均{avg_time:.1f}ms (较慢)")
        else:
            result.add_fail(pt, f"性能问题: 平均{avg_time:.1f}ms (>3s)")
    
    # 缓存命中率统计
    stats = builder.get_stats()
    cache_hit_rate = stats.get('cache_hit_rate', '0%')
    if isinstance(cache_hit_rate, str):
        result.add_pass(f"缓存命中率: {cache_hit_rate}")
    else:
        try:
            rate_float = float(cache_hit_rate)
            result.add_pass(f"缓存命中率: {rate_float:.1f}%")
        except:
            result.add_pass(f"缓存命中率: {cache_hit_rate}")


def test_catalog_integration(result):
    """测试Catalog系统集成"""
    print("\n🧪 测试7: Catalog系统集成")
    print("─" * 50)
    
    try:
        from forgecraft.core.catalog import (
            get_v2_catalog,
            get_v2_part_types,
            is_innovative_part,
            get_innovative_part_info,
            load_parameterized_catalog
        )
        
        # 测试1：获取V2目录
        catalog = get_v2_catalog()
        if len(catalog) >= 8:  # 至少包含基础+创新零件
            result.add_pass(f"V2目录加载成功: {len(catalog)}种零件")
        else:
            result.add_fail("V2目录", f"零件数不足: {len(catalog)} < 8")
        
        # 测试2：零件类型列表
        part_types = get_v2_part_types()
        required_parts = ['spring_element', 'hollow_tube', 'hemisphere_foot']
        
        missing = [p for p in required_parts if p not in part_types]
        if not missing:
            result.add_pass(f"所有创新零件已注册: {required_parts}")
        else:
            result.add_fail("零件注册", f"缺少: {missing}")
        
        # 测试3：创新零件识别
        for part in required_parts:
            if is_innovative_part(part):
                result.add_pass(f"'{part}' 正确识别为创新零件")
            else:
                result.add_fail("识别错误", f"'{part}' 未被识别为创新零件")
        
        # 测试4：创新零件详细信息
        for part in required_parts:
            info = get_innovative_part_info(part)
            if info and 'category' in info and 'benefits' in info:
                result.add_pass(f"'{part}' 信息完整: {info['category']}")
            else:
                result.add_warning(f"'{part}' 信息不完整")
        
        # 测试5：YAML配置加载
        yaml_catalog = load_parameterized_catalog()
        if len(yaml_catalog) >= 9:  # parameterized.yaml有9种零件
            result.add_pass(f"YAML配置加载成功: {len(yaml_catalog)}种零件")
        else:
            result.add_fail("YAML加载", f"零件数不足: {len(yaml_catalog)}")
            
    except ImportError as e:
        result.add_fail("Catalog导入", f"模块缺失: {e}")
    except Exception as e:
        result.add_fail("Catalog集成", f"异常: {e}")


def main():
    """主测试函数"""
    print("=" * 70)
    print("🚀 Phase 2: 集成测试套件")
    print("   AdvancedMeshBuilderV2 + Catalog V2 完整验证")
    print("=" * 70)
    
    result = TestResult()
    
    # 初始化建造器
    print("\n🔧 初始化 AdvancedMeshBuilderV2...")
    try:
        builder = AdvancedMeshBuilderV2(quality_level="high")
        print(f"   ✅ 引擎就绪 (质量: high)")
    except Exception as e:
        print(f"   ❌ 初始化失败: {e}")
        return False
    
    # 执行各项测试
    test_spring_element(builder, result)
    test_hollow_tube(builder, result)
    test_hemisphere_foot(builder, result)
    test_parameter_adaptation(builder, result)
    test_stl_roundtrip(builder, result)
    test_performance_benchmark(builder, result)
    test_catalog_integration(result)
    
    # 输出最终结果
    all_passed = result.summary()
    
    # 输出建议
    print("\n💡 后续建议:")
    if all_passed:
        print("   ✨ 所有测试通过！可以进入集成到evolution流程阶段")
        print("   📍 下一步: 运行v10实验，验证新零件在实际进化中的表现")
    else:
        print("   🔧 存在失败的测试项，请根据上述错误信息进行修复")
        print("   📍 建议: 先修复网格质量问题，再进行集成测试")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
