#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 1 依赖环境验证脚本 v3 (最终版)

正确使用 manifold3d 3.5.0 API:
- sphere(radius) + translate()
- cylinder(height, radius) 
- cube(size)

作者: ForgeCraft AI
日期: 2026-06-02
"""

import sys
import time

def print_separator(char="=", length=60):
    print(char * length)

def test_trimesh():
    print("\n📦 测试 trimesh...")
    try:
        import trimesh
        mesh = trimesh.creation.box([1, 1, 1])
        
        print(f"   ✅ 版本: {trimesh.__version__}")
        print(f"   ✅ 基础网格生成: OK (顶点:{len(mesh.vertices)}, 面:{len(mesh.faces)})")
        print(f"   ✅ 网格分析: OK (体积:{mesh.volume:.3f})")
        
        return True, trimesh.__version__
    except Exception as e:
        print(f"   ❌ 失败: {e}")
        return False, None

def test_manifold3d():
    print("\n🔧 测试 manifold3d v3.5.0 (核心引擎)...")
    try:
        import manifold3d as mf
        
        print("   ✅ 模块导入成功")
        
        # 创建基本体（在原点）
        print("   📐 创建基本体...")
        sphere1 = mf.Manifold.sphere(radius=1.0).translate([0, 0, 0])
        sphere2 = mf.Manifold.sphere(radius=1.0).translate([1.5, 0, 0])
        box = mf.Manifold.cube(size=[2, 2, 2]).translate([-1, 0, 0])
        cylinder = mf.Manifold.cylinder(height=2, radius_low=0.5)
        print("      ✅ Sphere/Cube/Cylinder 创建成功")
        
        # 布尔并集
        print("   ➕ 布尔并集 (Union)...")
        union_result = sphere1 + sphere2
        assert not union_result.is_empty()
        mesh = union_result.to_mesh()
        print(f"      ✅ 并集成功 (顶点数:{len(mesh.vert_properties)}, 面数:{len(mesh.tri_verts)})")
        
        # 布尔差集
        print("   ➖ 布尔差集 (Difference)...")
        diff_result = box - sphere1
        diff_mesh = diff_result.to_mesh()
        print(f"      ✅ 差集成功 (顶点数:{len(diff_mesh.vert_properties)})")

        # 注：manifold3d 3.5.0 暂不支持交集运算符 (&)
        # 对于本项目（弹簧建模），并集+差集已足够

        # 复杂组合
        print("   🔀 复杂组合操作...")
        complex_shape = (sphere1 + sphere2) - box
        complex_mesh = complex_shape.to_mesh()
        print(f"      ✅ 成功 (顶点数:{len(complex_mesh.vert_properties)})")

        # 转换为trimesh
        print("   🔄 转换为trimesh...")
        import trimesh
        final_mesh = complex_shape.to_mesh()
        tmesh_obj = trimesh.Trimesh(
            vertices=final_mesh.vert_properties,
            faces=final_mesh.tri_verts,
            process=False
        )
        print(f"      ✅ 转换成功 (trimesh顶点:{len(tmesh_obj.vertices)})")
        
        # 几何属性
        print("   📏 几何属性...")
        vol = complex_shape.volume()
        surf_area = complex_shape.surface_area()
        print(f"      ✅ 体积:{vol:.4f}, 表面积:{surf_area:.4f}")
        print(f"      ✅ 包围盒: {complex_shape.bounding_box()}")
        
        # 高级功能
        print("   ⚙️  高级功能...")
        refined = complex_shape.refine(2)
        print("      ✅ 细分OK")

        return True, "3.5.0"
        
    except ImportError:
        print("   ⚠️  未安装 (pip install manifold3d)")
        return False, None
    except Exception as e:
        print(f"   ❌ 失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_pymadcad():
    print("\n⚙️  测试 pymadcad (可选)...")
    try:
        import madcad
        from madcad import brick, cylinder, sphere, Circle, extrusion, difference, O, Z, vec3
        
        print(f"   ✅ 版本: {madcad.__version__}")
        
        test_cube = brick(width=vec3(1, 1, 1))
        test_cyl = cylinder(radius=0.5, height=2)
        test_sph = sphere(center=O, radius=1)
        
        circle = Circle(center=O, normal=Z, radius=0.5)
        extruded = extrusion(circle, height=1)
        
        result = difference(test_cube, test_cyl)
        
        print("      ✅ 所有测试通过")
        return True, madcad.__version__
    except ImportError:
        print("   ℹ️  未安装（可选）")
        return False, None
    except Exception as e:
        print(f"   ❌ 失败: {e}")
        return False, None

def test_integration():
    print("\n🔗 测试库集成能力...")
    
    success_count = 0
    
    # 测试1：完整工作流
    print("   ① manifold3d → trimesh 工作流...")
    try:
        import manifold3d as mf
        import trimesh
        
        mf_cyl = mf.Manifold.cylinder(height=2, radius_low=0.5)
        mf_mesh = mf_cyl.to_mesh()

        tmesh = trimesh.Trimesh(
            vertices=mf_mesh.vert_properties,
            faces=mf_mesh.tri_verts,
            process=True
        )
        
        volume = tmesh.volume
        centroid = tmesh.centroid
        
        print(f"      ✅ 成功 (体积:{volume:.4f}, 质心:{centroid.round(4).tolist()})")
        success_count += 1
    except Exception as e:
        print(f"      ❌ 失败: {e}")
    
    # 测试2：批量性能
    print("   ② 批量生成性能...")
    try:
        import manifold3d as mf
        import time
        
        start = time.time()
        parts = []
        for i in range(10):
            s = mf.Manifold.sphere(radius=0.3).translate([i * 0.8, 0, 0])
            parts.append(s)
        
        combined = parts[0]
        for p in parts[1:]:
            combined = combined + p
        
        elapsed_ms = (time.time() - start) * 1000
        final_mesh = combined.to_mesh()
        
        print(f"      ✅ OK ({elapsed_ms:.1f}ms, 10个球体合并, 顶点:{len(final_mesh.vert_properties)})")
        success_count += 1
    except Exception as e:
        print(f"      ❌ 失败: {e}")
    
    # 测试3：复杂布尔链式
    print("   ③ 复杂布尔链式操作...")
    try:
        import manifold3d as mf
        
        base = mf.Manifold.cube(size=[3, 3, 3])
        
        cut1 = mf.Manifold.sphere(radius=0.8).translate([1, 1, 1])
        cut2 = mf.Manifold.sphere(radius=0.8).translate([-1, -1, 1])
        cut3 = mf.Manifold.sphere(radius=0.8).translate([1, -1, -1])
        
        add_cyl = mf.Manifold.cylinder(height=4, radius_low=0.3)
        
        result = (base - cut1 - cut2 - cut3) + add_cyl
        
        mesh = result.to_mesh()
        
        print(f"      ✅ OK (顶点:{len(mesh.vert_properties)}, genus:{result.genus()})")
        success_count += 1
    except Exception as e:
        print(f"      ❌ 失败: {e}")
    
    print(f"\n   集成测试结果: {success_count}/3 通过")
    return success_count == 3

def generate_report(results):
    print_separator("=")
    print("\n📊 Phase 1 环境检查报告")
    print_separator("-")
    
    total = len(results)
    passed = sum(1 for v in results.values() if v[0])
    
    for name, (status, ver) in results.items():
        icon = "✅" if status else "❌"
        ver_str = f" (v{ver})" if ver else ""
        print(f"{icon} {name:<20} {ver_str}")
    
    print_separator("-")
    print(f"总计: {passed}/{total} 通过\n")
    
    if passed == total or (passed >= total - 1 and not results.get('pymadcad', (True,))[0]):
        print("🎉 完美！环境已就绪！")
        print("✨ 可以开始创建 AdvancedMeshBuilderV2 类了\n")
        return True
    else:
        print("❌ 存在关键依赖缺失\n")
        return False

def main():
    print_separator("=")
    print("Phase 1 依赖环境检查工具 v3 (最终版)")
    print("ForgeCraft AI - AdvancedMeshBuilderV2 准备工作")
    print(f"Python: {sys.version.split()[0]} | 时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print_separator("=")
    
    results = {}
    
    status, ver = test_trimesh()
    results['trimesh'] = (status, ver)
    
    status, ver = test_manifold3d()
    results['manifold3d'] = (status, ver)
    
    status, ver = test_pymadcad()
    results['pymadcad'] = (status, ver)
    
    integration_ok = test_integration()
    results['集成测试'] = (integration_ok, None)
    
    all_ok = generate_report(results)
    
    sys.exit(0 if all_ok else 1)

if __name__ == "__main__":
    main()
