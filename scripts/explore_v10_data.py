#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v10 实验数据深度探索工具
========================
功能: 深入分析checkpoint.pkl的内部结构，提取详细的零件使用信息
"""

import pickle
import json
from pathlib import Path
from typing import Dict, List, Any, Optional


def explore_checkpoint_structure(data: Dict[str, Any], max_depth: int = 3, prefix: str = ""):
    """递归探索数据结构"""
    if max_depth <= 0:
        print(f"{prefix}[...]")
        return
    
    if isinstance(data, dict):
        for key, value in data.items():
            print(f"{prefix}📁 {key}: ", end="")
            
            if isinstance(value, (dict, list)):
                print()
                explore_checkpoint_structure(value, max_depth - 1, prefix + "  ")
            else:
                print(f"{type(value).__name__} = {repr(value)[:100]}")
    
    elif isinstance(data, list):
        if len(data) == 0:
            print(f"{prefix}(空列表)")
        else:
            print(f"{prefix}[列表，长度={len(data)}]")
            if len(data) > 0:
                print(f"{prefix}  [0]: ", end="")
                explore_checkpoint_structure(data[0], max_depth - 1, prefix + "    ")


def extract_detailed_part_info(data: Dict[str, Any]) -> Dict[str, Any]:
    """提取详细的零件信息"""
    result = {
        'generations': [],
        'part_usage_by_generation': [],
        'innovative_parts_usage': {
            'spring_element': [],
            'hollow_tube': [],
            'hemisphere_foot': [],
        },
        'best_individuals': [],
    }
    
    # 尝试不同的键名
    generations = None
    
    if 'generations' in data:
        generations = data['generations']
    elif 'history' in data:
        generations = data['history']
    
    if not generations:
        print("⚠️ 未找到世代数据")
        return result
    
    print(f"\n📊 找到 {len(generations)} 代数据\n")
    
    for gen_idx, gen_data in enumerate(generations):
        gen_info = {
            'generation': gen_idx,
            'best_fitness': None,
            'population_size': 0,
            'parts_detail': [],
        }
        
        # 提取最佳适应度
        if isinstance(gen_data, dict):
            gen_info['best_fitness'] = gen_data.get('best_fitness', 
                                                   gen_data.get('best', 
                                                              gen_data.get('best_individual', {}).get('fitness')))
            
            population = gen_data.get('population', [])
            gen_info['population_size'] = len(population)
            
            # 遍历种群中的每个个体
            for ind_idx, individual in enumerate(population):
                ind_info = extract_individual_info(individual, gen_idx, ind_idx)
                
                if ind_info:
                    gen_info['parts_detail'].append(ind_info)
                    
                    # 检查是否是最佳个体
                    is_best = (ind_info.get('fitness') == gen_info['best_fitness'])
                    if is_best:
                        result['best_individuals'].append({
                            **ind_info,
                            'generation': gen_idx,
                        })
                    
                    # 统计创新零件
                    for part_type in ['spring_element', 'hollow_tube', 'hemisphere_foot']:
                        if part_type in ind_info.get('part_types', []):
                            result['innovative_parts_usage'][part_type].append({
                                'generation': gen_idx,
                                'individual': ind_idx,
                                'fitness': ind_info.get('fitness'),
                                **{k: v for k, v in ind_info.items() if k in ['n_parts', 'n_motors']},
                            })
        
        result['generations'].append(gen_info)
        
        # 简化的世代统计
        part_counts = {}
        for detail in gen_info['parts_detail']:
            for pt in detail.get('part_types', []):
                part_counts[pt] = part_counts.get(pt, 0) + 1
        
        result['part_usage_by_generation'].append({
            'generation': gen_idx,
            'best_fitness': gen_info['best_fitness'],
            'part_counts': part_counts,
        })
    
    return result


def extract_individual_info(individual: Any, gen_idx: int, ind_idx: int) -> Optional[Dict]:
    """提取单个个体的详细信息"""
    if not isinstance(individual, dict):
        return None
    
    info = {
        'generation': gen_idx,
        'individual_index': ind_idx,
        'fitness': individual.get('fitness'),
        'n_parts': 0,
        'n_motors': individual.get('n_motors', 0),
        'manufacturability': individual.get('manufacturability', 1.0),
        'part_types': [],
        'raw_data_keys': list(individual.keys()),
    }
    
    # 提取零件信息 - 尝试多种可能的路径
    parts = None
    
    # 路径1: 直接在individual中
    for key in ['parts', 'body_parts', 'components', 'morphology']:
        if key in individual:
            parts = individual[key]
            break
    
    # 路径2: 在body子对象中
    if parts is None and 'body' in individual:
        body = individual['body']
        if isinstance(body, dict):
            for key in ['parts', 'body_parts', 'components']:
                if key in body:
                    parts = body[key]
                    break
    
    # 路径3: 在morphology中
    if parts is None and 'morphology' in individual:
        morphology = individual['morphology']
        if isinstance(morphology, dict):
            parts = morphology.get('parts', morphology.get('components'))
    
    if parts and isinstance(parts, (list, tuple)):
        info['n_parts'] = len(parts)
        
        for part in parts:
            if isinstance(part, dict):
                # 尝试提取零件类型
                part_type = (
                    part.get('type') or 
                    part.get('part_type') or 
                    part.get('name') or
                    part.get('category') or
                    'unknown'
                )
                info['part_types'].append(part_type)
            elif isinstance(part, str):
                info['part_types'].append(part)
    
    return info


def main():
    checkpoint_path = 'checkpoint.pkl'
    
    print("=" * 70)
    print("🔬 v10 Checkpoint 数据深度探索工具")
    print("=" * 70)
    
    # 加载数据
    print(f"\n📂 加载断点文件: {checkpoint_path}")
    with open(checkpoint_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"✅ 成功加载数据\n")
    
    # 探索顶层结构
    print("=" * 70)
    print("📋 顶层结构概览:")
    print("=" * 70)
    explore_checkpoint_structure(data, max_depth=2)
    
    # 详细提取零件信息
    print("\n" + "=" * 70)
    print("🔍 提取详细零件信息...")
    print("=" * 70)
    
    detailed_info = extract_detailed_part_info(data)
    
    # 输出结果
    print("\n" + "=" * 70)
    print("📊 分析结果汇总")
    print("=" * 70)
    
    # 1. 总体统计
    total_gens = len(detailed_info['generations'])
    best_fitness_overall = 0
    best_gen = 0
    
    for gen in detailed_info['generations']:
        if gen['best_fitness'] and gen['best_fitness'] > best_fitness_overall:
            best_fitness_overall = gen['best_fitness']
            best_gen = gen['generation']
    
    print(f"\n✅ 总世代数: {total_gens}")
    print(f"✅ 最佳适应度: {best_fitness_overall:.4f} (Gen {best_gen})")
    
    # 2. 创新零件使用统计
    print(f"\n{'='*70}")
    print("🔬 创新零件使用详情")
    print(f"{'='*70}")
    
    for part_name, usage_list in detailed_info['innovative_parts_usage'].items():
        emoji = {'spring_element': '🔧', 'hollow_tube': '🔩', 'hemisphere_foot': '🦶'}.get(part_name, '❓')
        
        print(f"\n{emoji} {part_name}:")
        print(f"   使用次数: {len(usage_list)}")
        
        if usage_list:
            fitnesses = [u['fitness'] for u in usage_list if u.get('fitness')]
            if fitnesses:
                print(f"   平均适应度: {sum(fitnesses)/len(fitnesses):.4f}")
                print(f"   最高适应度: {max(fitnesses):.4f}")
                print(f"   最低适应度: {min(fitnesses):.4f}")
            
            # 显示前5个使用实例
            print(f"   使用实例(前5个):")
            for i, usage in enumerate(usage_list[:5]):
                print(f"     [{i+1}] Gen {usage['generation']:2d}, Ind {usage['individual']:2d}, "
                      f"Fitness={usage.get('fitness', 'N/A')}")
        else:
            print("   ⚠️ 未检测到使用记录")
    
    # 3. 最佳个体列表
    print(f"\n{'='*70}")
    print("🏆 最佳个体演变历程")
    print(f"{'='*70}")
    
    if detailed_info['best_individuals']:
        print(f"\n| Gen | Fitness | Parts | Motors | Mfg | Part Types |")
        print(f"|-----|---------|-------|--------|-----|------------|")
        
        for bi in detailed_info['best_individuals'][:15]:  # 只显示前15个
            part_types_str = ', '.join(bi.get('part_types', [])[:5])
            if len(bi.get('part_types', [])) > 5:
                part_types_str += f"... (+{len(bi['part_types'])-5} more)"
            
            is_global_best = bi['fitness'] == best_fitness_overall
            marker = " 🏆" if is_global_best else ""
            
            print(f"| {bi['generation']:3d} | {bi['fitness'] or 0:.4f}{marker} | "
                  f"{bi['n_parts']:5d} | {bi['n_motors']:6d} | {bi['manufacturability']:.2f} | "
                  f"{part_types_str[:40]} |")
    
    # 4. 各世代零件使用趋势
    print(f"\n{'='*70}")
    print("📈 各世代零件使用趋势 (每5代采样)")
    print(f"{'='*70}")
    
    for gen_usage in detailed_info['part_usage_by_generation'][::5]:
        gen_num = gen_usage['generation']
        fitness = gen_usage['best_fitness']
        counts = gen_usage['part_counts']
        
        if counts:
            top_parts = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:5]
            parts_str = ', '.join([f"{p}({c})" for p, c in top_parts])
        else:
            parts_str = "(无数据)"
        
        print(f"Gen {gen_num:2d}: Best={fitness or 0:.4f} | {parts_str}")
    
    # 保存完整数据到JSON
    output_file = 'v10_detailed_analysis.json'
    
    # 序列化时处理不能JSON编码的对象
    def make_serializable(obj):
        if isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_serializable(i) for i in obj]
        else:
            return str(obj)
    
    serializable_data = make_serializable(detailed_info)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 详细数据已保存至: {output_file}")


if __name__ == '__main__':
    main()
