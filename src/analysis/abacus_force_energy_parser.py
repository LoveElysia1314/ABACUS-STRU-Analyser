import os
import re
from typing import List, Tuple, Dict

def parse_running_md_log(log_path: str) -> Tuple[Dict[int, float], Dict[int, List[Tuple[str, float, float, float]]]]:
    energies = {}
    forces = {}
    current_frame = None
    current_forces = []
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 解析帧号
            if 'STEP OF MOLECULAR DYNAMICS' in line:
                match = re.search(r'STEP OF MOLECULAR DYNAMICS\s*:\s*(\d+)', line)
                if match:
                    current_frame = int(match.group(1))
                    current_forces = []
            # 解析能量
            elif 'final etot' in line and current_frame is not None:
                match = re.search(r'final etot is ([\-\d\.Ee]+) eV', line)
                if match:
                    energies[current_frame] = float(match.group(1))
            # 解析力
            elif 'TOTAL-FORCE' in line and current_frame is not None:
                current_forces = []
                next(f)  # 跳过分隔线
                for force_line in f:
                    if '-' * 10 in force_line or 'TOTAL-STRESS' in force_line:
                        break
                    parts = force_line.split()
                    if len(parts) == 4:
                        atom_label = parts[0]
                        fx, fy, fz = map(float, parts[1:])
                        current_forces.append((atom_label, fx, fy, fz))
                if current_forces:
                    forces[current_frame] = current_forces
    return energies, forces

def save_forces_to_csv(energies: Dict[int, float], forces: Dict[int, List[Tuple[str, float, float, float]]], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    for frame_id, energy in energies.items():
        csv_path = os.path.join(output_dir, f'frame_{frame_id}.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['atom', 'fx(eV/Angstrom)', 'fy(eV/Angstrom)', 'fz(eV/Angstrom)'])
            if frame_id in forces:
                for atom in forces[frame_id]:
                    writer.writerow([atom[0], atom[1], atom[2], atom[3]])
            writer.writerow([])
            writer.writerow(['final_etot(eV)', energy])

def main():
    log_file = os.path.join('OUT.ABACUS', 'running_md.log')
    output_dir = os.path.join('analysis_results', 'single_force_results')
    energies, forces = parse_running_md_log(log_file)
    save_forces_to_csv(energies, forces, output_dir)
    print(f'已将每帧能量和力保存到 {output_dir} 文件夹下的单帧csv文件中。')

if __name__ == '__main__':
    main()
