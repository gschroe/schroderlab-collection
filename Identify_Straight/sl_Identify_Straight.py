import mrcfile
import numpy as np
from scipy.signal import correlate2d
from scipy.stats import pearsonr
import os
import argparse
from typing import List, Tuple, Optional

def load_mrc_files(file_paths: List[str]) -> List[np.ndarray]:
    """
    Erste Funktion: Lädt mindestens 2 MRC-Dateien und gibt die 3D-Arrays zurück.
    
    Args:
        file_paths: Liste der Pfade zu den MRC-Dateien
    
    Returns:
        Liste von 3D numpy arrays (jeder enthält einen Stack von Bildern)
    """
    if len(file_paths) < 2:
        raise ValueError("Mindestens 2 MRC-Dateien müssen angegeben werden")
    
    mrc_stacks = []
    
    for file_path in file_paths:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Datei nicht gefunden: {file_path}")
        
        try:
            with mrcfile.open(file_path, mode='r') as mrc:
                # Daten als 3D array laden
                data = mrc.data.copy()
                mrc_stacks.append(data)
                print(f"Geladen: {file_path}, Shape: {data.shape}")
        
        except Exception as e:
            raise RuntimeError(f"Fehler beim Laden von {file_path}: {str(e)}")
    
    return mrc_stacks


def process_image_stack(image_stack: np.ndarray, n_pixels_crop: int, 
                       x_percent: float = 50.0) -> np.ndarray:
    """
    Zweite Funktion: Verarbeitet einen Bildstack durch Cropping und X-Projektion.
    
    Args:
        image_stack: 3D array (z, y, x) mit Bildstack
        n_pixels_crop: Anzahl Pixel die oben und unten weggeschnitten werden
        x_percent: Prozent der mittleren Pixel in X-Richtung für Projektion (default: 50%)
    
    Returns:
        3D array mit Shape (z, cropped_height, 1) - projizierte Bilder
    """
    if image_stack.ndim != 3:
        raise ValueError("Input muss ein 3D array sein (z, y, x)")
    
    z_size, y_size, x_size = image_stack.shape
    
    # Y-Cropping: n Pixel oben und unten entfernen
    if 2 * n_pixels_crop >= y_size:
        raise ValueError("Zu viele Pixel zum Croppen - Bild würde verschwinden")
    
    cropped_stack = image_stack[:, n_pixels_crop:y_size-n_pixels_crop, :]
    cropped_height = cropped_stack.shape[1]
    
    # Berechne mittlere X-Prozent für Projektion
    x_start = int(x_size * (100 - x_percent) / 200)
    x_end = int(x_size * (100 + x_percent) / 200)
    
    # X-Projektion: Summiere Pixel in mittleren X-Prozent
    projected_stack = np.zeros((z_size, cropped_height, 1))
    
    for z in range(z_size):
        # Summiere über die mittleren X-Pixel
        projection = np.sum(cropped_stack[z, :, x_start:x_end], axis=1)
        projected_stack[z, :, 0] = projection
    
    print(f"Stack verarbeitet: {image_stack.shape} -> {projected_stack.shape}")
    print(f"Y-Cropping: {n_pixels_crop} Pixel oben/unten entfernt")
    print(f"X-Projektion: Mittlere {x_percent}% verwendet (Pixel {x_start}:{x_end})")
    
    return projected_stack


def cross_correlation_analysis(projection_stack: np.ndarray, 
                             reference_stack: np.ndarray,
                             normalize: bool = True) -> np.ndarray:
    """
    Dritte Funktion: Berechnet normalisierte Kreuzkorrelation zwischen Projektionen 
    und Referenzbildern.
    
    Args:
        projection_stack: 3D array (z, height, 1) - projizierte Bilder aus Funktion 2
        reference_stack: 3D array (z, y, x) - Referenzbilder
        normalize: Ob die Kreuzkorrelation normalisiert werden soll
    
    Returns:
        2D Matrix (n_projections, n_references) mit Korrelationswerten
    """
    if projection_stack.ndim != 3 or projection_stack.shape[2] != 1:
        raise ValueError("Projection stack muss Form (z, height, 1) haben")
    
    if reference_stack.ndim != 3:
        raise ValueError("Reference stack muss ein 3D array sein")
    
    n_projections = projection_stack.shape[0]
    n_references = reference_stack.shape[0]
    
    # Ergebnismatrix initialisieren
    correlation_matrix = np.zeros((n_projections, n_references))
    
    print(f"Berechne Kreuzkorrelation: {n_projections} Projektionen x {n_references} Referenzen")
    
    for i in range(n_projections):
        # Extrahiere 1D Projektion
        projection = projection_stack[i, :, 0]
        
        for j in range(n_references):
            reference_img = reference_stack[j, :, :]
            
            # Berechne Kreuzkorrelation
            max_corr = calculate_max_correlation(projection, reference_img, normalize)
            correlation_matrix[i, j] = max_corr
        
        if (i + 1) % 10 == 0:  # Progress update
            print(f"Verarbeitet: {i + 1}/{n_projections} Projektionen")
    
    return correlation_matrix


def calculate_max_correlation(projection_1d: np.ndarray, 
                            image_2d: np.ndarray, 
                            normalize: bool = True) -> float:
    """
    Hilfsfunktion: Berechnet maximale Korrelation zwischen 1D Projektion und 2D Bild.
    
    Die 1D Projektion wird über das 2D Bild "geschoben" und die maximale 
    Korrelation wird zurückgegeben.
    """
    max_correlation = -np.inf
    
    # Verschiebe Projektion über jede Spalte des 2D Bildes
    for col in range(image_2d.shape[1]):
        column_data = image_2d[:, col]
        
        # Passe Längen an (nimm kürzere Länge)
        min_len = min(len(projection_1d), len(column_data))
        proj_cropped = projection_1d[:min_len]
        col_cropped = column_data[:min_len]
        
        if normalize:
            # Normalisierte Korrelation (Pearson)
            if np.std(proj_cropped) > 0 and np.std(col_cropped) > 0:
                corr, _ = pearsonr(proj_cropped, col_cropped)
                if not np.isnan(corr):
                    max_correlation = max(max_correlation, abs(corr))
        else:
            # Einfache Kreuzkorrelation
            correlation = np.corrcoef(proj_cropped, col_cropped)[0, 1]
            if not np.isnan(correlation):
                max_correlation = max(max_correlation, abs(correlation))
    
    return max_correlation if max_correlation != -np.inf else 0.0


def compare_reduced_to_all_stacks(projection_stack: np.ndarray, 
                                all_stacks: List[np.ndarray],
                                normalize: bool = True,
                                top_n_correlations: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vierte Funktion: Vergleicht jeden reduzierten Stack gegen alle bereitgestellten Stacks
    und gibt für jedes Bild im reduzierten Stack die prozentuale Zugehörigkeit zu jedem Stack zurück.
    
    Args:
        projection_stack: 3D array (z, height, 1) - projizierte Bilder aus Funktion 2
        all_stacks: Liste von 3D arrays - alle zu vergleichenden Stacks
        normalize: Ob die Kreuzkorrelation normalisiert werden soll
        top_n_correlations: Anzahl der höchsten Korrelationen pro Stack für Mittelwert (default: 5)
    
    Returns:
        Tuple von:
        - 1D array mit Overall-Fitness-Werten für jedes Bild im projection_stack
        - 2D array (n_projections, n_stacks) mit prozentualer Zugehörigkeit zu jedem Stack
    """
    if projection_stack.ndim != 3 or projection_stack.shape[2] != 1:
        raise ValueError("Projection stack muss Form (z, height, 1) haben")
    
    n_projections = projection_stack.shape[0]
    n_stacks = len(all_stacks)
    
    print(f"Vergleiche {n_projections} Projektionen gegen {n_stacks} Stacks")
    print(f"Verwende Top-{top_n_correlations} Korrelationen pro Stack für Durchschnitt")
    
    # Arrays für Ergebnisse
    fitness_values = np.zeros(n_projections)
    stack_percentages = np.zeros((n_projections, n_stacks))
    
    for proj_idx in range(n_projections):
        projection = projection_stack[proj_idx, :, 0]
        
        # Sammle Korrelationswerte für jeden Stack separat
        stack_correlations = []
        
        for stack_idx, stack in enumerate(all_stacks):
            correlations_for_this_stack = []
            
            for img_idx in range(stack.shape[0]):
                reference_img = stack[img_idx, :, :]
                corr = calculate_max_correlation(projection, reference_img, normalize)
                correlations_for_this_stack.append(corr)
            
            stack_correlations.append(correlations_for_this_stack)
        
        # Berechne Durchschnitt der Top-n Korrelationen pro Stack
        stack_averages = []
        for stack_corr in stack_correlations:
            if len(stack_corr) > 0:
                # Sortiere und nimm die top_n_correlations besten Werte
                sorted_corr = np.sort(stack_corr)[-top_n_correlations:]
                avg_corr = np.mean(sorted_corr)
                stack_averages.append(max(avg_corr, 0))  # Negative Korrelationen auf 0 setzen
            else:
                stack_averages.append(0)
        
        # Berechne prozentuale Verteilung
        total_correlation = sum(stack_averages)
        if total_correlation > 0:
            percentages = [avg / total_correlation * 100 for avg in stack_averages]
        else:
            # Falls alle Korrelationen 0 oder negativ sind, gleichmäßig verteilen
            percentages = [100.0 / n_stacks] * n_stacks
        
        # Speichere Ergebnisse
        fitness_values[proj_idx] = max(stack_averages)  # Beste durchschnittliche Korrelation
        stack_percentages[proj_idx, :] = percentages
        
        if (proj_idx + 1) % 10 == 0:
            print(f"Fitness berechnet für: {proj_idx + 1}/{n_projections} Projektionen")
    
    print(f"Fitness-Werte berechnet - Bereich: {np.min(fitness_values):.4f} bis {np.max(fitness_values):.4f}")
    return fitness_values, stack_percentages


def print_stack_analysis(stack_percentages: np.ndarray, fitness_values: np.ndarray, 
                        mrc_file_names: List[str], projection_stack_idx: int) -> None:
    """
    Hilfsfunktion: Druckt eine detaillierte Analyse der Stack-Zugehörigkeiten.
    """
    n_projections, n_stacks = stack_percentages.shape
    
    # Erstelle Stack-Namen (ohne den Projektions-Stack)
    stack_names = []
    stack_idx = 0
    for i, name in enumerate(mrc_file_names):
        if i != projection_stack_idx:
            stack_names.append(f"Stack_{stack_idx}({os.path.basename(name)})")
            stack_idx += 1
    
    print(f"\n=== STACK ZUGEHÖRIGKEITS-ANALYSE ===")
    print(f"Projektions-Stack: {os.path.basename(mrc_file_names[projection_stack_idx])}")
    print(f"Vergleichs-Stacks: {', '.join(stack_names)}")
    
    # Durchschnittliche Zugehörigkeit pro Stack
    avg_percentages = np.mean(stack_percentages, axis=0)
    print(f"\nDurchschnittliche Zugehörigkeit über alle Projektionen:")
    for i, (name, pct) in enumerate(zip(stack_names, avg_percentages)):
        print(f"  {name}: {pct:.1f}%")
    
    # Finde dominante Zugehörigkeiten
    dominant_stack = np.argmax(stack_percentages, axis=1)
    dominant_percentages = np.max(stack_percentages, axis=1)
    
    print(f"\nBilder mit stärkster Zugehörigkeit (>70%):")
    strong_matches = np.where(dominant_percentages > 70)[0]
    for img_idx in strong_matches[:10]:  # Zeige max. 10
        stack_idx = dominant_stack[img_idx]
        percentage = dominant_percentages[img_idx]
        fitness = fitness_values[img_idx]
        print(f"  Bild {img_idx}: {percentage:.1f}% → {stack_names[stack_idx]} (Fitness: {fitness:.3f})")
    
    if len(strong_matches) > 10:
        print(f"  ... und {len(strong_matches) - 10} weitere")
    
    # Statistiken pro Stack
    print(f"\nStatistiken pro Stack:")
    for i, name in enumerate(stack_names):
        stack_imgs = np.where(dominant_stack == i)[0]
        if len(stack_imgs) > 0:
            avg_pct = np.mean(stack_percentages[stack_imgs, i])
            print(f"  {name}: {len(stack_imgs)} Bilder dominant zugeordnet (Ø {avg_pct:.1f}%)")
        else:
            print(f"  {name}: 0 Bilder dominant zugeordnet")


def setup_argument_parser() -> argparse.ArgumentParser:
    """
    Setzt den Argument Parser für Kommandozeilen-Parameter auf.
    """
    parser = argparse.ArgumentParser(
        description="MRC Stack Processing und Cross-Correlation Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  python script.py stack1.mrc stack2.mrc stack3.mrc --projection-stack 0 --crop 10 --x-percent 60
  python script.py *.mrc --projection-stack 1 --crop 5 --x-percent 50 --output results.txt
        """
    )
    
    # Pflichtargumente
    parser.add_argument('mrc_files', nargs='+', 
                       help='Pfade zu den MRC-Dateien (mindestens 2)')
    
    # Optionale Argumente
    parser.add_argument('--projection-stack', '-p', type=int, default=0,
                       help='Index des Stacks für Projektionsberechnung (default: 0)')
    
    parser.add_argument('--crop', '-c', type=int, default=10,
                       help='Anzahl Pixel zum Croppen oben/unten (default: 10)')
    
    parser.add_argument('--x-percent', '-x', type=float, default=50.0,
                       help='Prozent der mittleren X-Pixel für Projektion (default: 50.0)')
    
    parser.add_argument('--top-n', '-n', type=int, default=5,
                       help='Anzahl der höchsten Korrelationen pro Stack für Durchschnitt (default: 5)')
    
    parser.add_argument('--output', '-o', type=str,
                       help='Ausgabedatei für Ergebnisse (optional)')
    
    parser.add_argument('--no-normalize', action='store_true',
                       help='Deaktiviert Normalisierung der Kreuzkorrelation')
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Ausführliche Ausgabe')
    
    return parser


def save_results(fitness_values: np.ndarray, stack_percentages: np.ndarray, 
                correlation_matrices: List[np.ndarray], output_path: str, 
                args: argparse.Namespace, mrc_file_names: List[str]) -> None:
    """
    Speichert die Ergebnisse inklusive Stack-Zugehörigkeiten in einer Datei.
    """
    # Erstelle Stack-Namen (ohne den Projektions-Stack)
    stack_names = []
    stack_idx = 0
    for i, name in enumerate(mrc_file_names):
        if i != args.projection_stack:
            stack_names.append(f"Stack_{stack_idx}({os.path.basename(name)})")
            stack_idx += 1
    
    with open(output_path, 'w') as f:
        f.write("# MRC Stack Analysis Results\n")
        f.write(f"# Projection stack: {os.path.basename(mrc_file_names[args.projection_stack])} (index: {args.projection_stack})\n")
        f.write(f"# Comparison stacks: {', '.join(stack_names)}\n")
        f.write(f"# Crop pixels: {args.crop}\n")
        f.write(f"# X percent: {args.x_percent}\n")
        f.write(f"# Number of stacks: {len(args.mrc_files)}\n")
        f.write(f"# Normalization: {not args.no_normalize}\n")
        f.write(f"# Top correlations used: {getattr(args, 'top_n', 5)}\n\n")
        
        # Header für die Haupttabelle
        f.write("# Detaillierte Ergebnisse pro Bild:\n")
        f.write("# Image_Index\tFitness_Value\t")
        for name in stack_names:
            f.write(f"{name}_Percent\t")
        f.write("Dominant_Stack\tDominant_Percent\n")
        
        # Schreibe Daten für jedes Bild
        for i in range(len(fitness_values)):
            dominant_stack_idx = np.argmax(stack_percentages[i, :])
            dominant_percent = stack_percentages[i, dominant_stack_idx]
            
            f.write(f"{i}\t{fitness_values[i]:.6f}\t")
            for j in range(len(stack_names)):
                f.write(f"{stack_percentages[i, j]:.2f}\t")
            f.write(f"{stack_names[dominant_stack_idx]}\t{dominant_percent:.2f}\n")
        
        # Statistiken
        f.write(f"\n# Fitness-Statistiken:\n")
        f.write(f"# Mean fitness: {np.mean(fitness_values):.6f}\n")
        f.write(f"# Std fitness: {np.std(fitness_values):.6f}\n")
        f.write(f"# Min fitness: {np.min(fitness_values):.6f}\n")
        f.write(f"# Max fitness: {np.max(fitness_values):.6f}\n")
        
        # Durchschnittliche Stack-Zugehörigkeit
        f.write(f"\n# Durchschnittliche Stack-Zugehörigkeit:\n")
        avg_percentages = np.mean(stack_percentages, axis=0)
        for i, (name, pct) in enumerate(zip(stack_names, avg_percentages)):
            f.write(f"# {name}: {pct:.2f}%\n")
        
        # Dominante Zugehörigkeiten
        dominant_stack = np.argmax(stack_percentages, axis=1)
        f.write(f"\n# Verteilung der dominanten Zugehörigkeiten:\n")
        for i, name in enumerate(stack_names):
            count = np.sum(dominant_stack == i)
            f.write(f"# {name}: {count} Bilder ({count/len(fitness_values)*100:.1f}%)\n")
        
        # Beste und schlechteste Bilder
        best_indices = np.argsort(fitness_values)[-5:][::-1]
        worst_indices = np.argsort(fitness_values)[:5]
        
        f.write(f"\n# Best 5 images:\n")
        for idx in best_indices:
            dominant_stack_idx = np.argmax(stack_percentages[idx, :])
            f.write(f"# {idx}: fitness={fitness_values[idx]:.6f}, dominant={stack_names[dominant_stack_idx]} ({stack_percentages[idx, dominant_stack_idx]:.1f}%)\n")
            
        f.write(f"\n# Worst 5 images:\n")
        for idx in worst_indices:
            dominant_stack_idx = np.argmax(stack_percentages[idx, :])
            f.write(f"# {idx}: fitness={fitness_values[idx]:.6f}, dominant={stack_names[dominant_stack_idx]} ({stack_percentages[idx, dominant_stack_idx]:.1f}%)\n")
    
    print(f"Detaillierte Ergebnisse gespeichert in: {output_path}")


def main():
    """
    Hauptfunktion mit Argument Parsing und vollständiger Pipeline
    """
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Validiere Argumente
    if len(args.mrc_files) < 2:
        parser.error("Mindestens 2 MRC-Dateien müssen angegeben werden")
    
    if args.projection_stack >= len(args.mrc_files):
        parser.error(f"Projection stack index {args.projection_stack} ist außerhalb des Bereichs (0-{len(args.mrc_files)-1})")
    
    if not (0 < args.x_percent <= 100):
        parser.error("X-Prozent muss zwischen 0 und 100 liegen")
    
    try:
        # 1. MRC-Dateien laden
        if args.verbose:
            print(f"Lade {len(args.mrc_files)} MRC-Dateien...")
        mrc_stacks = load_mrc_files(args.mrc_files)
        
        # 2. Projektions-Stack verarbeiten
        if args.verbose:
            print(f"Verwende Stack {args.projection_stack} für Projektionen")
        projection_stack = process_image_stack(
            mrc_stacks[args.projection_stack], 
            n_pixels_crop=args.crop,
            x_percent=args.x_percent
        )
        
        # 3. Erstelle Liste aller anderen Stacks (ohne den Projektions-Stack)
        comparison_stacks = [stack for i, stack in enumerate(mrc_stacks) if i != args.projection_stack]
        
        if args.verbose:
            print(f"Vergleiche gegen {len(comparison_stacks)} andere Stacks")
        
        # 4. Berechne Fitness-Werte und Stack-Zugehörigkeiten für jedes Bild im Projektions-Stack
        fitness_values, stack_percentages = compare_reduced_to_all_stacks(
            projection_stack, 
            comparison_stacks,
            normalize=not args.no_normalize,
            top_n_correlations=args.top_n
        )
        
        # 5. Zusätzlich: Detaillierte Kreuzkorrelations-Matrizen für jeden Stack
        correlation_matrices = []
        for i, stack in enumerate(comparison_stacks):
            if args.verbose:
                print(f"Berechne detaillierte Korrelationsmatrix für Stack {i+1}/{len(comparison_stacks)}")
            corr_matrix = cross_correlation_analysis(
                projection_stack, 
                stack,
                normalize=not args.no_normalize
            )
            correlation_matrices.append(corr_matrix)
        
        # 6. Ergebnisse ausgeben
        print(f"\n=== ERGEBNISSE ===")
        print(f"Anzahl Projektionen: {len(fitness_values)}")
        print(f"Fitness-Statistiken:")
        print(f"  Mittelwert: {np.mean(fitness_values):.4f}")
        print(f"  Standardabweichung: {np.std(fitness_values):.4f}")
        print(f"  Minimum: {np.min(fitness_values):.4f}")
        print(f"  Maximum: {np.max(fitness_values):.4f}")
        
        # Beste und schlechteste Bilder
        best_idx = np.argmax(fitness_values)
        worst_idx = np.argmin(fitness_values)
        print(f"  Bestes Bild: Index {best_idx} (Fitness: {fitness_values[best_idx]:.4f})")
        print(f"  Schlechtestes Bild: Index {worst_idx} (Fitness: {fitness_values[worst_idx]:.4f})")
        
        # Zeige Stack-Zugehörigkeits-Analyse
        print_stack_analysis(stack_percentages, fitness_values, args.mrc_files, args.projection_stack)
        
        # 7. Speichere Ergebnisse falls gewünscht
        if args.output:
            save_results(fitness_values, stack_percentages, correlation_matrices, args.output, args, args.mrc_files)
        
        return fitness_values, stack_percentages, correlation_matrices
        
    except Exception as e:
        print(f"Fehler: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return None, None, None
