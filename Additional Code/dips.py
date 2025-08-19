import os
import numpy as np

def find_dip(predictions, dates, threshold=0.2, min_length=30):
    """
    Findet den längsten zusammenhängenden Dip unterhalb des Schwellenwerts.

    Args:
        predictions (np.ndarray): Vorhersagewerte.
        dates (np.ndarray): Zeitstempel.
        threshold (float): Schwelle für "Dip".
        min_length (int): Mindestlänge des Dips.

    Returns:
        dip_start_index, dip_start_date, dip_level, dip_high
    """
    below_threshold = predictions < threshold
    dips = []
    count = 0
    start_idx = None

    for i, is_below in enumerate(below_threshold):
        if is_below:
            if count == 0:
                start_idx = i
            count += 1
        else:
            if count >= min_length:
                dips.append((start_idx, i))  # Start und Ende merken
            count = 0

    # auch am Ende prüfen
    if count >= min_length:
        dips.append((start_idx, len(predictions)))

    if dips:
        # den längsten auswählen
        longest = max(dips, key=lambda x: x[1] - x[0])
        dip_start_index = longest[0]
        dip_values = predictions[longest[0]:longest[1]]
        dip_start_date = dates[dip_start_index]
        dip_level = np.mean(dip_values)
        dip_high = np.max(dip_values)
        return dip_start_index, dip_start_date, dip_level, dip_high
    else:
        return None, None, None, None



def analyze_dips_in_models(base_dir='models', subfolders=['benchmark','not_nit','not_lyser','low_input']):
    results = {}

    for folder in subfolders:
        pred_path = os.path.join(base_dir, folder, 'predictions_full.npy')
        date_path = os.path.join(base_dir, folder, 'dates_full.npy')

        if os.path.exists(pred_path) and os.path.exists(date_path):
            preds = np.load(pred_path)
            dates = np.load(date_path)

            dip_idx, dip_date, dip_level, dip_high = find_dip(preds, dates)

            results[folder] = {
                'dip_start_index': dip_idx,
                'dip_start_date': str(dip_date) if dip_date is not None else None,
                'dip_prediction_level': dip_level,
                'dip_high': dip_high
            }
        else:
            results[folder] = 'File(s) not found'

    return results

results = analyze_dips_in_models()

for model, info in results.items():
    print(f"Modell: {model}")
    if isinstance(info, dict):
        print(f"  Startindex:         {info['dip_start_index']}")
        print(f"  Startzeitpunkt:     {info['dip_start_date']}")
        if info['dip_prediction_level'] is not None:
            print(f"  Dip-Vorhersagewert-Mittel: {info['dip_prediction_level']:.3f}")
            print(f"  Dip-Vorhersagewert-Max: {info['dip_high']:.3f}")
        else:
            print("  Dip-Vorhersagewert: Nicht gefunden")
    else:
        print(f"  {info}")
    print("-" * 40)

