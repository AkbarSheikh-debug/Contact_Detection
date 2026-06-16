import json, numpy as np

GT_FRAMES = [
    123, 187, 376, 452, 577, 586, 610, 768, 781, 856, 858, 933, 941,
    1326, 1391, 1408, 1659, 1670, 1851, 2014, 2139, 2153, 2192, 2374,
    2435, 2566, 2569, 2679, 2798, 2848, 2952, 3434, 3610, 3642, 3654, 3664,
    3684, 3700, 3704, 3724, 3736, 3863, 4160, 4249, 4302, 4314, 4316, 4456,
]

folder = r'C:\Users\XRIG\Downloads\sam3d_with_world_coords'
actions = json.load(open(folder + r'\fd7a77fd-588f-43ff-925f-ff5a648a246d.json'))['actions']

# For each GT frame, find which ASFormer actions cover it (ws-8 to we+12)
print('GT frame coverage by ASFormer windows (ws-8 to we+12):')
uncovered = []
for gf in GT_FRAMES:
    covering = [a for a in actions
                if (a['window_start'] - 8) <= gf <= (a['window_end'] + 12)]
    if not covering:
        print(f'  GT {gf:4d} -> UNCOVERED')
        uncovered.append(gf)
    else:
        best = min(covering, key=lambda a: abs(a['frame'] - gf))
        dist = abs(best['frame'] - gf)
        print(f'  GT {gf:4d} -> {len(covering)} window(s), best center={best["frame"]} dist={dist} conf={best["confidence"]:.2f}')

print(f'\nUncovered GT frames ({len(uncovered)}): {uncovered}')

# Close GT pairs that NMS cooldown must handle
print('\nClose GT pairs (< 20 frames apart):')
for i in range(len(GT_FRAMES) - 1):
    d = GT_FRAMES[i + 1] - GT_FRAMES[i]
    if d < 20:
        print(f'  frames {GT_FRAMES[i]} and {GT_FRAMES[i+1]} are {d} frames apart')
