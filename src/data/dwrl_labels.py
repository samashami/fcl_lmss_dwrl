CLASSES = ["PET", "PP", "PE", "TETRA", "PS", "PVC", "Other"]
CLASS2IDX = {c: i for i, c in enumerate(CLASSES)}
IDX2CLASS = {i: c for c, i in CLASS2IDX.items()}
NUM_CLASSES = len(CLASSES)